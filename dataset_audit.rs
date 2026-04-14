// Compile: rustc dataset_audit.rs -O -o dataset_audit
// Run: ./dataset_audit /path/to/data
// Exits nonzero if the observed split counts do not match Final Project.docx.

use std::env;
use std::fs::{self, File};
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::process;

const HDF5_MAGIC: [u8; 8] = [0x89, b'H', b'D', b'F', 0x0D, 0x0A, 0x1A, 0x0A];
const APPROX_TOTAL_SIZE_GB_DECIMAL: u64 = 52;

struct ExpectedSplit {
    name: &'static str,
    aliases: &'static [&'static str],
    expected_samples: usize,
}

const EXPECTED_SPLITS: [ExpectedSplit; 3] = [
    ExpectedSplit {
        name: "train",
        aliases: &["train"],
        expected_samples: 8_750,
    },
    ExpectedSplit {
        name: "valid",
        aliases: &["valid", "val"],
        expected_samples: 1_200,
    },
    ExpectedSplit {
        name: "test",
        aliases: &["test"],
        expected_samples: 1_300,
    },
];

#[derive(Default)]
struct SplitStats {
    directory: PathBuf,
    hdf5_files: usize,
    other_files: usize,
    invalid_hdf5_files: usize,
    total_bytes: u64,
    min_file_bytes: Option<u64>,
    max_file_bytes: Option<u64>,
}

fn main() {
    if let Err(err) = run() {
        eprintln!("[error] {err}");
        process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let root = parse_root_arg()?;
    let root = root
        .canonicalize()
        .map_err(|err| format!("failed to resolve dataset root {}: {err}", root.display()))?;

    println!("Dataset audit");
    println!("Root: {}", root.display());
    println!(
        "Expected split counts from Final Project.docx: train=8750, valid=1200, test=1300"
    );
    println!(
        "Expected total size from Final Project.docx: approximately {} GB",
        APPROX_TOTAL_SIZE_GB_DECIMAL
    );
    println!();

    let mut summaries = Vec::with_capacity(EXPECTED_SPLITS.len());
    let mut failed = false;

    for expected in EXPECTED_SPLITS {
        let split_dir = find_split_dir(&root, &expected)
            .ok_or_else(|| format!("missing split directory for {}", expected.name))?;
        let stats = scan_split(split_dir)
            .map_err(|err| format!("failed while scanning {}: {err}", expected.name))?;

        let count_ok = stats.hdf5_files == expected.expected_samples;
        let magic_ok = stats.invalid_hdf5_files == 0;
        let status = if count_ok && magic_ok { "OK" } else { "MISMATCH" };

        println!("{status} {}", expected.name);
        println!("  dir: {}", stats.directory.display());
        println!(
            "  samples: {} observed / {} expected",
            stats.hdf5_files, expected.expected_samples
        );
        println!("  bytes: {}", format_bytes(stats.total_bytes));
        println!(
            "  file size range: {} to {}",
            stats.min_file_bytes
                .map(format_bytes)
                .unwrap_or_else(|| "n/a".to_string()),
            stats.max_file_bytes
                .map(format_bytes)
                .unwrap_or_else(|| "n/a".to_string())
        );
        println!("  non-hdf5 files inside split: {}", stats.other_files);
        println!("  invalid hdf5 signatures: {}", stats.invalid_hdf5_files);
        println!();

        if !count_ok || !magic_ok {
            failed = true;
        }
        summaries.push(stats);
    }

    let total_samples: usize = summaries.iter().map(|stats| stats.hdf5_files).sum();
    let expected_total: usize = EXPECTED_SPLITS.iter().map(|split| split.expected_samples).sum();
    let total_bytes: u64 = summaries.iter().map(|stats| stats.total_bytes).sum();
    let total_ok = total_samples == expected_total;

    println!("Summary");
    println!(
        "  total samples: {} observed / {} expected",
        total_samples, expected_total
    );
    println!("  total bytes: {}", format_bytes(total_bytes));
    println!(
        "  approximate project-doc size target: {} GB",
        APPROX_TOTAL_SIZE_GB_DECIMAL
    );

    if !roughly_matches_project_size(total_bytes) {
        println!(
            "  size note: observed bytes are far from the project-doc scale, which may mean the dataset is incomplete or stored differently"
        );
    }

    println!();
    println!(
        "Note: this tool verifies split cardinality, on-disk bytes, and HDF5 file signatures. It does not inspect internal tensor shapes."
    );

    if failed || !total_ok {
        return Err("dataset audit failed".to_string());
    }

    println!("[done] dataset audit passed");
    Ok(())
}

fn parse_root_arg() -> Result<PathBuf, String> {
    let mut args = env::args_os();
    let program = args.next().unwrap_or_default();

    match (args.next(), args.next()) {
        (None, None) => Ok(PathBuf::from("data")),
        (Some(root), None) => Ok(PathBuf::from(root)),
        _ => Err(format!(
            "usage: {} [DATA_ROOT]",
            Path::new(&program)
                .file_name()
                .and_then(|value| value.to_str())
                .unwrap_or("dataset_audit")
        )),
    }
}

fn find_split_dir(root: &Path, expected: &ExpectedSplit) -> Option<PathBuf> {
    for alias in expected.aliases {
        let candidate = root.join(alias);
        if candidate.is_dir() {
            return Some(candidate);
        }
    }

    let nested_root = root.join("data");
    if nested_root.is_dir() {
        for alias in expected.aliases {
            let candidate = nested_root.join(alias);
            if candidate.is_dir() {
                return Some(candidate);
            }
        }
    }

    None
}

fn scan_split(directory: PathBuf) -> io::Result<SplitStats> {
    let mut stats = SplitStats {
        directory,
        ..SplitStats::default()
    };

    let mut stack = vec![stats.directory.clone()];
    while let Some(current_dir) = stack.pop() {
        for entry in fs::read_dir(&current_dir)? {
            let entry = entry?;
            let path = entry.path();
            let file_type = entry.file_type()?;

            if file_type.is_dir() {
                stack.push(path);
                continue;
            }

            if !file_type.is_file() {
                continue;
            }

            let metadata = entry.metadata()?;
            let file_size = metadata.len();

            if is_hdf5_path(&path) {
                stats.hdf5_files += 1;
                stats.total_bytes += file_size;
                stats.min_file_bytes = Some(
                    stats
                        .min_file_bytes
                        .map_or(file_size, |current| current.min(file_size)),
                );
                stats.max_file_bytes = Some(
                    stats
                        .max_file_bytes
                        .map_or(file_size, |current| current.max(file_size)),
                );

                if !has_hdf5_magic(&path)? {
                    stats.invalid_hdf5_files += 1;
                }
            } else {
                stats.other_files += 1;
            }
        }
    }

    Ok(stats)
}

fn is_hdf5_path(path: &Path) -> bool {
    matches!(
        path.extension().and_then(|value| value.to_str()),
        Some("hdf5") | Some("h5")
    )
}

fn has_hdf5_magic(path: &Path) -> io::Result<bool> {
    let mut file = File::open(path)?;
    let mut magic = [0_u8; 8];
    if file.read_exact(&mut magic).is_err() {
        return Ok(false);
    }
    Ok(magic == HDF5_MAGIC)
}

fn roughly_matches_project_size(total_bytes: u64) -> bool {
    let total_gb_decimal = total_bytes as f64 / 1_000_000_000_f64;
    total_gb_decimal >= 45.0 && total_gb_decimal <= 65.0
}

fn format_bytes(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KB", "MB", "GB", "TB"];

    let mut value = bytes as f64;
    let mut unit_index = 0_usize;
    while value >= 1000.0 && unit_index < UNITS.len() - 1 {
        value /= 1000.0;
        unit_index += 1;
    }

    if unit_index == 0 {
        format!("{bytes} {}", UNITS[unit_index])
    } else {
        format!("{value:.2} {}", UNITS[unit_index])
    }
}
