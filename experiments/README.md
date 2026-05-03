# Experiment Index

This directory keeps the main approaches separated so the project history is easy to explain without changing the current CNext model code.

| Folder | Purpose | Status |
|---|---|---|
| `01_ejepa_vicreg_baseline` | ConvNeXt-style JEPA/VICReg baseline; wrapper now reruns at 96x96 via `baseline_jepa` | Historical 224x224 recorded, 96x96 rerun ready |
| `02_convnext_sigreg_failure` | JEPA + SIGReg attempt and collapse analysis | Negative result |
| `03_cnext_unet_forecasting_final` | Final 96x96 CNext-U-Net future-frame forecaster | Best result |

Each folder contains notes and a command wrapper. The wrappers use the shared modules in `active_matter_ssl/`; they do not duplicate model code.

Before running any wrapper, set:

```bash
export DATA_ROOT=/path/to/active_matter
```

Run wrappers with `bash`, for example:

```bash
bash experiments/03_cnext_unet_forecasting_final/run_full_pipeline.sh
```
