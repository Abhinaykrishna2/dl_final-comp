# Experiment Index

This directory keeps the main approaches separated so the project history is easy to explain without changing the current CNext model code.

| Folder | Purpose | Status |
|---|---|---|
| `01_ejepa_vicreg_baseline` | Original ConvNeXt-style JEPA/VICReg baseline | Recorded in `baseline.txt` |
| `02_convnext_sigreg_failure` | JEPA + SIGReg attempt and collapse analysis | Negative result |
| `03_vjepa_ema_masking` | V-JEPA-style masked prediction with EMA target | Implemented ablation |
| `04_cnext_unet_forecasting_final` | Final 96x96 CNext-U-Net future-frame forecaster | Best result |

Each folder contains notes and a command wrapper. The wrappers use the shared modules in `active_matter_ssl/`; they do not duplicate model code.

Before running any wrapper, set:

```bash
export DATA_ROOT=/path/to/active_matter
```

Run wrappers with `bash`, for example:

```bash
bash experiments/04_cnext_unet_forecasting_final/run_full_pipeline.sh
```
