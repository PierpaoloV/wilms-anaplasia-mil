# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

All code runs inside the **pathology-pipeline Docker container**. The container paths used throughout the codebase:
- Features/coordinates: `/opt/app/user/postdoc/Anaplasia_Classification/virchow/virchow_features/`
- Labels CSV: `/opt/app/user/postdoc/Anaplasia_Classification/csvs/splits_updated.csv`
- WSI images: `/data/pa_cpgarchive/archives/kidney/Wilms_Tumor/Images/`
- Experiment outputs: `/opt/app/user/postdoc/Anaplasia_Classification/virchow/experiments_complete/`

The project is incomplete — additional files/folders may need to be copied from a Samba share via `scp`.

## Running Experiments

**Train (cross-validation) for a specific run:**
```bash
python virchow_code/mil_main.py --config virchow_code/runs.yaml --run <run_name>
```

**Train all runs defined in the config:**
```bash
python virchow_code/mil_main.py --config virchow_code/runs.yaml --run all
```

**Force rerun even if results exist:**
```bash
python virchow_code/mil_main.py --config virchow_code/runs.yaml --run <run_name> --rerun
```

**Generate inference/attention reports from a trained model:**
```bash
python virchow_code/mil_inference.py --config virchow_code/runs.yaml --device cuda
```

Available run names (defined in `runs.yaml`): `baseline`, `baseline_weighted`, `deep_attention`, `only_yes`, `deep_yes`, `yes_penalty`.

## Architecture

The pipeline has three stages:

1. **Preprocessing** (not yet in repo): WSI → tissue/background segmentation → patch extraction → CSV generation with slide metadata and fold assignments.

2. **Feature extraction** (not yet in repo): Patches → Virchow/Virchow2 or PRISM foundation model → `.pt` feature tensors + `.npy` coordinate files per slide.

3. **MIL classification** (`virchow_code/`): Pre-extracted features → AMIL model → binary prediction (Anaplasia vs. Non-Anaplasia).

### Key files in `virchow_code/`

| File | Role |
|------|------|
| `mil_modules.py` | Core: `MILSlideDataset`, `AttentionSingleBranch` (AMIL model), `cross_validate_mil`, attention report generation |
| `mil_main.py` | Training entry point — loads `runs.yaml`, calls `cross_validate_mil`, saves models/results/attention maps |
| `mil_inference.py` | Inference entry point — loads saved fold models, runs prediction, saves attention `.npz` + embeddings, generates visual reports |
| `runs.yaml` | Experiment registry — `defaults` block + per-run overrides merged at runtime |

### Data flow

- **Labels CSV** (`splits_updated.csv`): columns `slide_id`, `Diagnose` (`Not Anaplasia` / `Focal` / `Diffuse`), `fold`. Labels are binarized: `Not Anaplasia=0`, `Focal/Diffuse=1`.
- **Features**: `<features_dir>/features/<slide_id>.pt` — tensor `[N_patches, feat_dim]`
- **Coordinates**: `<features_dir>/coordinates/<slide_id>.npy` — structured array with `x`, `y` fields
- **Model outputs**: per-fold best models saved as `mil_best_fold{N}.pt`; attention maps as `{slide_id}_att_with_coords.npz`

### Config system

`runs.yaml` has a `defaults` block merged with each run's overrides at load time. Key parameters: `size` (MLP layer dimensions, e.g. `[1280, 512, 256]`), `weighted` (class-weighted sampling), `penalty_factor`, `epochs`, `lr`. Output directory is resolved as `output_base_dir/<run_name>`.
