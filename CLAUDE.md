# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

All code runs inside the **pathology-pipeline Docker container** (`pierpaolov93/pathology-pipeline:cellvit`). Current container paths:
- Virchow2 features: `/data/pathology/projects/pierpaolo/postdoc/Anaplasia_Classification/virchow2/2026-03-17_18_04/`
- Virchow1 features: `/data/pathology/projects/pierpaolo/postdoc/Anaplasia_Classification/virchow/virchow_features/2025-10-08_14_17/`
- Labels CSV: `/data/pathology/projects/pierpaolo/postdoc/Anaplasia_Classification/csvs/splits_updated.csv`
- WSI images: `/data/pa_cpgarchive/archives/kidney/Wilms_Tumor/Images/`
- Experiment outputs: `/home/user/experiments_complete/`

## Repository Structure

The `unified` branch was merged into `main` (March 2026). All code now lives under `pipeline/` and `configs/`:

```
pipeline/
  01_segmentation/       — TB segmentation scripts
  02_feature_extraction/ — prepare_csv.py for slide2vec
  03_classification/     — AMIL training, inference, preprocessing
    mil_modules.py       — Core model, training loop, visualization
    mil_main.py          — Training CLI
    mil_inference.py     — Inference + heatmap CLI
    preprocessing.py     — Patient-level fold generation
    linear_probing.py    — Linear/MLP baselines
configs/
  runs.yaml              — AMIL experiment registry
  linear_runs.yaml       — Linear/MLP baseline registry
  slide2vec.yaml         — Feature extraction config
scripts/
  run_pipeline.slurm     — End-to-end SLURM job
```

## Running Experiments

**Train a single run:**
```bash
python pipeline/03_classification/mil_main.py --config configs/runs.yaml --run <run_name>
```

**Train all runs (skip completed):**
```bash
python pipeline/03_classification/mil_main.py --config configs/runs.yaml --run all
```

**Force rerun:**
```bash
python pipeline/03_classification/mil_main.py --config configs/runs.yaml --run <run_name> --rerun
```

**Inference + heatmaps only:**
```bash
python pipeline/03_classification/mil_inference.py --config configs/runs.yaml --run all --device cuda
```

Available runs (defined in `configs/runs.yaml`): `baseline`, `baseline_weighted`, `deep_attention`, `only_yes`, `deep_yes`, `yes_penalty`.

Each completed run produces:
- `output_dir/training.log` — timestamped epoch-by-epoch log
- `output_dir/results/summary.csv` — mean/std/median per metric across folds
- `output_dir/results/per_fold_metrics.csv` — one row per fold
- `output_dir/results/per_slide_predictions.csv`
- `output_dir/models/mil_best_auc_fold{N}.pt` — best val AUC checkpoint per fold
- `output_dir/models/mil_best_loss_fold{N}.pt` — best val loss checkpoint per fold
- `output_dir/models/mil_best_gmean_fold{N}.pt` — best Gmean checkpoint per fold
- `combined_results.csv` in `output_base_dir` — cross-run summary

## Architecture

**Pipeline stages:**
1. Tissue/background segmentation → binary mask per WSI
2. Feature extraction (slide2vec + Virchow/Virchow2) → `.pt` features + `.npy` coordinates per slide
3. AMIL classification → binary prediction (Anaplasia vs Not Anaplasia)

**Model: `AttentionSingleBranch`**
- `size` parameter defines MLP layers: e.g. `[1280, 512, 128]` means `1280→512` MLP then gated attention with 128-dim bottleneck
- Minimum 2 elements in `size` (input + bottleneck, no hidden layer)
- Gated attention: `A = softmax(Wa·tanh(Vx) ⊙ sigmoid(Ux))`
- Output: slide embedding → linear classifier → logits

**Data:**
- `splits_updated.csv`: 188 slides, 24 patients, columns `Patient_id`, `slide_id`, `Diagnose` (`Not Anaplasia`/`Focal`/`Diffuse`), `fold` (1–5)
- Labels binarized: `Not Anaplasia=0`, `Focal/Diffuse=1`
- Patient-level splits (no leakage) — but unequal fold sizes due to 3 heavy Diffuse patients (19, 15, 14 slides)

## Current Default Hyperparameters (`configs/runs.yaml`)

```yaml
lr: 1e-5             # reduced from 1e-4 to slow attention learning
size: [1280, 512, 128]
weight_decay: 1e-4   # AdamW
label_smoothing: 0.1 # prevents overconfidence, improves calibration
gmean_threshold: 0.55 # gate: save only when sqrt(sens*spec) >= threshold
epochs: 20
weighted: false       # weighted sampling hurts on this ~50/50 dataset
```

**Checkpoint strategy:** all epochs are saved during training; after the fold completes, three named checkpoints are selected retrospectively and all per-epoch files are deleted. The first 3 epochs (warmup) are excluded from selection.

| Checkpoint | Criterion |
|---|---|
| `mil_best_auc_fold{N}.pt` | highest val AUC |
| `mil_best_loss_fold{N}.pt` | lowest val loss |
| `mil_best_gmean_fold{N}.pt` | highest Gmean = √(sens × spec) |

If two or more criteria point to the same epoch, the file is copied under each name and a log message notes the overlap. The canonical checkpoint used by the inference code is `mil_best_auc_fold{N}.pt`.

The `gmean_threshold` parameter is no longer used for checkpoint gating (it remains in `runs.yaml` for reference but has no effect on saving).

**Epoch progress display:** `Gmean | Sens | AUC` — F1 and ACC removed.

**Metrics reported:** `precision`, `sensitivity`, `f1`, `auc` — with mean ± std and median in `summary.csv` and `combined_results.csv`.

## Dataset Analysis

- **24 patients**, 188 slides total
- **13 positive patients** (100 slides): 11 Diffuse, 1 Focal (2 slides), 1 Focal (1 slide)
- **11 negative patients** (88 slides): all exactly 8 slides each
- **Fold imbalance** is unavoidable: 3 Diffuse patients with 19/15/14 slides dominate whichever fold they land in

**Fold composition (splits_updated.csv):**
| Fold | Slides | Pos | % Pos | Val patients |
|------|--------|-----|-------|--------------|
| 1 | 40 | 16 | 40% | 5 patients |
| 2 | 47 | 31 | 66% | heavily positive — hard fold |
| 3 | 38 | 22 | 58% | best performing fold |
| 4 | 34 | 18 | 53% | most balanced |
| 5 | 29 | 13 | 45% | smallest — noisy estimates |

## Experimental Results (Virchow2 features, `anaplasia_mil_baseline`)

Per-fold AUC with current config (`lr=1e-5`, `size=[1280,512,128]`, `label_smoothing=0.1`):

| Fold | AUC | Best epoch | Notes |
|------|-----|------------|-------|
| 1 | 0.932 | 3 | Solid |
| 2 | 0.744 | 1 | Hard fold (66% pos val) |
| 3 | 0.952 | 18 | Best — model still improving at epoch 20 |
| 4 | 0.726 | 18 | Improved vs old code (was 0.552 with Virchow1) |
| 5 | 0.736 | 11 | Small fold, noisy |

**Approximate summary:** mean AUC ~0.818, median ~0.736, std ~0.09.

**Key finding:** weighted sampling (`baseline_weighted`) is consistently worse than unweighted across 4/5 folds. Dataset is ~50/50 overall so weighting over-corrects.

**Virchow1 vs Virchow2:** Virchow2 features are substantially better, especially on fold 4 (AUC 0.726 vs 0.552). Fold 4 patients (000032, 000077, 000087, 000107, 000069) are the discriminating test case between the two models.

## Known Issues / TODO

- Fold 3 still improving at epoch 20 — consider increasing to 30–40 epochs
- Fold 5 (29 slides) is too small for reliable AUC estimates
- Only 2 Focal anaplasia patients in the entire dataset — model is effectively learning Diffuse vs Not Anaplasia
- External validation cohort inference available via `external_inference.ipynb`
- `runs.yaml` still has some old run configs (`deep_attention`, `only_yes`, etc.) that use outdated `labels_dir` paths — verify before running
