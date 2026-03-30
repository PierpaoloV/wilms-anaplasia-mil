#!/usr/bin/env python3
import argparse
import yaml
import os
import gc
import logging
import torch
import pandas as pd
from pathlib import Path
from mil_modules import cross_validate_mil, run_inference_fold, generate_experiment_reports

# --------------------------------------------------------------------
# HELPER
# --------------------------------------------------------------------
def fix_seed(seed=42):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False




def _safe_cast(value):
    """Safely cast YAML string values to int/float/bool if applicable."""
    if isinstance(value, str):
        val = value.strip().lower()
        if val in ("true", "false"):
            return val == "true"
        try:
            if any(c in val for c in [".", "e"]):
                return float(val)
            return int(val)
        except ValueError:
            return value
    elif isinstance(value, list):
        return [_safe_cast(v) for v in value]
    return value


def _deep_cast(obj):
    """Recursively apply _safe_cast to all leaf values in a nested dict."""
    if isinstance(obj, dict):
        return {k: _deep_cast(v) for k, v in obj.items()}
    return _safe_cast(obj)


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base, returning a new dict."""
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def load_config(config_path: str, run_name: str) -> dict:
    """Load experiment config, deep-merging defaults and run-specific overrides."""
    with open(config_path, "r") as f:
        config_all = yaml.safe_load(f)

    defaults = config_all.get("defaults", {})
    runs = config_all.get("runs", {})

    if run_name not in runs:
        raise ValueError(f"Run '{run_name}' not found in config file.")

    cfg = _deep_merge(defaults, runs[run_name])
    cfg = _deep_cast(cfg)

    # Build output_dir: explicit run-level override > experiment.output_base_dir + name
    if cfg.get("output_dir"):
        cfg["output_dir"] = str(Path(cfg["output_dir"]))
    else:
        cfg["output_dir"] = str(Path(cfg["experiment"]["output_base_dir"]) / cfg["name"])

    return cfg

def run_experiment(cfg):
    """Run a single MIL experiment (cross-validation + visualization)."""
    print(f"\n🚀 Starting experiment: {cfg['name']}")
    print("=" * 60)
    device = torch.device(cfg["experiment"].get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    print(f"🧠 Device: {device}\n")
    output_dir = cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    if cfg["experiment"].get("mode") == "reports_only":
        print("🖼️ mode=reports_only → skipping training, regenerating inference reports")
        df    = pd.read_csv(cfg["data"]["labels_csv"])
        folds = sorted(df["fold"].unique())
        for fold in folds:
            print(f"\n--- Fold {fold} ---")
            run_inference_fold(output_dir, fold, cfg, device=str(device), generate_reports=False)
        generate_experiment_reports(output_dir, cfg)
        return

    # --- Set up training log ---
    log_path = Path(output_dir) / "training.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(cfg["name"])
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(logging.Formatter("%(asctime)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(fh)
    logger.info(f"Experiment: {cfg['name']}")
    logger.info(f"Config: { {k: v for k, v in cfg.items() if k not in ('output_dir',)} }")

    try:
        # --- Run cross-validation ---
        results_df, metrics_df, summary_df = cross_validate_mil(
            splits_csv=cfg["data"]["labels_csv"],
            features_dir=os.path.join(cfg["data"]["base_dir"], "features/"),
            coord_dir=os.path.join(cfg["data"]["base_dir"], "coordinates/"),
            output_dir=output_dir,
            n_classes=cfg["model"]["n_classes"],
            epochs=int(cfg["training"]["epochs"]),
            lr=float(cfg["training"]["lr"]),
            batch_size=int(cfg["training"]["batch_size"]),
            penalty_factor=float(cfg["training"]["penalty_factor"]),
            size=tuple(cfg["model"]["size"]),
            device=device,
            weighted=cfg["training"].get("weighted", False),
            weight_decay=float(cfg["training"]["weight_decay"]),
            gmean_threshold=float(cfg["training"]["gmean_threshold"]),
            label_smoothing=float(cfg["training"]["label_smoothing"]),
            logger=logger,
        )

        # --- Generate attention heatmaps via the shared inference pipeline ---
        if cfg["experiment"].get("mode", "full") == "full":
            df    = pd.read_csv(cfg["data"]["labels_csv"])
            folds = sorted(df["fold"].unique())
            for fold in folds:
                print(f"\n--- Fold {fold} inference ---")
                run_inference_fold(output_dir, fold, cfg, device=str(device), generate_reports=False)
            generate_experiment_reports(output_dir, cfg)

        print(f"✅ Finished {cfg['name']}")
        torch.cuda.empty_cache()
        gc.collect()

    except Exception as e:
        print(f"⚠️ Run {cfg['name']} failed: {e}")
        torch.cuda.empty_cache()
        gc.collect()


# --------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Run MIL experiment(s)")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config file")
    parser.add_argument("--run", type=str, required=True, help="Run name or 'all' to execute every experiment")
    parser.add_argument("--rerun", action="store_true", help="Force rerun even if results already exist")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config_all = yaml.safe_load(f)
    all_runs = config_all.get("runs", {})

    all_summaries = []

    if args.run.lower() == "all":
        print(f"🧪 Running all {len(all_runs)} experiments defined in {args.config}\n")

        output_base_dir = None

        for run_name in all_runs:
            cfg = load_config(args.config, run_name)
            output_base_dir = cfg["experiment"]["output_base_dir"]
            summary_path = Path(cfg["output_dir"]) / "results" / "summary.csv"

            if summary_path.exists() and not args.rerun:
                print(f"⏭️ Skipping '{cfg['name']}' — summary already exists ({summary_path})")
            else:
                try:
                    print(f"\n🚀 Starting experiment: {cfg['name']}")
                    print(f"{'='*60}\n🧠 Device: {cfg['experiment'].get('device', 'cpu')}")
                    print(f"\n🔍 CONFIGURATION FOR {cfg['name']}:")
                    for k, v in cfg.items():
                        print(f"  {k}: {v} ({type(v).__name__})")
                    print("=" * 60)

                    run_experiment(cfg)

                except Exception as e:
                    print(f"⚠️ Run {cfg['name']} failed: {e}")
                finally:
                    torch.cuda.empty_cache()
                    gc.collect()

            # Collect metrics regardless of whether we just ran or skipped
            if summary_path.exists():
                df_s = pd.read_csv(summary_path, index_col=0)
                labels_path = cfg["data"]["labels_csv"]
                size = cfg["model"]["size"]
                row = {
                    "run_name":       cfg["name"],
                    "dataset":        "yes_only" if "selected_yes" in str(labels_path) else "all",
                    "weighted":       cfg["training"]["weighted"],
                    "penalty_factor": cfg["training"]["penalty_factor"],
                    "architecture":   "→".join(str(s) for s in size),
                    "f1":           f"{round(df_s.loc['mean','f1'],4)} ± {round(df_s.loc['std','f1'],4)}",
                    "f1_median":    round(df_s.loc['median','f1'], 4),
                    "auc":          f"{round(df_s.loc['mean','auc'],4)} ± {round(df_s.loc['std','auc'],4)}",
                    "auc_median":   round(df_s.loc['median','auc'], 4),
                    "precision":      f"{round(df_s.loc['mean','precision'],4)} ± {round(df_s.loc['std','precision'],4)}",
                    "prec_median":    round(df_s.loc['median','precision'], 4),
                    "sensitivity":    f"{round(df_s.loc['mean','sensitivity'],4)} ± {round(df_s.loc['std','sensitivity'],4)}",
                    "sens_median":    round(df_s.loc['median','sensitivity'], 4),
                }
                all_summaries.append(row)

        # Save combined results to output_base_dir
        if all_summaries and output_base_dir:
            combined_df = pd.DataFrame(all_summaries)
            combined_path = Path(output_base_dir) / "combined_results.csv"
            combined_df.to_csv(combined_path, index=False)
            print(f"\n📊 Combined results saved to: {combined_path}")
            print(combined_df.to_string(index=False))

        print("\n✅ All experiments completed successfully.")

    else:
        cfg = load_config(args.config, args.run)
        summary_path = Path(cfg["output_dir"]) / "results" / "summary.csv"
        if summary_path.exists() and not args.rerun:
            print(f"⏭️ Skipping '{cfg['name']}' — summary already exists ({summary_path})")
            return
        run_experiment(cfg)

if __name__ == "__main__":
    main()