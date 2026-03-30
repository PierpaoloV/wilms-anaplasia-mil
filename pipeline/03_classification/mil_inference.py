import os
import argparse
import yaml
import pandas as pd

from mil_modules import run_inference_fold, generate_experiment_reports
from mil_main import load_config


# ---------------------------------------------------------
# Run full experiment
# ---------------------------------------------------------
def run_experiment(cfg, device, extract_region, subplot_layout, checkpoint="auc", rerun=False):
    exp_name = cfg.get("name", cfg.get("run_key", "unknown"))

    labels_csv = cfg.get("labels_csv") or cfg.get("labels_dir")
    df = pd.read_csv(labels_csv)
    folds = sorted(df["fold"].unique())

    experiment_dir = os.path.join(cfg["output_base_dir"], exp_name)
    vis_dir = os.path.join(experiment_dir, "inference", "visual_reports")
    if not rerun and os.path.exists(vis_dir):
        n_files = len([f for f in os.listdir(vis_dir) if f.endswith(".png") or f.endswith(".jpg")])
        if n_files > 100:
            print(f"\n⏭️ Skipping run '{exp_name}' — inference already exists ({vis_dir})")
            return

    print(f"\n==============================")
    print(f"🚀 RUN: {exp_name}")
    print(f"📁 Dir: {experiment_dir}")
    print(f"Folds: {folds}")

    for fold in folds:
        run_inference_fold(
            experiment_dir,
            fold,
            cfg,
            device=device,
            checkpoint=checkpoint,
            generate_reports=False,
        )

    generate_experiment_reports(
        experiment_dir,
        cfg,
        extract_region=extract_region,
        subplot_layout=subplot_layout,
    )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run", default="all", help="Run name or 'all'")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--extract_region", action="store_true")
    parser.add_argument("--subplot_layout", default="horizontal")
    parser.add_argument("--checkpoint", default="auc", choices=["auc", "loss", "gmean"],
                        help="Which saved checkpoint to use for inference (default: auc)")
    parser.add_argument("--rerun", action="store_true", help="Regenerate reports even if they already exist")

    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    runs = config.get("runs", {})
    run_keys = runs.keys() if args.run == "all" else [args.run]

    for run_key in run_keys:
        cfg = load_config(args.config, run_key)
        run_experiment(
            cfg,
            device=args.device,
            extract_region=args.extract_region,
            subplot_layout=args.subplot_layout,
            checkpoint=args.checkpoint,
            rerun=args.rerun,
        )

    print("\n🎉 All runs completed.")
