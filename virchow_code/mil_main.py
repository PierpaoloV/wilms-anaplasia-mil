#!/usr/bin/env python3
import argparse
import yaml
import os
import gc
import torch
import pandas as pd
from pathlib import Path
from mil_modules import cross_validate_mil, generate_all_attention_reports_flat,generate_all_attention_reports

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
            # Try float first (handles scientific notation like 1e-4)
            if any(c in val for c in [".", "e"]):
                return float(val)
            # Try int
            return int(val)
        except ValueError:
            return value  # return original string if not numeric
    elif isinstance(value, list):
        # Recursively cast elements in lists
        return [_safe_cast(v) for v in value]
    return value


# def load_config(config_path: str, run_name: str):
#     """Load experiment config with safe casting and merged defaults."""
#     with open(config_path, "r") as f:
#         config_all = yaml.safe_load(f)

#     defaults = config_all.get("defaults", {})
#     runs = config_all.get("runs", {})

#     if run_name not in runs:
#         raise ValueError(f"Run '{run_name}' not found in config file.")

#     cfg = {**defaults, **runs[run_name]}
#     cfg = {k: _safe_cast(v) for k, v in cfg.items()}  # ✅ auto-cast everything

#     # Build output dir
#     base_exp_dir = Path("/opt/app/user/postdoc/Anaplasia_Classification/virchow/experiments_complete_debug")
#     cfg["output_dir"] = str(base_exp_dir / cfg["name"])

#     return cfg
def load_config(config_path: str, run_name: str):
    """Load experiment config with safe casting and merged defaults."""
    with open(config_path, "r") as f:
        config_all = yaml.safe_load(f)

    defaults = config_all.get("defaults", {})
    runs = config_all.get("runs", {})

    if run_name not in runs:
        raise ValueError(f"Run '{run_name}' not found in config file.")

    # merge defaults + run-specific
    cfg = {**defaults, **runs[run_name]}
    cfg = {k: _safe_cast(v) for k, v in cfg.items()}  # auto-cast

    # -----------------------------
    # OUTPUT DIR LOGIC (NEW)
    # -----------------------------
    # Priority:
    # 1) run-level output_dir (absolute path)
    # 2) output_base_dir + cfg["name"]
    # 3) fallback to old hard-coded base dir + cfg["name"]

    if "output_dir" in cfg and cfg["output_dir"]:
        # output_dir explicitly set in YAML (can be absolute or relative)
        cfg["output_dir"] = str(Path(cfg["output_dir"]))
    else:
        output_base_dir = cfg.get(
            "output_base_dir",
            "/opt/app/user/postdoc/Anaplasia_Classification/virchow/experiments_complete"
        )
        cfg["output_dir"] = str(Path(output_base_dir) / cfg["name"])

    return cfg

def run_experiment(cfg):
    """Run a single MIL experiment (cross-validation + visualization)."""
    print(f"\n🚀 Starting experiment: {cfg['name']}")
    print("=" * 60)
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    print(f"🧠 Device: {device}\n")
    output_dir = cfg["output_dir"]
    # output_dir = f"/opt/app/user/postdoc/Anaplasia_Classification/virchow/experiments_complete/{cfg['name']}"
    os.makedirs(output_dir, exist_ok=True)
    if cfg.get("only_reports", False):
        print("🖼️ only_reports=True → skipping training, generating attention reports only")
        generate_all_attention_reports(
            base_exp_dir=output_dir,
            wsi_dir=cfg["wsi_dir"],
            patch_size=int(cfg.get("patch_size", 224)),
            patch_level=1,
            vis_level=int(cfg.get("vis_level", -1)),
            alpha=float(cfg.get("alpha", 0.45)),
            convert_to_percentiles=bool(cfg.get("convert_to_percentiles", True)),
            max_size=int(cfg.get("max_size", 4096)),
            use_raw=bool(cfg.get("use_raw", True)),
            draw_topk=20,
            combine_subplots=True,
            subplot_layout='horizontal'
        )
        return

    try:
        # --- Run cross-validation ---
        results_df, metrics_df, summary_df = cross_validate_mil(
            splits_csv=cfg["labels_csv"],
            features_dir=os.path.join(cfg["base_dir"], "features/"),
            coord_dir=os.path.join(cfg["base_dir"], "coordinates/"),
            output_dir=output_dir,
            n_classes=cfg["n_classes"],
            epochs=int(cfg["epochs"]),
            lr=float(cfg["lr"]),
            batch_size=int(cfg["batch_size"]),
            penalty_factor=float(cfg["penalty_factor"]),
            size=tuple(cfg["size"]),
            device=device,
            weighted=cfg.get("weighted", False),
        )

        # --- Save t-SNE or attention maps (optional) ---
        if cfg.get("mode", "full") == "full":
            generate_all_attention_reports(
                base_exp_dir=output_dir,
                wsi_dir=cfg["wsi_dir"],
                patch_size=int(cfg.get("patch_size", 224)),
                patch_level=1,
                vis_level=int(cfg.get("vis_level", -1)),
                alpha=float(cfg.get("alpha", 0.45)),
                convert_to_percentiles=bool(cfg.get("convert_to_percentiles", True)),
                max_size=int(cfg.get("max_size", 4096)),
                use_raw=bool(cfg.get("use_raw", True)),
                draw_topk=20,
                combine_subplots=True,
                subplot_layout='horizontal'
            )

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

        for run_name in all_runs:
            cfg = load_config(args.config, run_name)
            summary_path = Path(cfg["output_dir"]) / "results" / "summary.csv"
        
            if summary_path.exists() and not args.rerun:
                print(f"⏭️ Skipping '{cfg['name']}' — summary already exists ({summary_path})")
                continue
        
            try:
                print(f"\n🚀 Starting experiment: {cfg['name']}")
                print(f"{'='*60}\n🧠 Device: {cfg.get('device', 'cpu')}")
                print(f"\n🔍 CONFIGURATION FOR {cfg['name']}:")
                for k, v in cfg.items():
                    print(f"  {k}: {v} ({type(v).__name__})")
                print("=" * 60)
        
                run_experiment(cfg)

                if summary_path.exists():
                    summary_df = pd.read_csv(summary_path, index_col=0).transpose()
                    summary_df["run_name"] = cfg["name"]
                    for k, v in cfg.items():
                        if isinstance(v, (int, float, str, bool, list)):
                            summary_df[k] = str(v)
                    all_summaries.append(summary_df)

            except Exception as e:
                print(f"⚠️ Run {cfg['name']} failed: {e}")
            finally:
                torch.cuda.empty_cache()
                gc.collect()

        # Combine summaries at the end
        if all_summaries:
            combined_df = pd.concat(all_summaries, ignore_index=True)
            combined_path = Path(f"{cfg['output_dir']}/all_experiments_summary.csv")
            combined_df.to_csv(combined_path, index=False)
            print(f"\n📊 Combined summary saved to: {combined_path}")

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