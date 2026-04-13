"""External-validation inference for survival MIL.

Loads a pretrained survival checkpoint, runs forward pass on all slides
listed in the survival labels CSV, saves per-slide attentions/embeddings
and risk scores, computes C-index + time-dependent AUC, plots a
Kaplan-Meier curve stratified by median risk, and (optionally) renders
attention heatmap reports.

Usage:
    python pipeline/04_survival/survival_inference.py \
        --config configs/survival_runs.yaml --run cnio_external --device cuda
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

_THIS_DIR = Path(__file__).resolve().parent
_CLASSIFICATION_DIR = _THIS_DIR.parent / "03_classification"
for _p in (_THIS_DIR, _CLASSIFICATION_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from mil_main import load_config  # noqa: E402
from mil_modules import generate_experiment_reports, get_coords  # noqa: E402

from survival_dataset import SurvivalSlideDataset  # noqa: E402
from survival_metrics import (  # noqa: E402
    compute_cindex,
    compute_time_dependent_auc,
    plot_km_by_median_risk,
)
from survival_model import load_survival_checkpoint  # noqa: E402


def _features_dir(cfg) -> str:
    data_cfg = cfg["data"]
    if data_cfg.get("features_dir"):
        return data_cfg["features_dir"]
    return os.path.join(data_cfg["base_dir"], "features")


def _coord_dir(cfg) -> str:
    data_cfg = cfg["data"]
    if data_cfg.get("coord_dir"):
        return data_cfg["coord_dir"]
    return os.path.join(data_cfg["base_dir"], "coordinates")


def run_external_validation(cfg, device: str = "cuda", rerun: bool = False, generate_reports: bool = True, heatmap_only: bool = False):
    survival_cfg = cfg.get("survival", {})
    checkpoint_path = survival_cfg.get("checkpoint_path")
    if not checkpoint_path:
        raise ValueError("cfg['survival']['checkpoint_path'] is required.")

    output_dir = Path(cfg["output_dir"])
    inference_dir = output_dir / "inference"
    att_dir = inference_dir / "attentions"
    emb_dir = inference_dir / "embeddings"
    results_dir = output_dir / "results"
    for d in (inference_dir, att_dir, emb_dir, results_dir):
        d.mkdir(parents=True, exist_ok=True)

    predictions_csv = results_dir / "predictions.csv"
    if predictions_csv.exists() and not rerun:
        print(f"⏭️  Predictions already exist at {predictions_csv}. Use --rerun to overwrite.")
    else:
        dev = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"📦 Loading checkpoint: {checkpoint_path}")
        model = load_survival_checkpoint(checkpoint_path, device=dev)
        print(f"   inferred size = {model.size}")

        ds = SurvivalSlideDataset(
            labels_csv=cfg["data"]["labels_csv"],
            features_dir=_features_dir(cfg),
            coord_dir=_coord_dir(cfg),
            feature_suffix=str(survival_cfg.get("feature_suffix", "")),
            duration_unit=str(survival_cfg.get("duration_unit", "days")),
            clamp_years=survival_cfg.get("clamp_years"),
        )
        loader = DataLoader(ds, batch_size=1, shuffle=False)
        print(f"📊 {len(ds)} slides to score")

        rows = []
        with torch.no_grad():
            for feats, event, duration_years, meta in loader:
                feats = feats.to(dev)
                if feats.dim() == 3:
                    feats = feats.squeeze(0)
                risk, out = model(feats)
                slide_id = meta["slide_id"][0]
                coords = get_coords(meta)

                att_raw = out.get("attention")
                if att_raw is not None:
                    att_raw_vec = att_raw.squeeze(0).squeeze(-1).cpu().numpy()
                    att_soft_vec = F.softmax(att_raw, dim=1).squeeze(0).squeeze(-1).cpu().numpy()
                    np.savez(
                        att_dir / f"{slide_id}_att_with_coords.npz",
                        attention=att_soft_vec,
                        attention_raw=att_raw_vec,
                        coords=coords,
                    )

                if "slide_embedding" in out:
                    np.save(emb_dir / f"{slide_id}_embedding.npy", out["slide_embedding"].cpu().numpy())

                rows.append({
                    "slide_id": slide_id,
                    "risk_score": float(risk.item()),
                    "event": int(event.item()),
                    "duration_years": float(duration_years.item()),
                    "duration_raw": float(meta["duration_raw"][0].item() if torch.is_tensor(meta["duration_raw"]) else meta["duration_raw"][0]),
                    "duration_unit": meta["duration_unit"][0],
                })

        pd.DataFrame(rows).to_csv(predictions_csv, index=False)
        print(f"💾 Wrote {predictions_csv}")

    # ---- metrics ---------------------------------------------------------
    df = pd.read_csv(predictions_csv)
    risk = df["risk_score"].to_numpy()
    event = df["event"].to_numpy()
    duration_unit = df["duration_unit"].iloc[0]
    duration_for_metrics = df["duration_raw"].to_numpy()  # native unit (days/years)

    cindex = compute_cindex(risk, event, duration_for_metrics)

    eval_times = survival_cfg.get("eval_time_points") or [180, 365, 730]
    time_auc = compute_time_dependent_auc(risk, event, duration_for_metrics, eval_times)

    metrics = {
        "n_slides": int(len(df)),
        "n_events": int(event.sum()),
        "duration_unit": duration_unit,
        "cindex": cindex,
        "time_dependent_auc": time_auc,
    }
    metrics_path = results_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"📈 C-index = {cindex:.3f}    AUC = {time_auc}")
    print(f"💾 Wrote {metrics_path}")

    # ---- Kaplan-Meier ----------------------------------------------------
    km_path = results_dir / "kaplan_meier_median_split.png"
    p_value = plot_km_by_median_risk(risk, event, duration_for_metrics, km_path, time_unit_label=duration_unit)
    print(f"📉 KM plot saved to {km_path} (log-rank p = {p_value:.3g})")

    # ---- attention heatmap reports --------------------------------------
    if generate_reports:
        print("🎨 Rendering attention heatmaps…")
        generate_experiment_reports(str(output_dir), cfg, heatmap_only=heatmap_only)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run", default="all", help="Run name or 'all'")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rerun", action="store_true",
                        help="Recompute predictions even if results/predictions.csv exists")
    parser.add_argument("--no_reports", action="store_true",
                        help="Skip attention heatmap rendering (faster smoke tests)")
    parser.add_argument("--heatmap_only", action="store_true",
                        help="Save only the WSI heatmap ({slide_id}_heatmap.jpg) without the top-K patch grid")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    run_keys = list(config.get("runs", {}).keys()) if args.run == "all" else [args.run]
    for run_key in run_keys:
        cfg = load_config(args.config, run_key)
        print(f"\n==============================")
        print(f"🚀 RUN: {cfg.get('name', run_key)}")
        print(f"📁 Dir: {cfg['output_dir']}")
        run_external_validation(
            cfg,
            device=args.device,
            rerun=args.rerun,
            generate_reports=not args.no_reports,
            heatmap_only=args.heatmap_only,
        )

    print("\n🎉 All runs completed.")
