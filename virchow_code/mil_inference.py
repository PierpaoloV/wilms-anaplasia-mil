import os
import argparse
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from mil_modules import (
    MILSlideDataset,
    AttentionSingleBranch,
    generate_all_attention_reports,
    get_coords,
)
from mil_main import load_config


# ---------------------------------------------------------
# Utilities
# ---------------------------------------------------------
def get_labels_csv(cfg):
    return cfg.get("labels_csv") or cfg.get("labels_dir")


# ---------------------------------------------------------
# Single fold inference
# ---------------------------------------------------------
def run_fold(
    experiment_dir,
    fold,
    cfg,
    device="cuda",
    extract_region=False,
    combine_subplots=True,
    subplot_layout="horizontal",
):
    labels_csv = get_labels_csv(cfg)
    features_dir = f"{cfg['base_dir']}/features"
    coord_dir = f"{cfg['base_dir']}/coordinates"
    wsi_dir = cfg["wsi_dir"]

    df = pd.read_csv(labels_csv)
    fold_ids = df.loc[df["fold"] == fold, "slide_id"].astype(str).tolist()

    if len(fold_ids) == 0:
        print(f"⚠️ No slides for fold {fold}")
        return

    model_path = os.path.join(
        experiment_dir,
        "models",
        f"mil_best_fold{fold}.pt"
    )

    if not os.path.exists(model_path):
        print(f"⚠️ Model missing: {model_path}")
        return

    print(f"\n🚀 Run: {cfg['name']} | Fold {fold}")
    print(f"🧠 Model: {model_path}")

    # Output dir per run/fold
    fold_out = os.path.join(experiment_dir, "inference")
    os.makedirs(fold_out, exist_ok=True)

    # Dataset
    ds = MILSlideDataset(
        labels_csv=labels_csv,
        features_dir=features_dir,
        coord_dir=coord_dir,
        slide_ids=fold_ids,
    )

    loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False)

    # Model
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model = AttentionSingleBranch(
        size=tuple(cfg["size"]),
        use_dropout=cfg.get("use_dropout", False),
        n_classes=cfg.get("n_classes", 2),
    )

    state = torch.load(model_path, map_location=dev)
    model.load_state_dict(state)
    model.to(dev)
    model.eval()

    att_dir = os.path.join(fold_out, "attentions")
    emb_dir = os.path.join(fold_out, "embeddings")

    os.makedirs(att_dir, exist_ok=True)
    os.makedirs(emb_dir, exist_ok=True)

    with torch.no_grad():
        for feats, label, meta in tqdm(loader):
            feats = feats.to(dev)
            logits, out = model(feats)

            slide_id = meta["slide_id"][0]
            coords = get_coords(meta)

            att_raw = out["attention"]
            att_raw_vec = att_raw.squeeze(0).squeeze(-1).cpu().numpy()
            att_soft = F.softmax(att_raw, dim=1)
            att_soft_vec = att_soft.squeeze(0).squeeze(-1).cpu().numpy()

            np.savez(
                os.path.join(att_dir, f"{slide_id}_att_with_coords.npz"),
                attention=att_soft_vec,
                attention_raw=att_raw_vec,
                coords=coords,
            )

            if "slide_embedding" in out:
                emb = out["slide_embedding"].cpu().numpy()
                np.save(os.path.join(emb_dir, f"{slide_id}_embedding.npy"), emb)

    # Generate visual reports
    generate_all_attention_reports(
        base_exp_dir=fold_out,
        wsi_dir=wsi_dir,
        patch_size=cfg.get("patch_size", 224),
        patch_level=1,
        extract_region=extract_region,
        combine_subplots=combine_subplots,
        subplot_layout=subplot_layout,
        use_raw=True,
    )

    print(f"✅ Done fold {fold}")


# ---------------------------------------------------------
# Run full experiment
# ---------------------------------------------------------
def run_experiment(cfg, device, extract_region, combine_subplots, subplot_layout, rerun=False):
    exp_name = cfg.get("name", cfg.get("run_key", "unknown"))

    labels_csv = get_labels_csv(cfg)
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
        run_fold(
            experiment_dir,
            fold,
            cfg,
            device=device,
            extract_region=extract_region,
            combine_subplots=combine_subplots,
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
    parser.add_argument("--combine_subplots", action="store_true")
    parser.add_argument("--subplot_layout", default="horizontal")
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
            combine_subplots=args.combine_subplots,
            subplot_layout=args.subplot_layout,
            rerun=args.rerun,
        )

    print("\n🎉 All runs completed.")