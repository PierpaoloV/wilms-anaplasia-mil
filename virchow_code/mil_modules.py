import os, math, time, random, json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

import gc

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
)

from tqdm import tqdm
from collections import Counter
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.patches as patches
import openslide
import cv2
import seaborn as sns
from scipy.stats import rankdata  
from PIL import Image, ImageDraw, ImageFont


class MILSlideDataset(Dataset):
    """
    Expects:
      - features_dir/<slide_id>.pt          -> Tensor [N_patches, feat_dim]
      - coord_dirs/<slide_id>_coords.npy    -> structured array with fields 'x' and 'y'
      - labels CSV with columns: slide_id, Diagnose
        Diagnose ∈ {"Not Anaplasia", "Focal", "Diffuse"}
    """
    def __init__(self, labels_csv, features_dir, coord_dir, slide_ids=None):
        self.df = pd.read_csv(labels_csv)
        if slide_ids is not None:
            self.df = self.df[self.df["slide_id"].isin(slide_ids)].reset_index(drop=True)
        self.features_dir = Path(features_dir)
        self.coord_dir = Path(coord_dir)

    def __len__(self):
        return len(self.df)

    def _load_features(self, slide_id):
        path = self.features_dir / f"{slide_id}.pt"
        feats = torch.load(path)  # expected tensor [N, D]
        if feats.dtype != torch.float32:
            feats = feats.float()
        return feats

    def _load_coords(self, slide_id):
        npy_path = self.coord_dir / f"{slide_id}.npy"
        if not npy_path.exists():
            print('Coordinates do not exists?')
            return None
        arr = np.load(npy_path)
        if "x" in arr.dtype.names and "y" in arr.dtype.names:
            return np.column_stack((arr["x"], arr["y"])).astype(np.float32)
        return None

    def _encode_label(self, diagnosis: str, binary: bool = True) -> int:
        """
        Converts Diagnose string to binary or multiclass label.
        Expected values in CSV: 'Not Anaplasia', 'Focal', 'Diffuse'.
        """
        label_map = {"Not Anaplasia": 0, "Focal": 1, "Diffuse": 2}
        binary_map = {"Not Anaplasia": 0, "Focal": 1, "Diffuse": 1}
    
        diag = diagnosis.strip()
    
        if binary:
            if diag not in binary_map:
                raise ValueError(f"Unknown label '{diagnosis}' in CSV (expected 'Not Anaplasia', 'Focal', or 'Diffuse')")
            return binary_map[diag]
        else:
            if diag not in label_map:
                raise ValueError(f"Unknown label '{diagnosis}' in CSV (expected 'Not Anaplasia', 'Focal', or 'Diffuse')")
            return label_map[diag]

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        slide_id = row["slide_id"]
        label = self._encode_label(row["Diagnose"], binary=True)
        # label = self._encode_label(row["Diagnose"])

        feats = self._load_features(slide_id)
        coords = self._load_coords(slide_id)

        meta = {"slide_id": slide_id, "coords": coords}
        return feats, torch.tensor(label, dtype=torch.long), meta


def initialize_weights(m: torch.nn.Module):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class GatedAttention(nn.Module):
    """Gated attention block used for MIL aggregation."""
    def __init__(self, input_dim, bottleneck_dim, dropout=False, n_branches=1):
        super().__init__()
        self.attention_a = nn.Sequential(
            nn.Linear(input_dim, bottleneck_dim),
            nn.Tanh(),
            nn.Dropout(0.25) if dropout else nn.Identity()
        )
        self.attention_b = nn.Sequential(
            nn.Linear(input_dim, bottleneck_dim),
            nn.Sigmoid(),
            nn.Dropout(0.25) if dropout else nn.Identity()
        )
        self.attention_c = nn.Linear(bottleneck_dim, n_branches)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        att = self.attention_c(a * b)  # [N, K]
        return att, x


class AttentionSingleBranch(nn.Module):
    """
    Flexible Attention MIL model supporting arbitrary depth.
    - Accepts size tuples like (1280, 512, 256) or (1280, 2048, 1024, 512, 256)
    - Outputs both class logits and a [1, 1280] slide embedding.
    """
    def __init__(self, size=(1280, 512, 256), use_dropout=False, n_classes=2):
        super().__init__()

        assert len(size) >= 3, "Size tuple must have at least 3 elements: (input, hidden, attention_bottleneck)"
        self.n_classes = n_classes
        self.use_dropout = use_dropout
        self.size = size

        # --- 1️⃣ Build flexible MLP for patch embedding ---
        fc_layers = []
        for i in range(len(size) - 2):  # all except last two (attention dims)
            fc_layers.append(nn.Linear(size[i], size[i + 1]))
            fc_layers.append(nn.ReLU())
            if use_dropout:
                fc_layers.append(nn.Dropout(0.25))
        self.patch_mlp = nn.Sequential(*fc_layers)

        # --- 2️⃣ Gated attention ---
        att_input_dim = size[-2]
        att_bottleneck = size[-1]
        self.attention = GatedAttention(
            input_dim=att_input_dim,
            bottleneck_dim=att_bottleneck,
            dropout=use_dropout,
            n_branches=1,
        )

        # --- 3️⃣ Projection back to original input dim (for comparability) ---
        in_dim = size[0]  # original patch feature dim, e.g., 1280
        self.slide_projector = nn.Linear(att_input_dim, in_dim)

        # --- 4️⃣ Classifier operating on projected slide-level features ---
        self.classifier = nn.Linear(in_dim, n_classes)

        # --- Init weights ---
        self.apply(initialize_weights)

    def relocate(self):
        """Moves model to CUDA if available."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

    def _compute_attention(self, x):
        """Computes attention scores and normalized weights."""
        x_embed = self.patch_mlp(x)  # patch-level MLP
        att_raw, x_transformed = self.attention(x_embed)
        att = torch.transpose(att_raw, 2, 1)  # [1, 1, N]
        att_soft = F.softmax(att, dim=2)
        return x_transformed, att_raw, att_soft

    def forward(self, x):
        """
        Args:
            x: [N_patches, in_dim]
        Returns:
            logits: [1, n_classes]
            results_dict: {
                "attention": raw attention scores,
                "slide_embedding": [1, in_dim]
            }
        """
        x, att_raw, att = self._compute_attention(x)
        pooled_features = torch.bmm(att, x).squeeze(1)  # [1, att_input_dim]

        # Slide-level embedding (projected back to e.g. 1280-D)
        slide_embedding = self.slide_projector(pooled_features)  # [1, in_dim]

        # Classification
        logits = self.classifier(slide_embedding)  # [1, n_classes]

        results_dict = {
            "attention": att_raw,
            "slide_embedding": slide_embedding.detach()
        }

        return logits, results_dict




def cross_validate_mil(
    splits_csv,
    features_dir,
    coord_dir,
    output_dir,
    n_classes=2,
    epochs=10,
    lr=1e-4,
    batch_size=1,
    penalty_factor=2.0,
    size=(1280, 512, 256),
    device=None,
    weighted=False,
    save_embeddings=True,
):
    """
    Perform 5-fold cross-validation using AttentionSingleBranch (MIL),
    with optional class-weighted loss, confidence penalty, and embedding saving.
    """

    os.makedirs(f"{output_dir}/models", exist_ok=True)
    os.makedirs(f"{output_dir}/attentions", exist_ok=True)
    os.makedirs(f"{output_dir}/results/plots", exist_ok=True)
    if save_embeddings:
        os.makedirs(f"{output_dir}/embeddings", exist_ok=True)

    splits = pd.read_csv(splits_csv)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_results = []
    fold_metrics = []
    all_train_losses, all_val_losses = [], []

    for fold in range(1, 6):
        print(f"\n===== FOLD {fold} =====")

        val_ids = splits.loc[splits["fold"] == fold, "slide_id"].tolist()
        train_ids = splits.loc[splits["fold"] != fold, "slide_id"].tolist()

        train_dataset = MILSlideDataset(splits_csv, features_dir, coord_dir, slide_ids=train_ids)
        val_dataset   = MILSlideDataset(splits_csv, features_dir, coord_dir, slide_ids=val_ids)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # === Compute class weights if requested ===
        train_labels = [train_dataset._encode_label(row["Diagnose"]) for _, row in train_dataset.df.iterrows()]
        counts = Counter(train_labels)
        total = sum(counts.values())

        if weighted:
            weights = np.array([total / counts[c] for c in sorted(counts)])
            weights = weights / weights.sum() * len(weights)
            class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
            print(f"Fold {fold} class counts: {dict(counts)} → weights = {np.round(weights, 2).tolist()}")
        else:
            class_weights = None

        # === Model setup ===
        model = AttentionSingleBranch(size=size, n_classes=n_classes).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        best_val_loss = float("inf")
        best_model_path = f"{output_dir}/models/mil_best_fold{fold}.pt"

        print(f"🧠 Using penalty_factor={penalty_factor}")
        train_losses, val_losses = [], []

        # === Training ===
        for epoch in range(1, epochs + 1):
            model.train()
            running_loss = 0.0

            for feats, label, _ in tqdm(train_loader, desc=f"[Fold {fold} | Epoch {epoch}] Train", leave=False):
                feats, label = feats.to(device), label.to(device)
                if label.ndim == 0:
                    label = label.unsqueeze(0)

                optimizer.zero_grad()
                logits, _ = model(feats)

                ce_loss = F.cross_entropy(
                    logits, label, reduction="none", weight=class_weights
                )

                probs = F.softmax(logits, dim=1)
                preds = probs.argmax(dim=1)
                confidences = probs[torch.arange(len(preds)), preds]
                correct = (preds == label).float()
                penalty = 1.0 + penalty_factor * (1.0 - correct) * confidences

                loss = (penalty * ce_loss).mean()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            train_loss = running_loss / len(train_loader)
            train_losses.append(train_loss)

            # === Validation ===
            model.eval()
            val_loss = 0.0
            y_true, y_pred, y_prob = [], [], []

            with torch.no_grad():
                for feats, label, meta in tqdm(val_loader, desc=f"[Fold {fold} | Epoch {epoch}] Val", leave=False):
                    feats, label = feats.to(device), label.to(device)
                    if label.ndim == 0:
                        label = label.unsqueeze(0)

                    logits, out = model(feats)
                    loss = F.cross_entropy(logits, label, weight=class_weights)
                    val_loss += loss.item()

                    probs = F.softmax(logits, dim=1).cpu().numpy()[0]
                    pred = int(np.argmax(probs))
                    true = int(label.cpu().numpy()[0])
                    slide_id = meta["slide_id"][0]

                    # ✅ Save attention + coords
                    coords = meta["coords"].squeeze(0).cpu().numpy() if isinstance(meta["coords"], torch.Tensor) else None
                    att_raw = out.get("attention")
                    if att_raw is not None:
                        #att = F.softmax(att_raw, dim=2).squeeze(0).squeeze(0).unsqueeze(1).cpu().numpy() #att_raw è [1,N,1] quindi questo è bug
                        att_raw_vec = att_raw.squeeze(0).squeeze(-1).detach().cpu().numpy()
                        att = F.softmax(att_raw, dim=1)
                        att = att.squeeze(0).squeeze(-1)
                        att = att.unsqueeze(1).cpu().numpy()  # [N, 1]
                        np.savez(f"{output_dir}/attentions/{slide_id}_att_with_coords.npz",
                                 attention=att, attention_raw=att_raw_vec, coords=coords)

                    if save_embeddings and "slide_embedding" in out:
                        emb = out["slide_embedding"].cpu().numpy()
                        np.save(f"{output_dir}/embeddings/{slide_id}_embedding.npy", emb)

                    y_true.append(true)
                    y_pred.append(pred)
                    y_prob.append(probs[1])

                val_loss /= len(val_loader)
                val_losses.append(val_loss)

            # === Metrics ===
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            auc_val = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan
            print(f"Fold {fold} | Epoch {epoch}: TrainLoss={train_loss:.4f} | ValLoss={val_loss:.4f} | "
                  f"ACC={acc:.3f} | F1={f1:.3f} | AUC={auc_val:.3f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), best_model_path)

        all_train_losses.append(train_losses)
        all_val_losses.append(val_losses)

        # === Final evaluation ===
        model.load_state_dict(torch.load(best_model_path))
        model.eval()
        y_true, y_pred, y_prob = [], [], []

        with torch.no_grad():
            for feats, label, meta in tqdm(val_loader, desc=f"[Fold {fold}] Final Eval", leave=False):
                feats = feats.to(device)
                logits, out = model(feats)
                probs = F.softmax(logits, dim=1).cpu().numpy()[0]
                pred = int(np.argmax(probs))
                true = int(label.cpu().numpy()[0])
                slide_id = meta["slide_id"][0]

                # ✅ Save embedding and attention one more time
                if save_embeddings and "slide_embedding" in out:
                    # emb = out["slide_embedding"].cpu().numpy()
                    torch.save(out["slide_embedding"].cpu(), f"{output_dir}/embeddings/{slide_id}_embedding.pt")
                    # np.save(f"{output_dir}/embeddings/{slide_id}_embedding.npy", emb)

                att_raw = out.get("attention")
                coords = meta["coords"].squeeze(0).cpu().numpy() if isinstance(meta["coords"], torch.Tensor) else None
                if att_raw is not None:
                    #att = F.softmax(att_raw, dim=2).squeeze(0).squeeze(0).unsqueeze(1).cpu().numpy() #come prima, sbagliato il softmax
                    att_raw_vec = att_raw.squeeze(0).squeeze(-1).detach().cpu().numpy()
                    att = F.softmax(att_raw, dim=1)
                    att = att.squeeze(0).squeeze(-1)
                    att = att.unsqueeze(1).cpu().numpy()  # [N, 1]
                    np.savez(f"{output_dir}/attentions/{slide_id}_att_with_coords.npz",
                             attention=att, attention_raw=att_raw_vec, coords=coords)

                y_true.append(true)
                y_pred.append(pred)
                y_prob.append(probs[1])
                all_results.append({
                    "fold": fold,
                    "slide_id": slide_id,
                    "true_label": true,
                    "pred_label": pred,
                    "prob_anaplasia": probs[1],
                })

        # === Confusion matrix per fold ===
        cm = confusion_matrix(y_true, y_pred)
        ConfusionMatrixDisplay(cm, display_labels=["Not Anaplasia", "Anaplasia"]).plot(cmap="Blues", values_format="d")
        plt.title(f"Confusion Matrix - Fold {fold}")
        plt.savefig(f"{output_dir}/results/plots/cm_fold{fold}.png", dpi=200, bbox_inches="tight")
        plt.close()

        # === Fold metrics ===
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        auc_val = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan

        fold_metrics.append({"fold": fold, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "auc": auc_val})

        # === Loss plot per fold ===
        plt.figure()
        plt.plot(train_losses, label="Train Loss")
        plt.plot(val_losses, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Loss per Epoch - Fold {fold}")
        plt.legend()
        plt.savefig(f"{output_dir}/results/plots/losses_fold{fold}.png", dpi=200, bbox_inches="tight")
        plt.close()

    # === Save outputs ===
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(f"{output_dir}/results/per_slide_predictions.csv", index=False)

    metrics_df = pd.DataFrame(fold_metrics)
    summary = metrics_df.mean(numeric_only=True).to_dict()
    summary_std = metrics_df.std(numeric_only=True).to_dict()
    summary_df = pd.DataFrame([summary], index=["mean"])
    summary_df.loc["std"] = summary_std

    summary_df.to_csv(f"{output_dir}/results/summary.csv")
    metrics_df.to_csv(f"{output_dir}/results/per_fold_metrics.csv", index=False)

    # === Average confusion + loss curves ===
    cm_all = confusion_matrix(results_df["true_label"], results_df["pred_label"])
    ConfusionMatrixDisplay(cm_all, display_labels=["Not Anaplasia", "Anaplasia"]).plot(cmap="Blues", values_format="d")
    plt.title("Average Confusion Matrix (All Folds)")
    plt.savefig(f"{output_dir}/results/plots/cm_average.png", dpi=200, bbox_inches="tight")
    plt.close()

    avg_train = np.nanmean(np.stack([np.pad(t, (0, epochs - len(t)), constant_values=np.nan) for t in all_train_losses]), axis=0)
    avg_val = np.nanmean(np.stack([np.pad(v, (0, epochs - len(v)), constant_values=np.nan) for v in all_val_losses]), axis=0)
    plt.figure()
    plt.plot(avg_train, label="Avg Train Loss")
    plt.plot(avg_val, label="Avg Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Average Loss Curve (across folds)")
    plt.legend()
    plt.savefig(f"{output_dir}/results/plots/losses_average.png", dpi=200, bbox_inches="tight")
    plt.close()

    print("\n✅ Cross-validation complete.")
    print(summary_df)

    return results_df, metrics_df, summary_df



def _find_wsi_path(wsi_dir: str, slide_id: str):
    """Try common extensions to locate the WSI."""
    wsi_dir = Path(wsi_dir)
    for ext in (".mrxs", ".svs", ".tif", ".tiff"):
        p = wsi_dir / f"{slide_id}{ext}"
        if p.exists():
            return str(p)
    raise FileNotFoundError(f"No WSI found for {slide_id} in {wsi_dir} (tried .mrxs/.svs/.tif/.tiff)")

def _load_att_npz(att_dir: str, slide_id: str):
    """Load attention+coords, trying filename WITHOUT fold first, then WITH fold."""
    att_dir = Path(att_dir)
    p_no_fold = att_dir / f"{slide_id}_att_with_coords.npz"
    if p_no_fold.exists():
        return np.load(p_no_fold)
    raise FileNotFoundError(f"No attention file for {slide_id} (tried {p_with_fold.name})")

def plot_attention_on_wsi(
    slide_id,
    att_dir,
    wsi_dir,
    thumbnail_level=5,
    mode="linear",                 # "linear" | "log" | "clipped"
    cmap_cv2="INFERNO",            # OpenCV colormap name (INFERNO, JET, etc.)
    patch_size=224,
    alpha=0.5,
    top_k=20,
    save_path=None,
    return_image=False,            # 👈 NEW
):
    """
    Overlay attention on a WSI thumbnail; draw rectangles on top-K patches.
    If return_image=True, returns the overlay (RGB NumPy array) instead of showing/saving.
    """
    slide_path = _find_wsi_path(wsi_dir, slide_id)
    slide = openslide.OpenSlide(slide_path)
    level_downsample = slide.level_downsamples[thumbnail_level]
    thumb_w, thumb_h = slide.level_dimensions[thumbnail_level]
    thumbnail = np.array(slide.read_region((0, 0), thumbnail_level, (thumb_w, thumb_h)))[:, :, :3]

    # --- load attention and coords ---
    data = _load_att_npz(att_dir, slide_id)
    att = data["attention"].squeeze()
    coords = data["coords"]

    coords_scaled = coords / level_downsample

    # --- normalize attention ---
    if mode == "linear":
        att_vis = att / (att.max() + 1e-8)
    elif mode == "log":
        att_vis = np.log1p(att / (att.max() + 1e-8))
        att_vis = (att_vis - att_vis.min()) / (att_vis.max() - att_vis.min() + 1e-8)
    elif mode == "clipped":
        q_low, q_high = np.quantile(att, [0.80, 0.999])
        att_clipped = np.clip(att, q_low, q_high)
        att_vis = (att_clipped - q_low) / (q_high - q_low + 1e-8)
    else:
        raise ValueError("mode must be one of {'linear','log','clipped'}")

    # --- build heatmap overlay ---
    heatmap = np.zeros(thumbnail.shape[:2], dtype=np.float32)
    coords_int = coords_scaled.astype(int)
    H, W = heatmap.shape
    for (x, y), w in zip(coords_int, att_vis):
        if 0 <= x < W and 0 <= y < H:
            heatmap[y, x] = max(heatmap[y, x], w)

    heatmap = cv2.GaussianBlur(heatmap, (25, 25), 0)
    cv2_cmap = getattr(cv2, f"COLORMAP_{cmap_cv2.upper()}", cv2.COLORMAP_INFERNO)
    heatmap_color = cv2.applyColorMap((255 * heatmap).astype(np.uint8), cv2_cmap)
    overlay = cv2.addWeighted(thumbnail, 1 - alpha, heatmap_color, alpha, 0)

    # --- draw rectangles on top-K patches ---
    side = patch_size / level_downsample
    for (x, y) in coords_scaled[np.argsort(att)[-top_k:]]:
        x0, y0 = int(x - side / 2), int(y - side / 2)
        x1, y1 = int(x + side / 2), int(y + side / 2)
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 255, 0), 1)

    # === RETURN IMAGE MODE ===
    if return_image:
        return overlay  # return raw RGB overlay for subplot use

    # === NORMAL DISPLAY/SAVE MODE ===
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(overlay)
    ax.axis("off")
    ax.set_title(f"{slide_id} — {mode}")

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

def generate_all_attention_reports_flat(
    base_exp_dir,
    wsi_dir,
    output_dir=None,
    patch_size=224,
    context_factor=2.0,
    top_k=20,
    thumbnail_level=5,
    cmap_cv2="INFERNO",
    results_csv=None,
    gc_every=5,
):
    """
    Memory-safe generation of 2×2 composite visualizations per slide.
    - Closes OpenSlide handles after use.
    - Clears matplotlib figures.
    - Forces garbage collection periodically.
    - Automatically skips slides whose combined image and top patches already exist.
    """

    base = Path(base_exp_dir)
    att_dir = base / "attentions"
    out_root = Path(output_dir or (base / "visual_reports"))
    out_root.mkdir(parents=True, exist_ok=True)

    # Load mapping slide_id -> fold
    results_csv = results_csv or (base / "results" / "per_slide_predictions.csv")
    df = pd.read_csv(results_csv)
    fold_map = {sid: int(f) for sid, f in zip(df["slide_id"], df["fold"])}

    npz_files = sorted(att_dir.glob("*_att_with_coords.npz"))
    print(f"🧭 Found {len(npz_files)} attention files")

    for idx, npz_path in enumerate(tqdm(npz_files, desc="Generating reports")):
        slide_id = npz_path.stem.replace("_att_with_coords", "")
        fold = fold_map.get(slide_id, None)
        fold_dir = out_root / (f"fold{fold}" if fold else "fold_unknown")

        vis_dir = out_root / "attention_maps"
        patch_dir = out_root / slide_id / "top_patches"
        vis_dir.mkdir(parents=True, exist_ok=True)
        patch_dir.mkdir(parents=True, exist_ok=True)

        save_path = vis_dir / f"{slide_id}_combined.png"

        # ✅ Skip automatically if outputs already exist
        patches_exist = patch_dir.exists() and any(patch_dir.glob("*.png"))
        if save_path.exists() and patches_exist:
            print(f"⏭️  Skipping {slide_id}: combined image and patches already exist.")
            continue

        npz_data = np.load(npz_path)
        att = npz_data["attention"].squeeze()
        coords = npz_data["coords"]
        npz_data.close()  # free NPZ memory

        try:
            slide_path = _find_wsi_path(wsi_dir, slide_id)
            slide = openslide.OpenSlide(slide_path)

            fig, axes = plt.subplots(2, 2, figsize=(10, 10))
            fig.suptitle(f"{slide_id} — Attention Overview", fontsize=12)
            modes = ["linear", "log", "clipped"]

            # --- Top row: attention maps ---
            for ax, mode in zip(axes[0], modes):
                try:
                    img = plot_attention_on_wsi(
                        slide_id=slide_id,
                        att_dir=str(att_dir),
                        wsi_dir=wsi_dir,
                        thumbnail_level=thumbnail_level,
                        mode=mode,
                        cmap_cv2=cmap_cv2,
                        patch_size=patch_size,
                        alpha=0.5,
                        top_k=top_k,
                        return_image=True,
                    )
                    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    ax.set_title(mode.capitalize(), fontsize=10)
                    ax.axis("off")
                    del img
                except Exception as e:
                    ax.text(0.5, 0.5, f"Failed {mode}\n{e}", ha="center", va="center")
                    ax.axis("off")

            # --- Bottom-right: top patches ---
            try:
                top_idx = np.argsort(att)[-top_k:]
                coords_top = coords[top_idx]
                att_top = att[top_idx]

                crop_size = int(patch_size * context_factor)
                patch_images = []

                for i, ((x, y), score) in enumerate(zip(coords_top, att_top)):
                    x0, y0 = int(x - crop_size / 2), int(y - crop_size / 2)
                    region = slide.read_region((x0, y0), 0, (crop_size, crop_size)).convert("RGB")
                    patch_path = patch_dir / f"{slide_id}_top{i+1:02d}_att{score:.4e}.png"
                    region.save(patch_path)  # ✅ save immediately
                    patch_images.append(np.array(region))
                    region.close()
                slide.close()

                # small preview grid for subplot
                cols = int(np.ceil(np.sqrt(len(patch_images))))
                rows = int(np.ceil(len(patch_images) / cols))
                patch_h, patch_w = patch_images[0].shape[:2]
                grid = np.ones((rows * patch_h, cols * patch_w, 3), dtype=np.uint8) * 255
                for i, patch in enumerate(patch_images):
                    r, c = divmod(i, cols)
                    grid[r * patch_h:(r + 1) * patch_h, c * patch_w:(c + 1) * patch_w] = patch
                axes[1, 1].imshow(grid)
                axes[1, 1].set_title(f"Top {top_k} patches", fontsize=10)
                axes[1, 1].axis("off")
                del patch_images, grid

            except Exception as e:
                axes[1, 1].text(0.5, 0.5, f"Patch fail\n{e}", ha="center", va="center")
                axes[1, 1].axis("off")

            # --- Bottom-left: placeholder ---
            axes[1, 0].text(0.5, 0.5, "Summary / Metrics", ha="center", va="center")
            axes[1, 0].axis("off")

            plt.tight_layout()
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            del fig, axes

        except Exception as e:
            print(f"⚠️ Failed for {slide_id}: {e}")
            continue

        # ✅ Free memory aggressively every few slides
        if (idx + 1) % gc_every == 0:
            gc.collect()

    print("✅ All attention reports generated safely.")


# def to_percentiles(x: np.ndarray) -> np.ndarray:
#     # CLAM usa percentili (0-100) :contentReference[oaicite:4]{index=4}
#     x = x.astype(np.float32)
#     r = rankdata(x, method="average")  # 1..N
#     return 100.0 * (r - 1) / (len(x) - 1 + 1e-12)

def to_percentiles_0_1(x: np.ndarray) -> np.ndarray:
    """Return percentiles in [0,1]."""
    x = x.astype(np.float32).reshape(-1)
    if len(x) < 2:
        return np.zeros_like(x)
    r = rankdata(x, method="average")  # 1..N
    return (r - 1) / (len(x) - 1 + 1e-12)

def _combine_subplot(
    left: Image.Image,
    right: Image.Image,
    layout="horizontal",
    pad=30,
    bg=(255, 255, 255),
    title=None,
    title_height=60
):
    left = left.convert("RGB")
    right = right.convert("RGB")

    if layout == "horizontal":
        if left.height != right.height:
            scale = left.height / right.height
            right = right.resize((int(right.width * scale), left.height), Image.BILINEAR)

        W = left.width + pad + right.width
        H = left.height
        top_pad = title_height if title else 0
        canvas = Image.new("RGB", (W, H + top_pad), bg)
        canvas.paste(left, (0, top_pad))
        canvas.paste(right, (left.width + pad, top_pad))

    elif layout == "vertical":
        if left.width != right.width:
            scale = left.width / right.width
            right = right.resize((left.width, int(right.height * scale)), Image.BILINEAR)

        W = left.width
        H = left.height + pad + right.height
        top_pad = title_height if title else 0
        canvas = Image.new("RGB", (W, H + top_pad), bg)
        canvas.paste(left, (0, top_pad))
        canvas.paste(right, (0, top_pad + left.height + pad))
    else:
        raise ValueError("layout must be 'horizontal' or 'vertical'")

    if title:
        draw = ImageDraw.Draw(canvas)
        draw.text((pad, 10), title, fill=(0, 0, 0))

    return canvas


    
# def _extract_top_patches_grid(
#     wsi: openslide.OpenSlide,
#     coords_lvl0: np.ndarray,    # (N,2) top-left in level-0
#     scores: np.ndarray,         # (N,)
#     k: int = 20,
#     patch_level: int = 1,       # <-- per te probabilmente 1 (0.5 mpp)
#     patch_size: int = 224,      # dimensione patch a patch_level
#     grid: tuple = (4, 5),       # 4 righe x 5 colonne = 20
#     pad: int = 6
# ) -> tuple[Image.Image, np.ndarray]:
#     """
#     Ritorna (grid_img, top_idx). Le patch sono estratte dalla WSI al patch_level,
#     con size=patch_size (pixel al patch_level).
#     """
#     scores = scores.reshape(-1)
#     top_idx = np.argsort(scores)[::-1][:k]

#     rows, cols = grid
#     assert rows * cols >= k, "grid troppo piccola per k"

#     tile_w = patch_size
#     tile_h = patch_size
#     grid_w = cols * tile_w + (cols + 1) * pad
#     grid_h = rows * tile_h + (rows + 1) * pad

#     canvas = Image.new("RGB", (grid_w, grid_h), (255, 255, 255))

#     for j, idx in enumerate(top_idx):
#         r = j // cols
#         c = j % cols
#         x0 = pad + c * (tile_w + pad)
#         y0 = pad + r * (tile_h + pad)

#         x, y = coords_lvl0[idx].astype(int).tolist()  # top-left level0
#         patch = wsi.read_region((x, y), patch_level, (patch_size, patch_size)).convert("RGB")
#         canvas.paste(patch, (x0, y0))

#     return canvas, top_idx

def _extract_top_regions_grid(
    wsi,
    coords_lvl0,
    scores,
    k=20,
    patch_level=1,
    patch_size=224,
    region_size=1024,
    grid=(4,5),
    pad=6
):
    """
    Estrae top-k region 1024x1024 (patch_level),
    con la patch evidenziata al centro.
    """

    scores = scores.reshape(-1)
    top_idx = np.argsort(scores)[::-1][:k]

    ds_patch = float(wsi.level_downsamples[patch_level])

    rows, cols = grid
    tile_w = region_size
    tile_h = region_size

    canvas_w = cols * tile_w + (cols + 1) * pad
    canvas_h = rows * tile_h + (rows + 1) * pad
    canvas = Image.new("RGB", (canvas_w, canvas_h), (255,255,255))

    for j, idx in enumerate(top_idx):
        r = j // cols
        c = j % cols

        x_pad = pad + c * (tile_w + pad)
        y_pad = pad + r * (tile_h + pad)

        x_lvl0, y_lvl0 = coords_lvl0[idx]

        # patch center in level 0
        patch_size_lvl0 = patch_size * ds_patch
        cx_lvl0 = x_lvl0 + patch_size_lvl0 / 2
        cy_lvl0 = y_lvl0 + patch_size_lvl0 / 2

        # region top-left in level 0
        region_size_lvl0 = region_size * ds_patch
        rx_lvl0 = int(cx_lvl0 - region_size_lvl0 / 2)
        ry_lvl0 = int(cy_lvl0 - region_size_lvl0 / 2)

        region = wsi.read_region(
            (rx_lvl0, ry_lvl0),
            patch_level,
            (region_size, region_size)
        ).convert("RGB")

        draw = ImageDraw.Draw(region)

        # rectangle for patch inside region (centered)
        patch_offset = (region_size - patch_size) // 2
        draw.rectangle(
            [
                patch_offset,
                patch_offset,
                patch_offset + patch_size,
                patch_offset + patch_size
            ],
            outline="black",
            width=3
        )

        canvas.paste(region, (x_pad, y_pad))

    return canvas


def clam_like_heatmap_with_topk_boxes(
    slide_path: str,
    coords_lvl0: np.ndarray,          # (N,2) top-left in level-0
    scores_raw: np.ndarray,           # (N,) raw logits (consigliato)
    vis_level: int = -1,
    patch_level: int = 1,             # <-- per te: 1
    patch_size: int = 224,            # patch size al patch_level
    alpha: float = 0.45,
    cmap_name: str = "inferno",
    convert_to_percentiles: bool = True,
    clip_percentiles: tuple = (5, 99),
    max_size: int | None = 4096,
    draw_topk: int = 20,
    box_width: int = 2,
):
    wsi = openslide.open_slide(slide_path)

    if vis_level < 0:
        vis_level = wsi.get_best_level_for_downsample(32)

    # downsample factors
    ds_vis = float(wsi.level_downsamples[vis_level])
    ds_patch = float(wsi.level_downsamples[patch_level])

    # patch size in level-0 pixels (per disegno rettangoli)
    patch_size_lvl0 = patch_size * ds_patch  # es: 224 * 2 = 448
    scale_vis = 1.0 / ds_vis

    # coords scaled to vis_level
    coords_vis = np.ceil(coords_lvl0 * scale_vis).astype(int)

    # patch size at vis_level (rettangoli)
    pw_vis = int(np.ceil(patch_size_lvl0 * scale_vis))
    ph_vis = int(np.ceil(patch_size_lvl0 * scale_vis))

    # prepare scores for heatmap
    s = scores_raw.astype(np.float32).reshape(-1)
    if convert_to_percentiles:
        s = to_percentiles_0_1(s)  # 0..1

    W, H = wsi.level_dimensions[vis_level]
    overlay = np.zeros((H, W), dtype=np.float32)

    # MAX pooling patch-aware
    for (x, y), val in zip(coords_vis, s):
        x0 = max(0, int(x))
        y0 = max(0, int(y))
        x1 = min(W, x0 + pw_vis)
        y1 = min(H, y0 + ph_vis)
        if x0 >= W or y0 >= H or x1 <= 0 or y1 <= 0:
            continue
        overlay[y0:y1, x0:x1] = np.maximum(overlay[y0:y1, x0:x1], float(val))

    # contrast stretch
    nz = overlay > 0
    if np.any(nz):
        lo, hi = np.percentile(overlay[nz], clip_percentiles)
        overlay = np.clip(overlay, lo, hi)
        overlay = (overlay - overlay.min()) / (overlay.max() - overlay.min() + 1e-12)

    # read base image at vis_level
    base = np.array(wsi.read_region((0, 0), vis_level, (W, H)).convert("RGB"))

    # colormap + blend
    cmap = plt.get_cmap(cmap_name)
    color = (cmap(overlay)[:, :, :3] * 255).astype(np.uint8)
    out = (base.astype(np.float32) * (1 - alpha) + color.astype(np.float32) * alpha).astype(np.uint8)
    out_img = Image.fromarray(out)

    # draw top-k rectangles (based on RAW logits ranking, not percentiles)
    raw = scores_raw.reshape(-1)
    top_idx = np.argsort(raw)[::-1][:draw_topk]
    draw = ImageDraw.Draw(out_img)
    for idx in top_idx:
        x, y = coords_vis[idx]
        x0 = int(x)
        y0 = int(y)
        x1 = int(min(W - 1, x0 + pw_vis))
        y1 = int(min(H - 1, y0 + ph_vis))
        draw.rectangle([x0, y0, x1, y1], outline="black", width=box_width)

    # resize for saving
    if max_size is not None:
        w, h = out_img.size
        if max(w, h) > max_size:
            f = max_size / max(w, h)
            out_img = out_img.resize((int(w * f), int(h * f)), Image.BILINEAR)

    return out_img, top_idx




def clam_like_heatmap(
    slide_path: str,
    coords_lvl0: np.ndarray,          # (N,2) in level-0 coords (top-left)
    scores: np.ndarray,               # (N,) raw logits preferred
    vis_level: int = -1,
    patch_size_lvl0: int | tuple = 224,
    alpha: float = 0.45,
    blank_canvas: bool = False,
    convert_to_percentiles: bool = True,
    blur: bool = False,
    overlap: float = 0.0,
    cmap_name: str = "inferno",
    max_size: int | None = 4096,
    clip_percentiles: tuple = (5, 99),
):
    wsi = openslide.open_slide(slide_path)

    if vis_level < 0:
        vis_level = wsi.get_best_level_for_downsample(32)

    downsample = float(wsi.level_downsamples[vis_level])
    scale = np.array([1.0 / downsample, 1.0 / downsample], dtype=np.float32)

    # patch size
    if isinstance(patch_size_lvl0, int):
        patch_size_lvl0 = (patch_size_lvl0, patch_size_lvl0)  # (w,h)
    patch_size_lvl = np.ceil(np.array(patch_size_lvl0) * scale).astype(int)
    pw, ph = int(patch_size_lvl[0]), int(patch_size_lvl[1])

    # coords to vis level
    coords = np.ceil(coords_lvl0 * scale).astype(int)

    # scores
    scores = scores.astype(np.float32).reshape(-1)
    if convert_to_percentiles:
        scores = to_percentiles_0_1(scores)  # already 0..1

    W, H = wsi.level_dimensions[vis_level]  # (W,H)

    overlay = np.zeros((H, W), dtype=np.float32)

    # ---- MAX pooling over patch rectangles ----
    for (x, y), s in zip(coords, scores):
        x0 = max(0, int(x))
        y0 = max(0, int(y))
        x1 = min(W, x0 + pw)
        y1 = min(H, y0 + ph)
        if x0 >= W or y0 >= H or x1 <= 0 or y1 <= 0:
            continue
        overlay[y0:y1, x0:x1] = np.maximum(overlay[y0:y1, x0:x1], float(s))

    # optional blur (usually not needed with MAX)
    if blur:
        kx = max(3, (int(pw * (1 - overlap)) * 2 + 1) | 1)
        ky = max(3, (int(ph * (1 - overlap)) * 2 + 1) | 1)
        overlay = cv2.GaussianBlur(overlay, (kx, ky), 0)

    # ---- contrast stretch on nonzero region ----
    nz = overlay > 0
    if np.any(nz):
        lo, hi = np.percentile(overlay[nz], clip_percentiles)
        overlay = np.clip(overlay, lo, hi)
        overlay = (overlay - overlay.min()) / (overlay.max() - overlay.min() + 1e-12)

    # canvas
    region_size = (W, H)
    if blank_canvas:
        img = np.ones((H, W, 3), dtype=np.uint8) * 255
    else:
        img = np.array(wsi.read_region((0, 0), vis_level, region_size).convert("RGB"))

    # colormap + alpha
    cmap = plt.get_cmap(cmap_name)
    color = (cmap(overlay)[:, :, :3] * 255).astype(np.uint8)
    out = (img.astype(np.float32) * (1 - alpha) + color.astype(np.float32) * alpha).astype(np.uint8)
    out_img = Image.fromarray(out)

    # resize
    if max_size is not None:
        w, h = out_img.size
        if max(w, h) > max_size:
            f = max_size / max(w, h)
            out_img = out_img.resize((int(w * f), int(h * f)), Image.BILINEAR)

    return out_img

def generate_all_attention_reports(
    base_exp_dir,
    wsi_dir,
    patch_size=224,
    patch_level=1,
    vis_level=-1,
    alpha=0.45,
    convert_to_percentiles=True,
    max_size=4096,
    use_raw=True,
    extract_region=False,
    draw_topk=20,
    combine_subplots=True,           # <-- NUOVO FLAG
    subplot_layout="horizontal"      # <-- NUOVO FLAG
):
    att_dir = os.path.join(base_exp_dir, "attentions")
    out_dir = os.path.join(base_exp_dir, "visual_reports")
    os.makedirs(out_dir, exist_ok=True)

    files = [f for f in os.listdir(att_dir) if f.endswith(".npz")]
    print(f"🧭 Found {len(files)} attention files")

    for fname in tqdm(files, desc="Generating attention reports"):
        data = np.load(os.path.join(att_dir, fname))

        coords = data["coords"].astype(np.float32)

        if use_raw and "attention_raw" in data:
            scores_raw = data["attention_raw"].astype(np.float32).reshape(-1)
        else:
            scores_raw = data["attention"].astype(np.float32).reshape(-1)

        slide_id = fname.replace("_att_with_coords.npz", "")
        slide_path = os.path.join(wsi_dir, slide_id + ".mrxs")

        if not os.path.exists(slide_path):
            print(f"⚠️ Slide not found: {slide_path}")
            continue

        # --- heatmap con topk box ---
        heatmap_img, top_idx = clam_like_heatmap_with_topk_boxes(
            slide_path=slide_path,
            coords_lvl0=coords,
            scores_raw=scores_raw,
            vis_level=vis_level,
            patch_level=patch_level,
            patch_size=patch_size,
            alpha=alpha,
            convert_to_percentiles=convert_to_percentiles,
            max_size=max_size,
            draw_topk=draw_topk,
        )

        

        # --- top patches grid ---
        wsi = openslide.open_slide(slide_path)
        if extract_region:
            topk = 6
            grid_img = _extract_top_regions_grid(
                wsi=wsi,
                coords_lvl0=coords,
                scores=scores_raw,
                k=topk,
                patch_level=patch_level,
                patch_size=patch_size,
                region_size=1024,
                grid=(3, 2),     # 5 regioni orizzontali
                pad=8,
            )
        else:
            topk=20
            grid_img, _ = _extract_top_patches_grid(
                wsi=wsi,
                coords_lvl0=coords,
                scores=scores_raw,
                k=draw_topk,
                patch_level=patch_level,
                patch_size=patch_size,
                grid=(4, 5),
                pad=6,
            )



        # --- subplot combine (opzionale) ---
        if combine_subplots:
            combined = _combine_subplot(
                heatmap_img,
                grid_img,
                layout=subplot_layout,
                pad=30,
                title=f"{slide_id} | Heatmap + Top {draw_topk} patches"
            )
            max_combined_size=9000
            if max(combined.size) > max_combined_size:
                f = max_combined_size / max(combined.size)
                combined = combined.resize(
                    (int(combined.width * f), int(combined.height * f)),
            Image.BILINEAR
                )
            combined_path = os.path.join(out_dir, f"{slide_id}_combined.jpg")
            combined.save(combined_path,quality=92,optimize=True,progressive=True)
        else:
            heatmap_path = os.path.join(out_dir, f"{slide_id}_heatmap_top{draw_topk}.jpg")
            heatmap_img.save(heatmap_path,quality=92,optimize=True,progressive=True)
            grid_path = os.path.join(out_dir, f"{slide_id}_top{draw_topk}_patches.jpg")
            grid_img.save(grid_path,quality=92,optimize=True,progressive=True)
        wsi.close()

    print("✅ Attention reports generated.")