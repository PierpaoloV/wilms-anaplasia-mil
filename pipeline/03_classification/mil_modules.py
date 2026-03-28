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

from rich.progress import (
    Progress, BarColumn, TextColumn, TimeElapsedColumn,
    MofNCompleteColumn, SpinnerColumn, track,
)
from rich.console import Console
from rich.table import Table

console = Console()
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
        feats = torch.load(path, weights_only=True)  # expected tensor [N, D]
        if feats.dtype != torch.float32:
            feats = feats.float()
        return feats

    def _load_coords(self, slide_id):
        npy_path = self.coord_dir / f"{slide_id}.npy"
        if not npy_path.exists():
            print(f"Coordinates not found: {npy_path}")
            return None
        arr = np.load(npy_path, allow_pickle=False)
        # Structured array with named x/y fields (legacy format)
        if arr.dtype.names and "x" in arr.dtype.names and "y" in arr.dtype.names:
            return np.column_stack((arr["x"], arr["y"])).astype(np.float32)
        # Plain (N, 2) array as output by slide2vec
        if arr.ndim == 2 and arr.shape[1] >= 2:
            return arr[:, :2].astype(np.float32)
        print(f"Unrecognised coordinate format for {slide_id}: shape={arr.shape} dtype={arr.dtype}")
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


def _save_attention(output_dir, slide_id, out, meta):
    """Save softmax attention scores + raw logits + coordinates to .npz."""
    att_raw = out.get("attention")
    coords = meta["coords"].squeeze(0).cpu().numpy() if isinstance(meta["coords"], torch.Tensor) else None
    if att_raw is not None:
        att_raw_vec = att_raw.squeeze(0).squeeze(-1).detach().cpu().numpy()
        att = F.softmax(att_raw, dim=1).squeeze(0).squeeze(-1).unsqueeze(1).cpu().numpy()  # [N, 1]
        np.savez(
            f"{output_dir}/attentions/{slide_id}_att_with_coords.npz",
            attention=att, attention_raw=att_raw_vec, coords=coords,
        )


def get_coords(meta):
    """Extract coords array from a DataLoader batch meta dict."""
    coords = meta.get("coords", None)
    if isinstance(coords, (list, tuple)):
        coords = coords[0] if len(coords) > 0 else None
    if coords is None:
        return None
    if torch.is_tensor(coords):
        return coords.squeeze(0).cpu().numpy()
    return coords


def generate_experiment_reports(
    experiment_dir,
    cfg,
    extract_region=False,
    combine_subplots=True,
    subplot_layout="horizontal",
):
    """Generate visual attention reports for all NPZs in experiment_dir/inference/."""
    fold_out    = os.path.join(experiment_dir, "inference")
    wsi_dir     = cfg["wsi_dir"]
    results_csv = os.path.join(experiment_dir, "results", "per_slide_predictions.csv")
    generate_all_attention_reports(
        base_exp_dir=fold_out,
        wsi_dir=wsi_dir,
        patch_size=int(cfg.get("patch_size", 224)),
        patch_level=int(cfg.get("patch_level", 1)),
        vis_level=int(cfg.get("vis_level", -1)),
        alpha=float(cfg.get("alpha", 0.6)),
        cmap_name=cfg.get("cmap_name", "plasma"),
        convert_to_percentiles=bool(cfg.get("convert_to_percentiles", True)),
        max_size=int(cfg.get("max_size", 4096)),
        use_raw=True,
        extract_region=extract_region,
        draw_topk=int(cfg.get("draw_topk", 20)),
        combine_subplots=combine_subplots,
        subplot_layout=subplot_layout,
        results_csv=results_csv if os.path.exists(results_csv) else None,
        num_workers=cfg.get("report_workers", None),
    )


def run_inference_fold(
    experiment_dir,
    fold,
    cfg,
    device="cuda",
    generate_reports=True,
    extract_region=False,
    combine_subplots=True,
    subplot_layout="horizontal",
):
    """
    Load the best saved model for a fold, run forward pass on the validation
    slides, and save attention NPZs.  When generate_reports=True (default)
    also renders visual reports immediately after.  Pass generate_reports=False
    when looping over all folds so that report generation happens once at the
    end rather than once per fold.
    """
    labels_csv = cfg.get("labels_csv") or cfg.get("labels_dir")
    features_dir = os.path.join(cfg["base_dir"], "features")
    coord_dir    = os.path.join(cfg["base_dir"], "coordinates")
    wsi_dir      = cfg["wsi_dir"]

    df       = pd.read_csv(labels_csv)
    fold_ids = df.loc[df["fold"] == fold, "slide_id"].astype(str).tolist()

    if not fold_ids:
        print(f"  No slides for fold {fold}")
        return

    model_path = os.path.join(experiment_dir, "models", f"mil_best_fold{fold}.pt")
    if not os.path.exists(model_path):
        print(f"  Model missing: {model_path}")
        return

    fold_out = os.path.join(experiment_dir, "inference")
    att_dir  = os.path.join(fold_out, "attentions")
    emb_dir  = os.path.join(fold_out, "embeddings")
    for d in [fold_out, att_dir, emb_dir]:
        os.makedirs(d, exist_ok=True)

    ds     = MILSlideDataset(labels_csv, features_dir, coord_dir, slide_ids=fold_ids)
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    dev   = torch.device(device if torch.cuda.is_available() else "cpu")
    model = AttentionSingleBranch(
        size=tuple(cfg["size"]),
        use_dropout=cfg.get("use_dropout", False),
        n_classes=cfg.get("n_classes", 2),
    )
    model.load_state_dict(torch.load(model_path, map_location=dev, weights_only=True))
    model.to(dev)
    model.eval()

    with torch.no_grad():
        for feats, label, meta in track(loader, description=f"  Fold {fold} inference", console=console):
            feats     = feats.to(dev)
            _, out    = model(feats)
            slide_id  = meta["slide_id"][0]
            coords    = get_coords(meta)

            att_raw     = out["attention"]
            att_raw_vec = att_raw.squeeze(0).squeeze(-1).cpu().numpy()
            att_soft_vec = F.softmax(att_raw, dim=1).squeeze(0).squeeze(-1).cpu().numpy()

            np.savez(
                os.path.join(att_dir, f"{slide_id}_att_with_coords.npz"),
                attention=att_soft_vec,
                attention_raw=att_raw_vec,
                coords=coords,
            )

            if "slide_embedding" in out:
                np.save(
                    os.path.join(emb_dir, f"{slide_id}_embedding.npy"),
                    out["slide_embedding"].cpu().numpy(),
                )

    if generate_reports:
        generate_experiment_reports(
            experiment_dir,
            cfg,
            extract_region=extract_region,
            combine_subplots=combine_subplots,
            subplot_layout=subplot_layout,
        )


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
    weight_decay=1e-4,
    gmean_threshold=0.55,
    logger=None,
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

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=28),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
        refresh_per_second=5,
    ) as progress:

      fold_task = progress.add_task("[bold cyan]Cross-validation", total=5)

      for fold in range(1, 6):
        val_ids = splits.loc[splits["fold"] == fold, "slide_id"].tolist()
        train_ids = splits.loc[splits["fold"] != fold, "slide_id"].tolist()

        train_pos = splits.loc[(splits["fold"] != fold) & (splits["Diagnose"] != "Not Anaplasia"), "slide_id"].count()
        val_pos   = splits.loc[(splits["fold"] == fold) & (splits["Diagnose"] != "Not Anaplasia"), "slide_id"].count()
        progress.console.rule(f"[bold]Fold {fold} / 5")
        progress.console.print(
            f"  [dim]train: {len(train_ids)} slides, {train_pos} pos  |  "
            f"val: {len(val_ids)} slides, {val_pos} pos[/dim]"
        )
        if logger:
            logger.info(f"--- Fold {fold} / 5 | train: {len(train_ids)} slides, {train_pos} pos | val: {len(val_ids)} slides, {val_pos} pos ---")

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
            progress.console.print(
                f"  Class counts: {dict(counts)} → weights = {np.round(weights, 2).tolist()}"
            )
        else:
            class_weights = None

        # === Model setup ===
        model = AttentionSingleBranch(size=size, n_classes=n_classes).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        best_val_auc = -1.0
        best_model_path = f"{output_dir}/models/mil_best_fold{fold}.pt"
        fallback_auc = -1.0
        fallback_state = None

        progress.console.print(f"  penalty_factor={penalty_factor}  lr={lr}")
        train_losses, val_losses = [], []

        epoch_task = progress.add_task(f"[blue]Fold {fold} epochs", total=epochs)

        # === Training ===
        for epoch in range(1, epochs + 1):
            model.train()
            running_loss = 0.0

            train_task = progress.add_task(
                f"[green]  Epoch {epoch:2d}/{epochs} train", total=len(train_loader)
            )
            for feats, label, _ in train_loader:
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
                progress.advance(train_task)
            progress.remove_task(train_task)

            train_loss = running_loss / len(train_loader)
            train_losses.append(train_loss)

            # === Validation ===
            model.eval()
            val_loss = 0.0
            y_true, y_pred, y_prob = [], [], []

            val_task = progress.add_task(
                f"[yellow]  Epoch {epoch:2d}/{epochs} val  ", total=len(val_loader)
            )
            with torch.no_grad():
                for feats, label, meta in val_loader:
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

                    _save_attention(output_dir, slide_id, out, meta)

                    if save_embeddings and "slide_embedding" in out:
                        emb = out["slide_embedding"].cpu().numpy()
                        np.save(f"{output_dir}/embeddings/{slide_id}_embedding.npy", emb)

                    y_true.append(true)
                    y_pred.append(pred)
                    y_prob.append(probs[1])
                    progress.advance(val_task)
            progress.remove_task(val_task)

            val_loss /= len(val_loader)
            val_losses.append(val_loss)

            # === Metrics ===
            f1 = f1_score(y_true, y_pred, zero_division=0)
            sens = recall_score(y_true, y_pred, zero_division=0)
            spec = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
            gmean = float(np.sqrt(sens * spec))
            auc_val = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan

            saved = (not np.isnan(auc_val)) and (auc_val > best_val_auc) and (gmean >= gmean_threshold)
            epoch_line = (
                f"  Epoch {epoch:2d}/{epochs} | "
                f"Train {train_loss:.4f} "
                f"Val {val_loss:.4f} | "
                f"Gmean {gmean:.3f} "
                f"Sens {sens:.3f} "
                f"AUC {auc_val:.3f}"
                + (" ✓ best" if saved else "")
            )
            progress.console.print(
                f"  Epoch {epoch:2d}/{epochs} | "
                f"Train [red]{train_loss:.4f}[/red] "
                f"Val [yellow]{val_loss:.4f}[/yellow] | "
                f"Gmean [green]{gmean:.3f}[/green] "
                f"Sens [cyan]{sens:.3f}[/cyan] "
                f"AUC [magenta]{auc_val:.3f}[/magenta]"
                + (" [bold green]✓ best[/bold green]" if saved else "")
            )
            if logger:
                logger.info(epoch_line)

            if saved:
                best_val_auc = auc_val
                torch.save(model.state_dict(), best_model_path)

            # Track best-AUC fallback regardless of Gmean gate
            if (not np.isnan(auc_val)) and (auc_val > fallback_auc):
                fallback_auc = auc_val
                fallback_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            progress.advance(epoch_task)

        progress.remove_task(epoch_task)
        all_train_losses.append(train_losses)
        all_val_losses.append(val_losses)

        # === Final evaluation ===
        model.load_state_dict(torch.load(best_model_path, weights_only=True))
        model.eval()
        y_true, y_pred, y_prob = [], [], []

        final_task = progress.add_task(
            f"[bold]  Final eval fold {fold}", total=len(val_loader)
        )
        with torch.no_grad():
            for feats, label, meta in val_loader:
                feats = feats.to(device)
                logits, out = model(feats)
                probs = F.softmax(logits, dim=1).cpu().numpy()[0]
                pred = int(np.argmax(probs))
                true = int(label.cpu().numpy()[0])
                slide_id = meta["slide_id"][0]

                if save_embeddings and "slide_embedding" in out:
                    torch.save(out["slide_embedding"].cpu(), f"{output_dir}/embeddings/{slide_id}_embedding.pt")

                _save_attention(output_dir, slide_id, out, meta)

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
                progress.advance(final_task)
        progress.remove_task(final_task)

        # === Fallback: save best-AUC model if Gmean gate blocked all epochs ===
        if not Path(best_model_path).exists() and fallback_state is not None:
            torch.save(fallback_state, best_model_path)
            msg = (f"Fold {fold}: no epoch passed Gmean >= {gmean_threshold} — "
                   f"saving fallback model (best AUC={fallback_auc:.3f})")
            progress.console.print(f"  [yellow]⚠ {msg}[/yellow]")
            if logger:
                logger.warning(msg)

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

        fold_metrics.append({"fold": fold, "precision": prec, "sensitivity": rec, "f1": f1, "auc": auc_val})
        if logger:
            logger.info(f"Fold {fold} final | precision={prec:.4f} sensitivity={rec:.4f} f1={f1:.4f} auc={auc_val:.4f}")

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

        progress.advance(fold_task)

    # === Save outputs ===
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(f"{output_dir}/results/per_slide_predictions.csv", index=False)

    metrics_df = pd.DataFrame(fold_metrics)
    summary = metrics_df.mean(numeric_only=True).to_dict()
    summary_std = metrics_df.std(numeric_only=True).to_dict()
    summary_median = metrics_df.median(numeric_only=True).to_dict()
    summary_df = pd.DataFrame([summary], index=["mean"])
    summary_df.loc["std"] = summary_std
    summary_df.loc["median"] = summary_median

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

    table = Table(title="Cross-validation summary", show_header=True, header_style="bold magenta")
    for col in ["metric", "mean", "std", "median"]:
        table.add_column(col)
    for metric in summary_df.columns:
        table.add_row(
            metric,
            f"{summary_df.loc['mean', metric]:.4f}",
            f"{summary_df.loc['std', metric]:.4f}",
            f"{summary_df.loc['median', metric]:.4f}",
        )
    console.print(table)

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

def _rank_color(rank, n_total, cmap=None):
    """Return an inferno RGB tuple for a given rank (1-based). Rank 1 → brightest."""
    if cmap is None:
        cmap = plt.get_cmap("inferno")
    val = 1.0 - (rank - 1) / max(n_total - 1, 1)  # 1.0 for rank 1, 0.0 for last
    return tuple(int(c * 255) for c in cmap(val)[:3])


def _draw_label(draw, text, pos, font=None, text_color=(255, 255, 255), shadow_color=(0, 0, 0)):
    """Draw text with a 1-pixel shadow for visibility on any background."""
    x, y = pos
    draw.text((x + 1, y + 1), text, fill=shadow_color, font=font)
    draw.text((x, y), text, fill=text_color, font=font)


def _extract_top_regions_grid(
    wsi,
    coords_lvl0,
    scores,
    k=6,
    patch_level=1,
    patch_size=224,
    region_size=1024,
    grid=(3, 2),
    pad=8,
    cmap_name="plasma",
):
    """
    Extract top-k 1024×1024 context regions centred on the highest-attention patches.
    Each region is annotated with rank (#1, #2, …) and attention score,
    and bordered with an inferno-scale color (bright yellow = highest, dark purple = lowest).
    """
    scores = scores.reshape(-1)
    top_idx = np.argsort(scores)[::-1][:k]
    top_scores = scores[top_idx]

    ds_patch = float(wsi.level_downsamples[patch_level])
    rows, cols = grid
    canvas_w = cols * (region_size + pad) + pad
    canvas_h = rows * (region_size + pad) + pad
    canvas = Image.new("RGB", (canvas_w, canvas_h), (240, 240, 240))

    cmap = plt.get_cmap(cmap_name)
    border_w = 10

    for j, (idx, score) in enumerate(zip(top_idx, top_scores)):
        rank = j + 1
        r, c = divmod(j, cols)
        x_pad = pad + c * (region_size + pad)
        y_pad = pad + r * (region_size + pad)

        x_lvl0, y_lvl0 = coords_lvl0[idx]
        patch_size_lvl0 = patch_size * ds_patch
        cx_lvl0 = x_lvl0 + patch_size_lvl0 / 2
        cy_lvl0 = y_lvl0 + patch_size_lvl0 / 2
        region_size_lvl0 = region_size * ds_patch
        rx_lvl0 = int(cx_lvl0 - region_size_lvl0 / 2)
        ry_lvl0 = int(cy_lvl0 - region_size_lvl0 / 2)

        region = wsi.read_region(
            (rx_lvl0, ry_lvl0), patch_level, (region_size, region_size)
        ).convert("RGB")

        draw = ImageDraw.Draw(region)
        color = _rank_color(rank, k, cmap)

        # Colored border
        for bi in range(border_w):
            draw.rectangle([bi, bi, region_size - 1 - bi, region_size - 1 - bi], outline=color)

        # Central patch rectangle in same color
        patch_offset = (region_size - patch_size) // 2
        draw.rectangle(
            [patch_offset, patch_offset, patch_offset + patch_size, patch_offset + patch_size],
            outline=color, width=3,
        )

        # Rank + score label
        _draw_label(draw, f"#{rank}  {score:.4f}", (border_w + 4, border_w + 4))

        canvas.paste(region, (x_pad, y_pad))

    return canvas


def _extract_top_patches_grid(
    wsi,
    coords_lvl0,
    scores,
    k=20,
    patch_level=1,
    patch_size=224,
    grid=(4, 5),
    pad=6,
    cmap_name="plasma",
):
    """
    Extract top-k patches and arrange them in a grid ordered by rank.
    Each patch is annotated with rank and attention score, and bordered
    with an inferno-scale color (bright yellow = highest, dark purple = lowest).
    """
    scores = scores.reshape(-1)
    top_idx = np.argsort(scores)[::-1][:k]
    top_scores = scores[top_idx]

    rows, cols = grid
    canvas_w = cols * (patch_size + pad) + pad
    canvas_h = rows * (patch_size + pad) + pad
    canvas = Image.new("RGB", (canvas_w, canvas_h), (240, 240, 240))

    cmap = plt.get_cmap(cmap_name)
    border_w = 4

    for j, (idx, score) in enumerate(zip(top_idx, top_scores)):
        rank = j + 1
        r, c = divmod(j, cols)
        x_pad = pad + c * (patch_size + pad)
        y_pad = pad + r * (patch_size + pad)

        x_lvl0, y_lvl0 = coords_lvl0[idx]
        patch = wsi.read_region(
            (int(x_lvl0), int(y_lvl0)), patch_level, (patch_size, patch_size)
        ).convert("RGB")

        draw = ImageDraw.Draw(patch)
        color = _rank_color(rank, k, cmap)

        # Colored border
        for bi in range(border_w):
            draw.rectangle([bi, bi, patch_size - 1 - bi, patch_size - 1 - bi], outline=color)

        # Rank in top-left, score in bottom-left
        _draw_label(draw, f"#{rank}", (border_w + 2, border_w + 2))
        _draw_label(draw, f"{score:.4f}", (border_w + 2, patch_size - 14))

        canvas.paste(patch, (x_pad, y_pad))

    return canvas, top_idx


def wsi_attention_heatmap(
    slide_path: str,
    coords_lvl0: np.ndarray,          # (N,2) top-left in level-0
    scores_raw: np.ndarray,           # (N,) raw logits (consigliato)
    vis_level: int = -1,
    patch_level: int = 1,             # <-- per te: 1
    patch_size: int = 224,            # patch size al patch_level
    alpha: float = 0.6,               # max opacity for high-attention regions
    cmap_name: str = "plasma",
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

    # Attention-proportional alpha: α(pixel) = overlay * alpha_max
    # Low-attention pixels are nearly transparent → tissue shows through cleanly.
    # High-attention pixels reach full alpha_max opacity.
    cmap = plt.get_cmap(cmap_name)
    color = (cmap(overlay)[:, :, :3] * 255).astype(np.uint8)
    alpha_map = (overlay * alpha)[..., np.newaxis]          # [H, W, 1], range 0..alpha
    out = (base.astype(np.float32) * (1 - alpha_map) + color.astype(np.float32) * alpha_map).astype(np.uint8)
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


_LABEL_NAMES = {0: "Not Anaplasia", 1: "Anaplasia"}


def _available_cpus():
    """Return the number of CPUs available to this process.

    Respects SLURM allocations (SLURM_CPUS_PER_TASK) before falling back to
    os.cpu_count(), so it works correctly inside SLURM Docker jobs.
    """
    slurm = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm:
        return int(slurm)
    return os.cpu_count() or 1


def _render_one_slide(job):
    """
    Render the attention report for a single slide.
    Runs in a subprocess — must not reference Rich console or unpicklable state.
    Returns (slide_id, error_str_or_None).
    """
    import numpy as np
    import openslide
    from PIL import Image
    import os

    fname            = job["fname"]
    att_dir          = job["att_dir"]
    out_dir          = job["out_dir"]
    wsi_dir          = job["wsi_dir"]
    pred_map         = job["pred_map"]
    vis_level        = job["vis_level"]
    patch_level      = job["patch_level"]
    patch_size       = job["patch_size"]
    alpha            = job["alpha"]
    cmap_name        = job["cmap_name"]
    convert_to_pct   = job["convert_to_percentiles"]
    max_size         = job["max_size"]
    use_raw          = job["use_raw"]
    extract_region   = job["extract_region"]
    draw_topk        = job["draw_topk"]
    combine_subplots = job["combine_subplots"]
    subplot_layout   = job["subplot_layout"]

    data = np.load(os.path.join(att_dir, fname))
    coords = data["coords"].astype(np.float32)
    scores_raw = (
        data["attention_raw"].astype(np.float32).reshape(-1)
        if use_raw and "attention_raw" in data
        else data["attention"].astype(np.float32).reshape(-1)
    )

    slide_id = fname.replace("_att_with_coords.npz", "")

    try:
        slide_path = _find_wsi_path(wsi_dir, slide_id)
    except FileNotFoundError as e:
        return slide_id, str(e)

    try:
        heatmap_img, _ = wsi_attention_heatmap(
            slide_path=slide_path,
            coords_lvl0=coords,
            scores_raw=scores_raw,
            vis_level=vis_level,
            patch_level=patch_level,
            patch_size=patch_size,
            alpha=alpha,
            cmap_name=cmap_name,
            convert_to_percentiles=convert_to_pct,
            max_size=max_size,
            draw_topk=draw_topk,
        )

        wsi = openslide.open_slide(slide_path)
        if extract_region:
            grid_img = _extract_top_regions_grid(
                wsi=wsi, coords_lvl0=coords, scores=scores_raw,
                k=6, patch_level=patch_level, patch_size=patch_size,
                region_size=1024, grid=(3, 2), pad=8, cmap_name=cmap_name,
            )
        else:
            grid_img, _ = _extract_top_patches_grid(
                wsi=wsi, coords_lvl0=coords, scores=scores_raw,
                k=draw_topk, patch_level=patch_level, patch_size=patch_size,
                grid=(4, 5), pad=6, cmap_name=cmap_name,
            )
        wsi.close()

        title = _build_title(slide_id, pred_map)

        if combine_subplots:
            combined = _combine_subplot(
                heatmap_img, grid_img, layout=subplot_layout, pad=30, title=title,
            )
            if max(combined.size) > 9000:
                f = 9000 / max(combined.size)
                combined = combined.resize(
                    (int(combined.width * f), int(combined.height * f)), Image.BILINEAR
                )
            combined.save(
                os.path.join(out_dir, f"{slide_id}_combined.jpg"),
                quality=92, optimize=True, progressive=True,
            )
        else:
            heatmap_img.save(
                os.path.join(out_dir, f"{slide_id}_heatmap_top{draw_topk}.jpg"),
                quality=92, optimize=True, progressive=True,
            )
            grid_img.save(
                os.path.join(out_dir, f"{slide_id}_top{draw_topk}_patches.jpg"),
                quality=92, optimize=True, progressive=True,
            )

        return slide_id, None

    except Exception as e:
        return slide_id, str(e)


def _load_predictions(base_exp_dir, results_csv=None):
    """Load per-slide predictions from CSV. Returns dict: slide_id -> {true_label, pred_label, prob}."""
    csv_path = results_csv or os.path.join(base_exp_dir, "results", "per_slide_predictions.csv")
    if not os.path.exists(csv_path):
        return {}
    df = pd.read_csv(csv_path)
    return {
        str(row["slide_id"]): {
            "true_label": int(row["true_label"]),
            "pred_label": int(row["pred_label"]),
            "prob":       float(row["prob_anaplasia"]),
        }
        for _, row in df.iterrows()
    }


def _build_title(slide_id, pred_map):
    """Build title string with GT and prediction if available."""
    info = pred_map.get(slide_id)
    if not info:
        return slide_id
    gt_str   = _LABEL_NAMES.get(info["true_label"], "?")
    pred_str = _LABEL_NAMES.get(info["pred_label"], "?")
    return f"{slide_id} | GT: {gt_str} | Pred: {pred_str} (p={info['prob']:.2f})"


def generate_all_attention_reports(
    base_exp_dir,
    wsi_dir,
    patch_size=224,
    patch_level=1,
    vis_level=-1,
    alpha=0.6,
    cmap_name="plasma",
    convert_to_percentiles=True,
    max_size=4096,
    use_raw=True,
    extract_region=False,
    draw_topk=20,
    combine_subplots=True,
    subplot_layout="horizontal",
    results_csv=None,
    num_workers=None,
):
    from concurrent.futures import ProcessPoolExecutor, as_completed

    att_dir = os.path.join(base_exp_dir, "attentions")
    out_dir = os.path.join(base_exp_dir, "visual_reports")
    os.makedirs(out_dir, exist_ok=True)

    pred_map = _load_predictions(base_exp_dir, results_csv)
    files    = [f for f in os.listdir(att_dir) if f.endswith(".npz")]
    n        = len(files)
    workers  = min(num_workers or _available_cpus(), n)

    console.print(f"Found [bold]{n}[/bold] attention files — using [bold]{workers}[/bold] workers")

    shared = dict(
        att_dir=att_dir, out_dir=out_dir, wsi_dir=wsi_dir, pred_map=pred_map,
        vis_level=vis_level, patch_level=patch_level, patch_size=patch_size,
        alpha=alpha, cmap_name=cmap_name, convert_to_percentiles=convert_to_percentiles,
        max_size=max_size, use_raw=use_raw, extract_region=extract_region,
        draw_topk=draw_topk, combine_subplots=combine_subplots,
        subplot_layout=subplot_layout,
    )
    jobs = [{**shared, "fname": fname} for fname in files]

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=28),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Generating attention reports", total=n)

        if workers <= 1:
            for job in jobs:
                slide_id, err = _render_one_slide(job)
                if err:
                    progress.console.print(f"[red]  Failed {slide_id}: {err}")
                progress.advance(task)
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(_render_one_slide, job): job["fname"] for job in jobs}
                for future in as_completed(futures):
                    slide_id, err = future.result()
                    if err:
                        progress.console.print(f"[red]  Failed {slide_id}: {err}")
                    progress.advance(task)

    console.print("[bold green]Attention reports generated.")