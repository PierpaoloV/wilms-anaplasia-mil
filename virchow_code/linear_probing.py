#!/usr/bin/env python3
"""
linear_probing.py — Linear and MLP classifiers on pre-extracted slide features.

CLI usage:
    python linear_probing.py --config linear_runs.yaml --run linear_all_0penalty
    python linear_probing.py --config linear_runs.yaml --run all
    python linear_probing.py --config linear_runs.yaml --run all --rerun
"""
import os
import argparse
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from abc import ABC, abstractmethod
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class FeatureDataset(Dataset):
    """
    Loads pre-extracted slide-level features (.pt files) with labels from a CSV.

    Expected CSV columns: slide_id, Patient_id, Diagnose, fold.
    Expected feature file: <feature_dir>/<slide_id>.pt

    Supports binary (Not Anaplasia=0, Focal/Diffuse=1) and
    3-class (Not Anaplasia=0, Focal=1, Diffuse=2) labelling.
    """

    BINARY_MAP = {"Not Anaplasia": 0, "Focal": 1, "Diffuse": 1}
    LABEL_MAP  = {"Not Anaplasia": 0, "Focal": 1, "Diffuse": 2}

    def __init__(self, feature_dir, label_csv, fold=None, split="train", binary=True):
        self.slide_ids = []
        self.patient_ids = []
        self.features = []
        self.labels = []

        df = pd.read_csv(label_csv)
        if fold is not None:
            df = df[df["fold"] != fold] if split == "train" else df[df["fold"] == fold]

        label_map = self.BINARY_MAP if binary else self.LABEL_MAP

        for _, row in df.iterrows():
            slide_id = str(row["slide_id"])
            pt_path = os.path.join(feature_dir, f"{slide_id}.pt")
            if not os.path.exists(pt_path):
                continue

            feat = torch.load(pt_path, weights_only=True)
            if isinstance(feat, dict):
                feat = feat.get("features") or feat.get("slide_embedding") or list(feat.values())[0]
            feat = feat.squeeze().cpu().numpy()

            label_str = row["Diagnose"]
            if label_str not in label_map:
                continue

            self.features.append(feat)
            self.labels.append(label_map[label_str])
            self.slide_ids.append(slide_id)
            self.patient_ids.append(str(row["Patient_id"]))

        self.features = torch.tensor(np.array(self.features)).float()
        self.labels = torch.tensor(self.labels).long()

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx], idx


# ---------------------------------------------------------------------------
# Base adaptor
# ---------------------------------------------------------------------------

class CaseLevelTaskAdaptor(ABC):
    def __init__(self, shot_features, shot_labels, test_features, shot_extra_labels=None):
        self.shot_features = shot_features
        self.shot_labels = shot_labels
        self.test_features = test_features
        self.shot_extra_labels = shot_extra_labels

    @abstractmethod
    def fit(self, penalty_factor=0.0):
        pass

    @abstractmethod
    def predict(self):
        pass


# ---------------------------------------------------------------------------
# PyTorch model backbones
# ---------------------------------------------------------------------------

class LinearClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)


class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        self.mlp = nn.Sequential(*layers)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        features = self.mlp(x)
        return self.fc(features), features


# ---------------------------------------------------------------------------
# Adaptors
# ---------------------------------------------------------------------------

def _train_loop(model, shot_features, shot_labels, optimizer, num_epochs, patience, penalty_factor, has_aux_output=False):
    """Shared training loop with confidence penalty and early stopping."""
    device = shot_features.device
    best_loss, best_epoch, best_state = float("inf"), 0, model.state_dict()

    for epoch in tqdm(range(num_epochs), desc="Training", unit="epoch", leave=True):
        model.train()
        optimizer.zero_grad()

        out = model(shot_features)
        logits = out[0] if has_aux_output else out

        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        confidences = probs[torch.arange(len(preds)), preds]

        correct = (preds == shot_labels).float()
        penalty = 1.0 + penalty_factor * (1.0 - correct) * confidences
        ce_loss = nn.CrossEntropyLoss(reduction="none")(logits, shot_labels)
        loss = (penalty * ce_loss).mean()

        loss.backward()
        optimizer.step()

        epoch_loss = loss.item()
        if epoch_loss < best_loss:
            best_loss, best_epoch, best_state = epoch_loss, epoch, model.state_dict()
        elif epoch - best_epoch > patience:
            tqdm.write(f"Early stopping at epoch {epoch + 1}")
            break

    model.load_state_dict(best_state)
    return model


class LinearProbing(CaseLevelTaskAdaptor):
    def __init__(self, shot_features, shot_labels, test_features,
                 num_epochs=100, learning_rate=0.001, patience=10,
                 return_probabilities=False, **kwargs):
        super().__init__(shot_features, shot_labels, test_features)
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.patience = patience
        self.return_probabilities = return_probabilities

    def fit(self, penalty_factor=0.0):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        X = torch.tensor(self.shot_features, dtype=torch.float32).to(device)
        y = torch.tensor(self.shot_labels, dtype=torch.long).to(device)
        self.test_features = torch.tensor(self.test_features, dtype=torch.float32).to(device)

        num_classes = int(self.shot_labels.max()) + 1
        self.model = LinearClassifier(X.shape[1], num_classes).to(device)
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.model = _train_loop(self.model, X, y, optimizer, self.num_epochs, self.patience, penalty_factor)

    def predict(self):
        self.model.eval()
        with torch.no_grad():
            logits = self.model(self.test_features)
            if self.return_probabilities:
                return torch.softmax(logits, dim=1).cpu().numpy()
            return torch.argmax(logits, dim=1).cpu().numpy()


class MultiLayerPerceptron(CaseLevelTaskAdaptor):
    def __init__(self, shot_features, shot_labels, test_features,
                 hidden_dim=256, num_layers=3, num_epochs=100,
                 learning_rate=0.001, patience=10, return_probabilities=False, **kwargs):
        super().__init__(shot_features, shot_labels, test_features)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.patience = patience
        self.return_probabilities = return_probabilities

    def fit(self, penalty_factor=0.0):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        X = torch.tensor(self.shot_features, dtype=torch.float32).to(device)
        y = torch.tensor(self.shot_labels, dtype=torch.long).to(device)
        self.test_features = torch.tensor(self.test_features, dtype=torch.float32).to(device)

        num_classes = int(self.shot_labels.max()) + 1
        self.model = MLPClassifier(X.shape[1], self.hidden_dim, num_classes, self.num_layers).to(device)
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.model = _train_loop(self.model, X, y, optimizer, self.num_epochs, self.patience, penalty_factor, has_aux_output=True)

    def predict(self):
        self.model.eval()
        with torch.no_grad():
            logits, features = self.model(self.test_features)
            if self.return_probabilities:
                return torch.softmax(logits, dim=1).cpu().numpy(), features.cpu().numpy()
            return torch.argmax(logits, dim=1).cpu().numpy()


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def cross_validate_model(feature_dir, label_csv, output_dir, model_name="mlp",
                         penalty_factor=0.0, binary=True, hdim=256,
                         num_epochs=100, lr=0.001, patience=10):
    """
    5-fold cross-validation over pre-extracted features using LinearProbing or MLP.

    Returns results_df, metrics_df, summary_df, roc_info.
    """
    os.makedirs(output_dir, exist_ok=True)

    all_results, fold_metrics, roc_info = [], [], []

    for fold in range(1, 6):
        print(f"\n===== FOLD {fold} =====")
        train_ds = FeatureDataset(feature_dir, label_csv, fold=fold, split="train", binary=binary)
        val_ds   = FeatureDataset(feature_dir, label_csv, fold=fold, split="val",   binary=binary)

        X_train, y_train = train_ds.features.numpy(), train_ds.labels.numpy()
        X_val,   y_val   = val_ds.features.numpy(),   val_ds.labels.numpy()

        model_args = dict(shot_features=X_train, shot_labels=y_train, test_features=X_val,
                          return_probabilities=True, num_epochs=num_epochs,
                          learning_rate=lr, patience=patience)

        if model_name == "mlp":
            clf = MultiLayerPerceptron(**model_args, hidden_dim=hdim)
        elif model_name == "linear":
            clf = LinearProbing(**model_args)
        else:
            raise ValueError(f"Unknown model: {model_name}. Use 'mlp' or 'linear'.")

        clf.fit(penalty_factor=penalty_factor)

        if model_name == "mlp":
            probs, feats = clf.predict()
        else:
            probs = clf.predict()
            feats = X_val

        preds = np.argmax(probs, axis=1)
        confidences = probs[np.arange(len(preds)), preds]

        acc  = accuracy_score(y_val, preds)
        prec = precision_score(y_val, preds, average="macro", zero_division=0)
        rec  = recall_score(y_val, preds, average="macro", zero_division=0)
        f1   = f1_score(y_val, preds, average="macro", zero_division=0)
        auc_val = roc_auc_score(y_val, probs[:, 1]) if len(np.unique(y_val)) > 1 else float("nan")

        print(f"Acc={acc:.3f} | Prec={prec:.3f} | Rec={rec:.3f} | F1={f1:.3f} | AUC={auc_val:.3f}")
        fold_metrics.append({"fold": fold, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "auc": auc_val})

        if binary and len(np.unique(y_val)) > 1:
            fpr, tpr, _ = roc_curve(y_val, probs[:, 1])
            roc_info.append((fpr, tpr, auc(fpr, tpr)))

        for i in range(len(val_ds)):
            all_results.append({
                "slide_id":   val_ds.slide_ids[i],
                "patient_id": val_ds.patient_ids[i],
                "true_label": int(y_val[i]),
                "pred_label": int(preds[i]),
                "confidence": float(confidences[i]),
                "fold":       fold,
                "feature":    feats[i].tolist(),
            })

    results_df = pd.DataFrame(all_results)
    metrics_df = pd.DataFrame(fold_metrics)
    summary_df = pd.DataFrame([metrics_df.mean(numeric_only=True)], index=["mean"])
    summary_df.loc["std"] = metrics_df.std(numeric_only=True)

    results_df.to_csv(os.path.join(output_dir, "results.csv"), index=False)
    metrics_df.to_csv(os.path.join(output_dir, "metrics_per_fold.csv"), index=False)
    summary_df.to_csv(os.path.join(output_dir, "metrics_summary.csv"))

    _save_confusion_matrices(results_df, output_dir)
    if roc_info:
        _save_mean_roc_curve(roc_info, os.path.join(output_dir, "mean_roc_curve.png"))

    print("\nCross-validation complete.")
    print(summary_df)
    return results_df, metrics_df, summary_df, roc_info


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _save_confusion_matrices(results_df, output_dir, class_names=None, dpi=200):
    os.makedirs(output_dir, exist_ok=True)
    if class_names is None:
        class_names = ["Not Anaplasia", "Anaplasia"]

    for fold in sorted(results_df["fold"].unique()):
        fold_df = results_df[results_df["fold"] == fold]
        cm = confusion_matrix(fold_df["true_label"], fold_df["pred_label"])
        fig, ax = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay(cm, display_labels=class_names).plot(ax=ax, cmap="Blues", values_format="d")
        plt.title(f"Confusion Matrix – Fold {fold}")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"confusion_matrix_fold{fold}.png"), dpi=dpi)
        plt.close(fig)


def _save_mean_roc_curve(roc_info, output_path, model_name="", dpi=200):
    mean_fpr = np.linspace(0, 1, 100)
    tprs, aucs = [], []
    for fpr, tpr, roc_auc in roc_info:
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(roc_auc)

    mean_tpr = np.mean(tprs, axis=0)
    std_tpr  = np.std(tprs, axis=0)
    mean_auc = np.mean(aucs)
    std_auc  = np.std(aucs)

    plt.figure(figsize=(7, 6))
    plt.plot(mean_fpr, mean_tpr, color="blue",
             label=f"Mean ROC (AUC = {mean_auc:.2f} ± {std_auc:.2f})", lw=2)
    plt.fill_between(mean_fpr,
                     np.maximum(mean_tpr - std_tpr, 0),
                     np.minimum(mean_tpr + std_tpr, 1),
                     color="blue", alpha=0.2, label="±1 std dev")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve{' — ' + model_name if model_name else ''}")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def _safe_cast(value):
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("true", "false"):
            return v == "true"
        try:
            return float(v) if any(c in v for c in [".", "e"]) else int(v)
        except ValueError:
            return value
    elif isinstance(value, list):
        return [_safe_cast(v) for v in value]
    return value


def load_config(config_path, run_name):
    with open(config_path) as f:
        config_all = yaml.safe_load(f)

    defaults = config_all.get("defaults", {})
    runs = config_all.get("runs", {})
    if run_name not in runs:
        raise ValueError(f"Run '{run_name}' not found in {config_path}. Available: {list(runs)}")

    cfg = {**defaults, **runs[run_name]}
    cfg = {k: _safe_cast(v) for k, v in cfg.items()}

    output_base = cfg.get("output_base_dir", "Experiments")
    cfg["output_dir"] = str(Path(output_base) / cfg["name"])
    return cfg


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run linear/MLP probing experiments")
    parser.add_argument("--config", required=True, help="Path to linear_runs.yaml")
    parser.add_argument("--run", required=True, help="Run name or 'all'")
    parser.add_argument("--rerun", action="store_true", help="Force rerun even if results exist")
    args = parser.parse_args()

    with open(args.config) as f:
        config_all = yaml.safe_load(f)
    all_runs = config_all.get("runs", {})

    run_names = list(all_runs) if args.run.lower() == "all" else [args.run]

    for run_name in run_names:
        cfg = load_config(args.config, run_name)
        summary_path = Path(cfg["output_dir"]) / "metrics_summary.csv"

        if summary_path.exists() and not args.rerun:
            print(f"Skipping '{cfg['name']}' — results exist ({summary_path})")
            continue

        print(f"\n{'='*60}\nRun: {cfg['name']}\nOutput: {cfg['output_dir']}\n{'='*60}")

        cross_validate_model(
            feature_dir   = cfg["feature_dir"],
            label_csv     = cfg["label_csv"],
            output_dir    = cfg["output_dir"],
            model_name    = cfg.get("model", "linear"),
            penalty_factor= float(cfg.get("penalty_factor", 0.0)),
            binary        = bool(cfg.get("binary", True)),
            hdim          = int(cfg.get("hdim", 256)),
            num_epochs    = int(cfg.get("num_epochs", 100)),
            lr            = float(cfg.get("lr", 0.001)),
            patience      = int(cfg.get("patience", 10)),
        )

    print("\nAll runs completed.")


if __name__ == "__main__":
    main()
