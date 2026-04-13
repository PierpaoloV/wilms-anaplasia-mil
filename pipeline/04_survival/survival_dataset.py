"""Survival MIL dataset.

Reads a CSV with at least: slide_id, event, duration. Loads pre-extracted
patch features (.pt) and coordinates (.npy) from disk and returns them
together with the survival label.

The dataset is feature-dimension agnostic — it loads whatever is in the .pt
file (1280 / 2048 / 10240 are all valid) and lets the model handle the input
width.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class SurvivalSlideDataset(Dataset):
    """
    Expects:
      - features_dir/<slide_id><suffix>.pt   -> Tensor [N_patches, feat_dim]
      - coord_dir/<slide_id>.npy             -> structured (x,y) or plain (N,2) array
      - labels CSV with columns: slide_id, event, duration

    Args:
        labels_csv:        path to CSV
        features_dir:      directory containing per-slide .pt features
        coord_dir:         directory containing per-slide .npy coordinates
        slide_ids:         optional whitelist
        feature_suffix:    string appended to slide_id when locating the .pt
                           (e.g. "_total" for multi-tissue features); default ""
        duration_unit:     unit of the `duration` column — "days" (default)
                           or "years". Internally always converted to years.
        clamp_years:       optional (low, high) tuple to clamp duration_years
                           into a fixed range (matches the tutorial's
                           [0.09, 4.37] convention)
    """

    def __init__(
        self,
        labels_csv,
        features_dir,
        coord_dir,
        slide_ids=None,
        feature_suffix: str = "",
        duration_unit: str = "days",
        clamp_years=None,
    ):
        self.df = pd.read_csv(labels_csv)
        if slide_ids is not None:
            self.df = self.df[self.df["slide_id"].isin(slide_ids)].reset_index(drop=True)

        for col in ("slide_id", "event", "duration"):
            if col not in self.df.columns:
                raise ValueError(f"Survival CSV must contain '{col}' column. Found: {list(self.df.columns)}")

        self.features_dir = Path(features_dir)
        self.coord_dir = Path(coord_dir)
        self.feature_suffix = feature_suffix
        self.duration_unit = duration_unit.lower()
        if self.duration_unit not in ("days", "years"):
            raise ValueError(f"duration_unit must be 'days' or 'years', got '{duration_unit}'")
        self.clamp_years = tuple(clamp_years) if clamp_years is not None else None

    def __len__(self):
        return len(self.df)

    def _load_features(self, slide_id):
        path = self.features_dir / f"{slide_id}{self.feature_suffix}.pt"
        feats = torch.load(path, weights_only=True)
        if feats.dtype != torch.float32:
            feats = feats.float()
        return feats

    def _load_coords(self, slide_id):
        npy_path = self.coord_dir / f"{slide_id}.npy"
        if not npy_path.exists():
            return None
        arr = np.load(npy_path, allow_pickle=False)
        if arr.dtype.names and "x" in arr.dtype.names and "y" in arr.dtype.names:
            return np.column_stack((arr["x"], arr["y"])).astype(np.float32)
        if arr.ndim == 2 and arr.shape[1] >= 2:
            return arr[:, :2].astype(np.float32)
        return None

    def _to_years(self, duration_raw: float) -> float:
        years = float(duration_raw) / 365.0 if self.duration_unit == "days" else float(duration_raw)
        if self.clamp_years is not None:
            lo, hi = self.clamp_years
            years = max(lo, min(hi, years))
        return years

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        slide_id = str(row["slide_id"])

        feats = self._load_features(slide_id)
        coords = self._load_coords(slide_id)

        event = int(row["event"])
        duration_raw = float(row["duration"])
        duration_years = self._to_years(duration_raw)

        meta = {
            "slide_id": slide_id,
            "coords": coords,
            "duration_raw": duration_raw,
            "duration_unit": self.duration_unit,
        }
        return (
            feats,
            torch.tensor(event, dtype=torch.long),
            torch.tensor(duration_years, dtype=torch.float32),
            meta,
        )
