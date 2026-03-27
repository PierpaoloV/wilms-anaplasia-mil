#!/usr/bin/env python3
"""
Download model weights from HuggingFace Hub on demand.

Usage:
    python3 download_models.py tb
    python3 download_models.py epithelium
    python3 download_models.py multi-tissue
    python3 download_models.py all

Configuration via environment variables:
  HF_REPO_ID   — HuggingFace repo   (default: hardcoded below)
  HF_TOKEN     — token for private repos (optional)
  MODELS_DIR   — local destination   (default: /home/user/source/models)
"""

import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------
HF_REPO_ID = os.environ.get("HF_REPO_ID", "PierpaoloV93/pathology-segmentation-models")
HF_TOKEN   = os.environ.get("HF_TOKEN", None)
MODELS_DIR = Path(os.environ.get("MODELS_DIR", "/home/user/source/models"))

# Each family maps to:
#   "allow_patterns" — files to download from the HF repo
#   "sentinel"       — a single file that confirms the family is present
FAMILIES = {
    "tb": {
        "allow_patterns": ["tb/**"],
        "sentinel": "tb/playground_soft-cloud-137_best_model.pt",
    },
    "epithelium": {
        "allow_patterns": ["epithelium/**"],
        "sentinel": "epithelium/best_models/Tumour_vs_Healthy_Epitheilum_vivid-dew-9_best_model.pt",
    },
    "multi-tissue": {
        "allow_patterns": ["multi-tissue/**"],
        "sentinel": "multi-tissue/best_models/Multi_Tissue_augmentations_treasured-planet-1_best_model.pt",
    },
}

VALID_FLAGS = list(FAMILIES.keys()) + ["all"]


def download_family(name: str) -> None:
    cfg = FAMILIES[name]
    sentinel = MODELS_DIR / cfg["sentinel"]
    if sentinel.exists():
        print(f"[{name}] Already present — skipping.")
        return
    print(f"[{name}] Downloading from {HF_REPO_ID} ...")
    snapshot_download(
        repo_id=HF_REPO_ID,
        repo_type="model",
        local_dir=str(MODELS_DIR),
        token=HF_TOKEN,
        allow_patterns=cfg["allow_patterns"],
        ignore_patterns=["*.git*", "*.gitattributes"],
    )
    print(f"[{name}] Done → {MODELS_DIR / name}")


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in VALID_FLAGS:
        print(f"Usage: python3 download_models.py [{' | '.join(VALID_FLAGS)}]")
        sys.exit(1)

    flag = sys.argv[1]
    targets = list(FAMILIES.keys()) if flag == "all" else [flag]

    for name in targets:
        download_family(name)
