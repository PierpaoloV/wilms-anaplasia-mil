"""Survival MIL model.

Thin wrapper around `AttentionSingleBranch` (from pipeline/03_classification)
configured with `n_classes=1` so the final Linear emits a single risk score
suitable for Cox-style survival training.

Feature dimension is taken from `size[0]` and is fully flexible:
1280 (Virchow2), 2048 (UNI / single-tissue), 10240 (multi-tissue concat) all
work without code changes.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

# Make pipeline/03_classification importable when running as a script
_THIS_DIR = Path(__file__).resolve().parent
_CLASSIFICATION_DIR = _THIS_DIR.parent / "03_classification"
if str(_CLASSIFICATION_DIR) not in sys.path:
    sys.path.insert(0, str(_CLASSIFICATION_DIR))

from mil_modules import AttentionSingleBranch, load_mil_checkpoint  # noqa: E402


class SurvivalMIL(nn.Module):
    """Wraps `AttentionSingleBranch(n_classes=1)` for survival prediction.

    Forward signature: x -> (risk, results_dict)
        risk           : Tensor [1] — single scalar risk score per slide
        results_dict   : {"attention": [1, N, 1], "slide_embedding": [1, in_dim]}

    The risk score is the raw output of the final Linear layer (no sigmoid).
    For Cox loss this is the log-hazard ratio; for ranking metrics
    (C-index, time-dependent AUC) it is the score we sort by.
    """

    def __init__(self, size=(1280, 512, 128), use_dropout: bool = False):
        super().__init__()
        self.backbone = AttentionSingleBranch(
            size=tuple(size), use_dropout=use_dropout, n_classes=1,
        )
        self.size = tuple(size)

    def forward(self, x):
        logits, results_dict = self.backbone(x)
        risk = logits.squeeze(-1)  # [1, 1] -> [1]
        return risk, results_dict


def load_survival_checkpoint(path, device: str = "cpu") -> SurvivalMIL:
    """Load a survival MIL checkpoint.

    Reuses the auto-detecting `load_mil_checkpoint` from mil_modules, which
    infers `size` and `n_classes` from the state dict. We then wrap the
    returned `AttentionSingleBranch` (with `n_classes=1`) inside a `SurvivalMIL`.
    """
    backbone = load_mil_checkpoint(path, device=device)
    if getattr(backbone, "n_classes", None) != 1:
        raise ValueError(
            f"Checkpoint at {path} has n_classes={backbone.n_classes}, "
            "but a survival checkpoint must have n_classes=1."
        )

    wrapper = SurvivalMIL(size=backbone.size)
    wrapper.backbone = backbone
    wrapper.to(device)
    wrapper.eval()
    return wrapper
