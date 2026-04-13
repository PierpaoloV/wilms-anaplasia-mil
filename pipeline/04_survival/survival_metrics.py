"""Survival evaluation metrics and Kaplan-Meier plotting helpers.

Wraps torchsurv (C-index, time-dependent AUC) and lifelines (Kaplan-Meier).
All inputs are 1-D arrays/tensors aligned by sample.
"""

from pathlib import Path

import numpy as np
import torch


def _as_1d_tensor(x, dtype=torch.float32) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().reshape(-1).to(dtype)
    return torch.as_tensor(np.asarray(x).reshape(-1), dtype=dtype)


def compute_cindex(risk, event, time) -> float:
    """Concordance index. Higher risk should align with shorter time-to-event.

    Args:
        risk:  predicted risk score per sample (higher = higher risk)
        event: 1 if event observed, 0 if censored
        time:  time-to-event (any consistent unit)
    """
    from torchsurv.metrics.cindex import ConcordanceIndex

    risk_t = _as_1d_tensor(risk)
    event_t = _as_1d_tensor(event, dtype=torch.bool)
    time_t = _as_1d_tensor(time)
    cindex = ConcordanceIndex()
    return float(cindex(risk_t, event_t, time_t))


def compute_time_dependent_auc(risk, event, time, eval_times) -> dict:
    """Time-dependent AUC at a list of evaluation times.

    Returns a dict {eval_time: auc_value}. NaN entries are dropped.
    """
    from torchsurv.metrics.auc import Auc

    risk_t = _as_1d_tensor(risk)
    event_t = _as_1d_tensor(event, dtype=torch.bool)
    time_t = _as_1d_tensor(time)
    eval_t = _as_1d_tensor(eval_times)

    auc = Auc()
    values = auc(risk_t, event_t, time_t, new_time=eval_t)
    out = {}
    for t, v in zip(eval_t.tolist(), values.tolist()):
        if not (isinstance(v, float) and np.isnan(v)):
            out[float(t)] = float(v)
    return out


def plot_km_by_median_risk(risk, event, time, out_path, time_unit_label: str = "days"):
    """Stratify by median predicted risk, plot Kaplan-Meier curves and log-rank p-value."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import logrank_test

    risk_arr = _as_1d_tensor(risk).numpy()
    event_arr = _as_1d_tensor(event).numpy().astype(int)
    time_arr = _as_1d_tensor(time).numpy()

    median_risk = float(np.median(risk_arr))
    high_mask = risk_arr >= median_risk
    low_mask = ~high_mask

    fig, ax = plt.subplots(figsize=(7, 5))
    kmf_high = KaplanMeierFitter()
    kmf_low = KaplanMeierFitter()
    kmf_high.fit(time_arr[high_mask], event_arr[high_mask], label=f"High risk (n={high_mask.sum()})")
    kmf_low.fit(time_arr[low_mask], event_arr[low_mask], label=f"Low risk (n={low_mask.sum()})")
    kmf_high.plot_survival_function(ax=ax, ci_show=True, color="tab:red")
    kmf_low.plot_survival_function(ax=ax, ci_show=True, color="tab:blue")

    p_value = float("nan")
    if high_mask.sum() > 0 and low_mask.sum() > 0:
        lr = logrank_test(
            time_arr[high_mask], time_arr[low_mask],
            event_observed_A=event_arr[high_mask],
            event_observed_B=event_arr[low_mask],
        )
        p_value = float(lr.p_value)

    ax.set_xlabel(f"Time ({time_unit_label})")
    ax.set_ylabel("Survival probability")
    title = "Kaplan-Meier by median predicted risk"
    if not np.isnan(p_value):
        title += f"  (log-rank p={p_value:.3g})"
    ax.set_title(title)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left")
    fig.tight_layout()

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return p_value
