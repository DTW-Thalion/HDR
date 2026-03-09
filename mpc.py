from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


def rmse(y_true: np.ndarray, y_pred: np.ndarray, axis: int | None = None) -> np.ndarray:
    return np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2, axis=axis))


def classification_metrics(y_true: Sequence[int], y_pred: Sequence[int], y_prob: np.ndarray | None = None, positive_label: int = 1) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    accuracy = float(np.mean(y_true == y_pred)) if y_true.size else float("nan")
    tp = float(np.sum((y_true == positive_label) & (y_pred == positive_label)))
    fp = float(np.sum((y_true != positive_label) & (y_pred == positive_label)))
    fn = float(np.sum((y_true == positive_label) & (y_pred != positive_label)))
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    out = {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}
    if y_prob is not None:
        target = (y_true == positive_label).astype(float)
        p = np.clip(np.asarray(y_prob)[:, positive_label], 1e-6, 1 - 1e-6)
        out["brier"] = float(np.mean((p - target) ** 2))
    return out


def reliability_bins(y_true_binary: np.ndarray, y_prob: np.ndarray, bins: int = 10) -> pd.DataFrame:
    y_true_binary = np.asarray(y_true_binary, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi if i < bins - 1 else y_prob <= hi)
        if not np.any(mask):
            rows.append({"bin": i, "p_mean": (lo + hi) / 2.0, "empirical": np.nan, "count": 0})
        else:
            rows.append({"bin": i, "p_mean": float(np.mean(y_prob[mask])), "empirical": float(np.mean(y_true_binary[mask])), "count": int(np.sum(mask))})
    return pd.DataFrame(rows)


def time_in_target(in_target: np.ndarray) -> float:
    in_target = np.asarray(in_target, dtype=float)
    return float(np.mean(in_target)) if in_target.size else float("nan")


def burden_adherence(control_history: np.ndarray, budget: float) -> float:
    used = float(np.sum(np.abs(control_history)))
    return float(used <= budget)


def circadian_adherence(allowed_mask: np.ndarray, control_history: np.ndarray, eps: float = 1e-8) -> float:
    illegal = np.abs(control_history) > eps
    if illegal.size == 0:
        return float("nan")
    violations = np.sum(illegal & (~allowed_mask.astype(bool)))
    total = np.sum(illegal)
    if total == 0:
        return 1.0
    return float(1.0 - violations / total)


def safety_violation_rate(violations: np.ndarray) -> float:
    violations = np.asarray(violations, dtype=bool)
    return float(np.mean(violations)) if violations.size else float("nan")


def recovery_time_from_challenge(dist_series: np.ndarray, challenge_idx: int, tol: float = 1.1) -> float:
    baseline = float(np.median(dist_series[max(0, challenge_idx - 8):challenge_idx + 1]))
    threshold = tol * max(baseline, 1e-6)
    for t in range(challenge_idx + 1, len(dist_series)):
        if dist_series[t] <= threshold:
            return float(t - challenge_idx)
    return float(len(dist_series) - challenge_idx)


def fit_linear_relationship(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    X = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ beta
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2) + 1e-12)
    r2 = 1.0 - ss_res / ss_tot
    return {"intercept": float(beta[0]), "slope": float(beta[1]), "r2": float(r2)}


def bootstrap_ci(values: Sequence[float], rng: np.random.Generator, n_boot: int = 200) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    boots = []
    for _ in range(n_boot):
        sample = rng.choice(arr, size=arr.size, replace=True)
        boots.append(float(np.mean(sample)))
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return float(np.mean(arr)), float(lo), float(hi)
