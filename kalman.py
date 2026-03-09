from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .utils import ensure_dir


def save_line_plot(path: Path | str, x: np.ndarray, ys: dict[str, np.ndarray], title: str, xlabel: str, ylabel: str) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, y in ys.items():
        ax.plot(x, y, label=label)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if len(ys) > 1:
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def save_scatter_plot(path: Path | str, x: np.ndarray, y: np.ndarray, title: str, xlabel: str, ylabel: str, line: tuple[float, float] | None = None) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(x, y, s=18)
    if line is not None:
        xs = np.linspace(float(np.min(x)), float(np.max(x)), 100)
        ax.plot(xs, line[0] + line[1] * xs)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def save_bar_plot(path: Path | str, labels: list[str], values: list[float], title: str, ylabel: str) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, values)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def save_heatmap(path: Path | str, matrix: np.ndarray, xlabels: list[str], ylabels: list[str], title: str, colorbar_label: str) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(matrix, aspect="auto")
    ax.set_xticks(range(len(xlabels)), labels=xlabels)
    ax.set_yticks(range(len(ylabels)), labels=ylabels)
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    cb = fig.colorbar(im, ax=ax)
    cb.set_label(colorbar_label)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def save_calibration_plot(path: Path | str, df: pd.DataFrame, title: str = "Calibration") -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], linestyle="--")
    valid = df["count"] > 0
    ax.plot(df.loc[valid, "p_mean"], df.loc[valid, "empirical"], marker="o")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Empirical frequency")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path
