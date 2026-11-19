"""Figures for the paper/report (300 dpi, serif font)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
                     "font.size": 9, "axes.linewidth": 0.6, "savefig.dpi": 300, "figure.dpi": 100})


def _save(fig, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_persistence_diagrams(D_benign, D_malignant, path):
    """Real persistence diagrams (H0 and H1) of one benign and one malignant image."""
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.4), sharex=True, sharey=True)
    for ax, D, title in zip(axes, (D_benign, D_malignant), ("Benign", "Malignant")):
        assert (D[:, 1] >= D[:, 0]).all(), "death must not be smaller than birth"
        for q, col, lab in ((0, "tab:blue", "$H_0$"), (1, "tab:orange", "$H_1$")):
            m = (D[:, 2] == q) & (D[:, 1] > D[:, 0])
            ax.scatter(D[m, 0], D[m, 1], s=6, alpha=0.5, c=col, label=lab)
        ax.plot([0, 100], [0, 100], "k--", lw=0.8)
        ax.set_xlim(-2, 102)
        ax.set_ylim(-2, 102)
        ax.set_title(title)
        ax.set_xlabel("Birth")
    axes[0].set_ylabel("Death")
    axes[0].legend(loc="lower right", frameon=False)
    _save(fig, path)


def plot_pi_pair(img_benign, img_malignant, path, dim=1):
    """img_*: (2, bins, bins) persistence images for one benign and one malignant sample."""
    fig, axes = plt.subplots(1, 2, figsize=(5.2, 2.6))
    for ax, im, t in zip(axes, (img_benign, img_malignant), ("Benign", "Malignant")):
        h = ax.imshow(im[dim], origin="lower", cmap="viridis")
        ax.set_title(f"{t} ($H_{dim}$)")
        ax.set_xlabel("birth bin")
        ax.set_ylabel("persistence bin")
        fig.colorbar(h, ax=ax, fraction=0.046)
    _save(fig, path)


def plot_median_pi(med_benign, med_malignant, path, dim=1):
    fig, axes = plt.subplots(1, 2, figsize=(5.2, 2.6))
    for ax, im, t in zip(axes, (med_benign, med_malignant), ("Median benign", "Median malignant")):
        h = ax.imshow(im[dim], origin="lower", cmap="viridis")
        ax.set_title(f"{t} ($H_{dim}$)")
        ax.set_xlabel("birth bin")
        ax.set_ylabel("persistence bin")
        fig.colorbar(h, ax=ax, fraction=0.046)
    _save(fig, path)


def plot_roc(curves, path):
    """curves: list of (label, fpr, tpr, auc)."""
    fig, ax = plt.subplots(figsize=(3.4, 3.2))
    for lab, fpr, tpr, auc in curves:
        ax.plot(fpr, tpr, lw=1.4, label=f"{lab} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(loc="lower right", frameon=False, fontsize=7)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    _save(fig, path)


def plot_confusion(cm, path, labels=("Benign", "Malignant")):
    fig, ax = plt.subplots(figsize=(2.8, 2.6))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="black")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    _save(fig, path)


def plot_bars(names, means, stds, path, ylabel="Accuracy (%)", colors=None, ylim=(0, 110)):
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    b = ax.bar(names, means, yerr=stds, capsize=2, color=colors or "#1f5fa8", width=0.6, error_kw={"lw": 0.7})
    for r, m, s in zip(b, means, stds):
        ax.text(r.get_x() + r.get_width() / 2, m + s + 1, f"{m:.1f}", ha="center", va="bottom", fontsize=7.5)
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel)
    ax.spines[["top", "right"]].set_visible(False)
    plt.setp(ax.get_xticklabels(), fontsize=7)
    _save(fig, path)
