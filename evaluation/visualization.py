"""
Visualisation utilities: comparison grids and metric bar-charts.

Outputs land in:
    results/qualitative_comparison/
    results/failure_cases/
    results/plots/
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


# ── helpers ───────────────────────────────────────────────────────────────────

def _to_pil(array: np.ndarray) -> Image.Image:
    return Image.fromarray(array.astype(np.uint8))


# ── comparison grid ────────────────────────────────────────────────────────────

def save_comparison_grid(
    sample_id: str,
    shot: np.ndarray,
    mask: np.ndarray,
    ground_truth: np.ndarray,
    predictions: dict[str, np.ndarray],
    output_dir: Path,
) -> None:
    """
    Save a horizontal strip: shot | mask | gt | model_1 | model_2 | ...

    Args:
        sample_id:    Used as the filename stem.
        shot:         Original image with the object present.
        mask:         Mask that was applied (white = inpainted region).
        ground_truth: Clean background (the target).
        predictions:  Dict mapping model name → inpainted RGB array.
        output_dir:   Directory to write the PNG into.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    columns = ["Shot", "Mask"] + ["GT"] + list(predictions.keys())
    images = [shot, np.stack([mask, mask, mask], axis=-1), ground_truth] + list(predictions.values())

    fig, axes = plt.subplots(1, len(columns), figsize=(4 * len(columns), 4))
    for ax, title, img in zip(axes, columns, images):
        ax.imshow(img)
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(output_dir / f"{sample_id}.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def save_failure_case(
    sample_id: str,
    model_name: str,
    shot: np.ndarray,
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    failure_type: str,
    description: str,
    output_dir: Path,
) -> None:
    """
    Save an annotated three-panel figure for one documented failure case.

    Args:
        failure_type:  Short label, e.g. "residual_content".
        description:   One-sentence explanation shown as the figure title.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, title, img in zip(axes, ["Shot (input)", f"{model_name} (prediction)", "Ground truth"],
                               [shot, prediction, ground_truth]):
        ax.imshow(img)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    fig.suptitle(f"[{failure_type}] {description}", fontsize=10, wrap=True)
    fig.tight_layout()

    fname = f"{sample_id}_{model_name}_{failure_type}.png"
    fig.savefig(output_dir / fname, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ── metric plots ───────────────────────────────────────────────────────────────

def save_metric_plots(summary: dict, output_dir: Path) -> None:
    """
    Save bar charts comparing models across all metrics.

    Args:
        summary:     {model_name: {metric_name: mean_value}}.
                     Produced by benchmark.py after aggregation.
        output_dir:  Directory to write PNG files into.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "psnr_region": ("PSNR (region) ↑", True),
        "ssim_region": ("SSIM (region) ↑", True),
        "lpips_region": ("LPIPS (region) ↓", False),
        "inference_time": ("Inference time (s) ↓", False),
    }

    models = list(summary.keys())
    x = np.arange(len(models))

    for metric_key, (label, higher_is_better) in metrics.items():
        values = [summary[m].get(metric_key, float("nan")) for m in models]

        fig, ax = plt.subplots(figsize=(7, 4))
        bars = ax.bar(x, values, color="steelblue", edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=15, ha="right")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
        fig.tight_layout()
        fig.savefig(output_dir / f"{metric_key}.png", dpi=120)
        plt.close(fig)
