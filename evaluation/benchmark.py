"""
Evaluation pipeline — Evaluation Mode CLI.

Usage
-----
    python benchmark.py --dataset data/corne-val --output results/

The script loads every sample in the CORNE-Val dataset, runs all
available models under each requested mask variant, computes
PSNR / SSIM / LPIPS / inference-time on the masked region, and writes:

    results/
    ├── metrics.csv              per-sample, per-model, per-mask rows
    ├── summary.csv              mean ± std aggregated by model + mask
    ├── qualitative_comparison/  side-by-side PNG per sample
    ├── failure_cases/           annotated failure PNGs
    └── plots/                   bar charts per metric

CORNE-Val directory layout expected
-------------------------------------
    corne-val/
    └── <sample_id>/
        ├── shot.jpg   (or .png)
        ├── bg.jpg
        ├── mask_sam.png
        └── mask_eff.png
"""

import argparse
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from evaluation.masks import MaskFactory, MaskType
from evaluation.metrics import compute_all
from evaluation.visualization import save_comparison_grid, save_metric_plots
from models.opencv_telea import OpenCVTelea
from models.lama import LaMaInpainter


# ── dataset loading ────────────────────────────────────────────────────────────

def _find_file(directory: Path, stem: str) -> Path | None:
    """Return the first file matching `stem.*` inside `directory`."""
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        candidate = directory / (stem + ext)
        if candidate.exists():
            return candidate
    return None


def load_sample(sample_dir: Path) -> dict | None:
    """
    Load one CORNE-Val sample.

    Returns a dict with keys: id, shot, bg, mask_sam, mask_eff.
    Returns None if any required file is missing.
    """
    shot_path = _find_file(sample_dir, "shot")
    bg_path = _find_file(sample_dir, "bg")
    mask_sam_path = _find_file(sample_dir, "mask_sam")
    mask_eff_path = _find_file(sample_dir, "mask_eff")

    if not all([shot_path, bg_path, mask_sam_path, mask_eff_path]):
        return None

    return {
        "id": sample_dir.name,
        "shot": np.array(Image.open(shot_path).convert("RGB")),
        "bg": np.array(Image.open(bg_path).convert("RGB")),
        "mask_sam": np.array(Image.open(mask_sam_path).convert("L")),
        "mask_eff": np.array(Image.open(mask_eff_path).convert("L")),
    }


def load_dataset(dataset_path: Path) -> list[dict]:
    """Return all valid samples from `dataset_path`."""
    samples = []
    for sample_dir in sorted(dataset_path.iterdir()):
        if not sample_dir.is_dir():
            continue
        sample = load_sample(sample_dir)
        if sample is None:
            print(f"  [skip] {sample_dir.name}: missing files")
        else:
            samples.append(sample)
    return samples


# ── model registry ─────────────────────────────────────────────────────────────

def build_models() -> dict:
    """
    Instantiate all available models.

    Models that are not yet implemented (ViTInpainter, FluxFill)
    are silently skipped so the benchmark runs with whatever is ready.
    """
    models = {}

    models["OpenCV Telea"] = OpenCVTelea()

    try:
        models["LaMa"] = LaMaInpainter()
    except Exception as e:
        print(f"  [skip] LaMa could not be loaded: {e}")

    # Alexander's model — add when implemented
    try:
        from models.vitinpainter import ViTInpainter
        models["ViTInpainter"] = ViTInpainter()
    except NotImplementedError:
        pass  # not yet implemented
    except Exception as e:
        print(f"  [skip] ViTInpainter: {e}")

    # Egor's model — add when implemented
    try:
        from models.flux_fill import FluxFillInpainter
        models["FLUX.1-Fill"] = FluxFillInpainter()
    except NotImplementedError:
        pass  # not yet implemented
    except Exception as e:
        print(f"  [skip] FLUX.1-Fill: {e}")

    return models


# ── mask experiment plan ───────────────────────────────────────────────────────

# Each entry: (mask_experiment_name, base_mask_key, MaskType, dilation_radius)
MASK_EXPERIMENTS = [
    ("sam_native",   "mask_sam", MaskType.NATIVE,   0),
    ("sam_dilated10","mask_sam", MaskType.DILATED,  10),
    ("sam_dilated20","mask_sam", MaskType.DILATED,  20),
    ("sam_bbox",     "mask_sam", MaskType.BBOX,      0),
    ("sam_ellipse",  "mask_sam", MaskType.ELLIPSE,   0),
    ("eff_native",   "mask_eff", MaskType.NATIVE,    0),
]


# ── benchmark loop ─────────────────────────────────────────────────────────────

def run_benchmark(dataset_path: Path, output_path: Path) -> None:
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset from {dataset_path} ...")
    samples = load_dataset(dataset_path)
    print(f"  {len(samples)} samples found.")

    print("Loading models ...")
    models = build_models()
    print(f"  Models ready: {list(models.keys())}")

    rows = []  # one dict per (sample, model, mask_experiment)

    for sample in tqdm(samples, desc="Samples"):
        sample_id = sample["id"]
        shot = sample["shot"]
        gt = sample["bg"]

        predictions_for_grid: dict[str, np.ndarray] = {}

        for exp_name, mask_key, mask_type, dil_radius in MASK_EXPERIMENTS:
            base_mask = sample[mask_key]
            mask = MaskFactory.make(base_mask, mask_type, dil_radius)

            for model_name, model in models.items():
                try:
                    result = model.inpaint(shot, mask)
                except Exception:
                    print(f"\n  [error] {model_name} on {sample_id}/{exp_name}")
                    traceback.print_exc()
                    continue

                metrics = compute_all(result.image, gt, mask)

                rows.append({
                    "sample_id":      sample_id,
                    "model":          model_name,
                    "mask_experiment": exp_name,
                    "psnr_region":    metrics.psnr_region,
                    "ssim_region":    metrics.ssim_region,
                    "lpips_region":   metrics.lpips_region,
                    "psnr_full":      metrics.psnr_full,
                    "ssim_full":      metrics.ssim_full,
                    "lpips_full":     metrics.lpips_full,
                    "inference_time": result.inference_time,
                })

                # collect first mask experiment (sam_native) for the grid
                if exp_name == "sam_native":
                    predictions_for_grid[model_name] = result.image

        # qualitative comparison grid for this sample
        if predictions_for_grid:
            save_comparison_grid(
                sample_id=sample_id,
                shot=shot,
                mask=MaskFactory.native(sample["mask_sam"]),
                ground_truth=gt,
                predictions=predictions_for_grid,
                output_dir=output_path / "qualitative_comparison",
            )

    # ── save per-row CSV ───────────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    df.to_csv(output_path / "metrics.csv", index=False)
    print(f"\nSaved metrics.csv  ({len(df)} rows)")

    # ── aggregate summary ──────────────────────────────────────────────────────
    metric_cols = ["psnr_region", "ssim_region", "lpips_region", "psnr_full",
                   "ssim_full", "lpips_full", "inference_time"]
    summary_df = (
        df.groupby(["model", "mask_experiment"])[metric_cols]
        .agg(["mean", "std"])
        .round(4)
    )
    summary_df.to_csv(output_path / "summary.csv")
    print(f"Saved summary.csv")

    # ── plots ──────────────────────────────────────────────────────────────────
    # Build a simple {model: {metric: mean}} dict from sam_native only
    sam_native = df[df["mask_experiment"] == "sam_native"]
    summary_dict = {
        model: sam_native[sam_native["model"] == model][metric_cols].mean().to_dict()
        for model in sam_native["model"].unique()
    }
    save_metric_plots(summary_dict, output_path / "plots")
    print(f"Saved plots/")

    print(f"\nAll results written to {output_path}")


# ── CLI entry point ────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="CORNE-Val inpainting benchmark"
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to the CORNE-Val dataset directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results"),
        help="Directory to write all results into (default: results/).",
    )
    args = parser.parse_args()

    if not args.dataset.exists():
        parser.error(f"Dataset path does not exist: {args.dataset}")

    run_benchmark(args.dataset, args.output)


if __name__ == "__main__":
    main()
