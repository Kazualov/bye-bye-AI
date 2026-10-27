"""
Evaluation pipeline — Evaluation Mode CLI.

Usage
-----
    # Run all models
    python -m evaluation.benchmark --dataset data/corne-val --output results/

    # Run specific model(s)
    python -m evaluation.benchmark --dataset data/corne-val --output results/ LaMa
    python -m evaluation.benchmark --dataset data/corne-val --output results/ LaMa "OpenCV Telea"

The script loads every sample in the CORNE-Val dataset, runs each requested
model under every mask variant, computes PSNR / SSIM / LPIPS / inference-time
on the masked region, and writes per-model output directories:

    results/
    └── <ModelName>/
        ├── metrics.csv              per-sample, per-mask rows
        ├── summary.csv              mean ± std aggregated by mask
        ├── qualitative_comparison/  side-by-side PNG per sample
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


# ── model registry ─────────────────────────────────────────────────────────────

# Canonical model names — used for CLI validation and build_models().
AVAILABLE_MODELS = ("OpenCV Telea", "LaMa", "ViTInpainter", "FLUX.1-Fill")


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


# ── model loading ──────────────────────────────────────────────────────────────

def _want(name: str, model_filter: list[str] | None) -> bool:
    """Return True if `name` is requested by the filter (case-insensitive)."""
    if model_filter is None:
        return True
    return any(n.lower() == name.lower() for n in model_filter)


def build_models(model_filter: list[str] | None = None) -> dict:
    """
    Instantiate requested models.

    Parameters
    ----------
    model_filter : list[str] | None
        If provided, only models whose names appear in this list are loaded
        (matched case-insensitively). If None, all available models are loaded.

    Models that are not yet implemented (ViTInpainter, FluxFill) are silently
    skipped so the benchmark runs with whatever is ready.
    """
    models = {}

    if _want("OpenCV Telea", model_filter):
        models["OpenCV Telea"] = OpenCVTelea()

    if _want("LaMa", model_filter):
        try:
            models["LaMa"] = LaMaInpainter()
        except Exception as e:
            print(f"  [skip] LaMa could not be loaded: {e}")

    # Alexander's model — add when implemented
    if _want("ViTInpainter", model_filter):
        try:
            from models.vitinpainter import ViTInpainter
            models["ViTInpainter"] = ViTInpainter()
        except NotImplementedError:
            pass  # not yet implemented
        except Exception as e:
            print(f"  [skip] ViTInpainter: {e}")

    # Egor's model — add when implemented
    if _want("FLUX.1-Fill", model_filter):
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
BEST_METRIC = ("lpips_full", False)
MASK_EXPERIMENTS = [
    ("sam_native",   "mask_sam", MaskType.NATIVE,   0),
    ("sam_dilated10","mask_sam", MaskType.DILATED,  10),
    ("sam_dilated20","mask_sam", MaskType.DILATED,  20),
    ("sam_bbox",     "mask_sam", MaskType.BBOX,      0),
    ("sam_ellipse",  "mask_sam", MaskType.ELLIPSE,   0),
    ("eff_native",   "mask_eff", MaskType.NATIVE,    0),
]


# ── benchmark loop ─────────────────────────────────────────────────────────────

def run_benchmark(
    dataset_path: Path,
    output_path: Path,
    model_filter: list[str] | None = None,
) -> None:
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset from {dataset_path} ...")
    samples = load_dataset(dataset_path)
    print(f"  {len(samples)} samples found.")

    print("Loading models ...")
    models = build_models(model_filter)
    print(f"  Models ready: {list(models.keys())}")

    metric_cols = [
        "psnr_region", "ssim_region", "lpips_region",
        "psnr_full",   "ssim_full",   "lpips_full",
        "inference_time",
    ]

    metric_name, higher_is_better = BEST_METRIC

    for model_name, model in models.items():
        model_dir = output_path / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        rows = []        # (sample, mask_experiment)
        best_rows = []   # победитель на каждый sample

        for sample in tqdm(samples, desc=model_name):
            sample_id = sample["id"]
            shot = sample["shot"]
            gt = sample["bg"]
            best = None  # (score, exp_name, mask, pred)

            for exp_name, mask_key, mask_type, dil_radius in MASK_EXPERIMENTS:
                mask = MaskFactory.make(sample[mask_key], mask_type, dil_radius)

                try:
                    result = model.inpaint(shot, mask)
                except Exception:
                    print(f"\n  [error] {model_name} on {sample_id}/{exp_name}")
                    traceback.print_exc()
                    continue

                metrics = compute_all(result.image, gt, mask)

                rows.append({
                    "sample_id":       sample_id,
                    "model":           model_name,
                    "mask_experiment": exp_name,
                    "psnr_region":     metrics.psnr_region,
                    "ssim_region":     metrics.ssim_region,
                    "lpips_region":    metrics.lpips_region,
                    "psnr_full":       metrics.psnr_full,
                    "ssim_full":       metrics.ssim_full,
                    "lpips_full":      metrics.lpips_full,
                    "inference_time":  result.inference_time,
                })

                score = getattr(metrics, metric_name)
                if np.isnan(score):
                    continue
                if best is None or (score > best[0] if higher_is_better else score < best[0]):
                    best = (score, exp_name, mask, result.image)

            # одна лучшая картинка на сэмпл
            if best is not None:
                score, exp_name, mask, pred = best
                save_comparison_grid(
                    sample_id=f"{sample_id}__{exp_name}",
                    shot=shot,
                    mask=mask,                      # реально использованная маска
                    ground_truth=gt,
                    predictions={model_name: pred},
                    output_dir=model_dir / "qualitative_best",
                )
                best_rows.append({
                    "sample_id": sample_id,
                    "best_mask": exp_name,
                    metric_name: score,
                })

        # ── per-row CSV ────────────────────────────────────────────────────────
        df = pd.DataFrame(rows)
        df.to_csv(model_dir / "metrics.csv", index=False)
        print(f"\n[{model_name}] Saved metrics.csv  ({len(df)} rows)")

        # ── summary по маскам ──────────────────────────────────────────────────
        summary_df = (
            df.groupby("mask_experiment")[metric_cols]
            .agg(["mean", "std"])
            .round(4)
        )
        summary_df.to_csv(model_dir / "summary.csv")

        # ── какая маска лучше справилась ───────────────────────────────────────
        best_df = pd.DataFrame(best_rows)
        best_df.to_csv(model_dir / "best_per_sample.csv", index=False)

        wins = best_df["best_mask"].value_counts().rename("wins")
        mean_score = df.groupby("mask_experiment")[metric_name].mean().rename(f"mean_{metric_name}")
        ranking = (
            pd.concat([wins, mean_score], axis=1)
            .fillna({"wins": 0})
            .astype({"wins": int})
            .sort_values(f"mean_{metric_name}", ascending=not higher_is_better)
        )
        ranking.to_csv(model_dir / "mask_ranking.csv")

        print(f"\n[{model_name}] Ranking масок по {metric_name}:")
        print(ranking.to_string())
        print(f"[{model_name}] Лучшая маска: {ranking.index[0]}")

        # ── plots (как раньше, по sam_native) ──────────────────────────────────
        sam_native = df[df["mask_experiment"] == "sam_native"]
        save_metric_plots({model_name: sam_native[metric_cols].mean().to_dict()}, model_dir / "plots")

        print(f"[{model_name}] All results written to {model_dir}")

    print(f"\nDone. Output root:{output_path}")


# ── CLI entry point ────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="CORNE-Val inpainting benchmark"
    )
    parser.add_argument(
        "models",
        nargs="*",
        metavar="MODEL",
        help=(
            "Model(s) to benchmark. If omitted, all available models are run. "
            f"Available: {', '.join(AVAILABLE_MODELS)}."
        ),
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

    # Validate model names before loading anything
    if args.models:
        unknown = [
            m for m in args.models
            if not any(m.lower() == a.lower() for a in AVAILABLE_MODELS)
        ]
        if unknown:
            parser.error(
                f"Unknown model(s): {', '.join(repr(m) for m in unknown)}. "
                f"Available: {', '.join(AVAILABLE_MODELS)}"
            )

    if not args.dataset.exists():
        parser.error(f"Dataset path does not exist: {args.dataset}")

    run_benchmark(args.dataset, args.output, model_filter=args.models or None)


if __name__ == "__main__":
    main()
