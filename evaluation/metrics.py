"""
Evaluation metrics: PSNR, SSIM, LPIPS.

All three metrics focus on the **masked region** so that the large
unchanged area of the image does not dominate the score.
The full-image value is also stored as a secondary reference.
"""

from dataclasses import dataclass

import numpy as np
from skimage.metrics import structural_similarity as _ssim


# lpips is imported lazily so the module loads even without a GPU
_lpips_model = None


def _get_lpips():
    """Load the LPIPS VGG model once and cache it."""
    global _lpips_model
    if _lpips_model is None:
        import lpips
        _lpips_model = lpips.LPIPS(net="vgg")
    return _lpips_model


@dataclass
class MetricsResult:
    """All metrics for one (prediction, ground_truth, mask) triple."""

    # Region-focused (primary)
    psnr_region: float
    ssim_region: float
    lpips_region: float

    # Full-image (secondary reference)
    psnr_full: float
    ssim_full: float
    lpips_full: float


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Return (y0, y1, x0, x1) bounding box of non-zero mask pixels."""
    rows = np.any(mask > 0, axis=1)
    cols = np.any(mask > 0, axis=0)
    y0, y1 = np.where(rows)[0][[0, -1]]
    x0, x1 = np.where(cols)[0][[0, -1]]
    return int(y0), int(y1 + 1), int(x0), int(x1 + 1)


def compute_psnr(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray | None = None) -> float:
    """
    Peak Signal-to-Noise Ratio in dB. Higher is better.

    Args:
        pred:  Predicted RGB image, uint8 (H, W, 3).
        gt:    Ground-truth RGB image, uint8 (H, W, 3).
        mask:  Optional binary mask, uint8 (H, W).
               When provided, MSE is computed only on masked pixels.
    """
    p = pred.astype(np.float64)
    g = gt.astype(np.float64)

    if mask is not None:
        m = mask > 0
        p, g = p[m], g[m]

    mse = np.mean((p - g) ** 2)
    if mse == 0:
        return float("inf")
    return float(20.0 * np.log10(255.0) - 10.0 * np.log10(mse))


def compute_ssim(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray | None = None) -> float:
    """
    Structural Similarity Index. Higher is better, range [-1, 1].

    When a mask is given, the metric is computed on the tight
    bounding-box crop of the masked region for spatial context.
    """
    if mask is not None and np.any(mask > 0):
        y0, y1, x0, x1 = _mask_bbox(mask)
        pred = pred[y0:y1, x0:x1]
        gt = gt[y0:y1, x0:x1]

    # skimage needs at least 7×7 for the default window
    h, w = pred.shape[:2]
    win = min(7, h, w)
    if win < 1:
        return float("nan")

    value, _ = _ssim(pred, gt, win_size=win, channel_axis=2, full=True, data_range=255)
    return float(value)


def compute_lpips(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray | None = None) -> float:
    """
    Learned Perceptual Image Patch Similarity. Lower is better.

    When a mask is given, the metric is computed on the bounding-box
    crop so the model receives a spatially meaningful patch.
    """
    import torch

    if mask is not None and np.any(mask > 0):
        y0, y1, x0, x1 = _mask_bbox(mask)
        pred = pred[y0:y1, x0:x1]
        gt = gt[y0:y1, x0:x1]

    def to_tensor(img: np.ndarray) -> torch.Tensor:
        # LPIPS expects (1, 3, H, W) float in [-1, 1]
        t = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0)
        return t / 127.5 - 1.0

    model = _get_lpips()
    with torch.no_grad():
        score = model(to_tensor(pred), to_tensor(gt))
    return float(score)


def compute_all(
    pred: np.ndarray,
    gt: np.ndarray,
    mask: np.ndarray,
) -> MetricsResult:
    """
    Compute PSNR, SSIM, and LPIPS both on the masked region and
    on the full image.

    Args:
        pred:  Inpainted image, uint8 (H, W, 3).
        gt:    Clean ground-truth image, uint8 (H, W, 3).
        mask:  Binary mask, uint8 (H, W); non-zero = inpainted region.
    """
    return MetricsResult(
        psnr_region=compute_psnr(pred, gt, mask),
        ssim_region=compute_ssim(pred, gt, mask),
        lpips_region=compute_lpips(pred, gt, mask),
        psnr_full=compute_psnr(pred, gt),
        ssim_full=compute_ssim(pred, gt),
        lpips_full=compute_lpips(pred, gt),
    )
