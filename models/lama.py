"""
LaMa (Large Mask inpainting) — CNN-based inpainting model.

Loads the official big-lama TorchScript weights directly from
HuggingFace Hub (smartywu/big-lama), so no intermediate wrapper
package with stale Pillow pins is required.

Weights are cached by huggingface_hub on first use (~200 MB).
"""

import time
from pathlib import Path

import numpy as np
import torch
from torch.hub import download_url_to_file

from .base import BaseInpainter, InpaintResult

# TorchScript export of big-lama (hosted by the IOPaint project)
_WEIGHTS_URL = (
    "https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt"
)
_WEIGHTS_PATH = Path.home() / ".cache" / "inpainting-bench" / "big-lama.pt"


def _ensure_weights() -> Path:
    if not _WEIGHTS_PATH.exists():
        _WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = _WEIGHTS_PATH.with_suffix(".part")
        download_url_to_file(_WEIGHTS_URL, str(tmp))
        tmp.rename(_WEIGHTS_PATH)  # only appears once fully downloaded
    return _WEIGHTS_PATH


class LaMaInpainter(BaseInpainter):
    def __init__(self):
        weights_path = _ensure_weights()
        self._device = self._pick_device()
        self._model = torch.jit.load(str(weights_path), map_location=self._device)
        self._model.eval()

    @staticmethod
    def _pick_device() -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    @property
    def name(self) -> str:
        return "LaMa"

    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> InpaintResult:
        # LaMa requires spatial dims to be multiples of 8
        orig_h, orig_w = image.shape[:2]
        image_padded, mask_padded = self._pad(image, mask)

        img_t  = self._img_to_tensor(image_padded)
        mask_t = self._mask_to_tensor(mask_padded)

        start = time.perf_counter()
        with torch.no_grad():
            out_t = self._model(img_t, mask_t)
        elapsed = time.perf_counter() - start

        output = self._tensor_to_img(out_t)
        # Crop padding back to original size
        output = output[:orig_h, :orig_w]
        return InpaintResult(image=output, inference_time=elapsed)

    # ── private helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _pad(image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Pad to the next multiple of 8 using reflect padding."""
        h, w = image.shape[:2]
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8
        if pad_h or pad_w:
            image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
            mask  = np.pad(mask,  ((0, pad_h), (0, pad_w)),          mode="reflect")
        return image, mask

    def _img_to_tensor(self, image: np.ndarray) -> torch.Tensor:
        """(H, W, 3) uint8 -> (1, 3, H, W) float32 in [0, 1]."""
        t = torch.from_numpy(image).float() / 255.0
        return t.permute(2, 0, 1).unsqueeze(0).to(self._device)

    def _mask_to_tensor(self, mask: np.ndarray) -> torch.Tensor:
        """(H, W) any dtype -> (1, 1, H, W) float32 with 1 = fill."""
        binary = (mask > 0).astype(np.float32)
        return torch.from_numpy(binary).unsqueeze(0).unsqueeze(0).to(self._device)

    @staticmethod
    def _tensor_to_img(t: torch.Tensor) -> np.ndarray:
        """(1, 3, H, W) float32 [0, 1] -> (H, W, 3) uint8."""
        arr = t.squeeze(0).permute(1, 2, 0).cpu().float().numpy()
        return (arr * 255).clip(0, 255).astype(np.uint8)
