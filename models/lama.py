"""
LaMa (Large Mask inpainting) — CNN-based inpainting model.

Loads the official big-lama TorchScript weights directly from
HuggingFace Hub (smartywu/big-lama), so no intermediate wrapper
package with stale Pillow pins is required.

Weights are cached by huggingface_hub on first use (~200 MB).
"""

import time

import numpy as np
import torch
from huggingface_hub import hf_hub_download

from .base import BaseInpainter, InpaintResult

_HF_REPO = "smartywu/big-lama"
_HF_FILE = "big-lama.pt"


class LaMaInpainter(BaseInpainter):
    """
    LaMa inpainter using the big-lama TorchScript model.

    The model is loaded once at construction and reused for every
    `inpaint` call. Uses MPS on Apple Silicon, CUDA if available,
    otherwise CPU.
    """

    def __init__(self):
        weights_path = hf_hub_download(repo_id=_HF_REPO, filename=_HF_FILE)
        self._device = self._pick_device()
        self._model = torch.jit.load(weights_path, map_location=self._device)
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
