"""
LaMa (Large Mask inpainting) — CNN-based inpainting model.

Loads the official big-lama TorchScript weights (hosted by the IOPaint
project). Weights are downloaded automatically on first use (~200 MB) into
~/.cache/inpainting-bench/ and reused afterwards.

The download uses CA certificates from `certifi`, so it works on macOS /
Windows / Linux without any manual SSL setup.
"""

import ssl
import threading
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import torch

from .base import BaseInpainter, InpaintResult

# TorchScript export of big-lama (hosted by the IOPaint project)
_WEIGHTS_URL = (
    "https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt"
)
_WEIGHTS_PATH = Path.home() / ".cache" / "inpainting-bench" / "big-lama.pt"
_download_lock = threading.Lock()


# ── mask preprocessing ─────────────────────────────────────────────────────────

def prepare_mask(mask: np.ndarray, dilation_frac: float = 0.01, min_px: int = 3) -> np.ndarray:
    """Binarize + close small holes + dilate. Returns uint8 {0, 1}.

    Works for masks in {0, 255} as well as {0, 1}.
    """
    h, w = mask.shape[:2]
    binary = (mask > (127 if mask.max() > 1 else 0)).astype(np.uint8)

    k = max(min_px, int(round(dilation_frac * max(h, w))))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)  # close holes inside
    return cv2.dilate(binary, kernel)


# ── weights download ───────────────────────────────────────────────────────────

def _ssl_context() -> ssl.SSLContext:
    """CA bundle from certifi; falls back to the system store if missing."""
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def _download(url: str, dest: Path, retries: int = 3) -> None:
    tmp = dest.with_suffix(".part")
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            tmp.unlink(missing_ok=True)
            req = urllib.request.Request(url, headers={"User-Agent": "inpainting-bench"})
            with urllib.request.urlopen(req, context=_ssl_context(), timeout=60) as r, \
                    open(tmp, "wb") as f:
                total = int(r.headers.get("Content-Length") or 0)
                done, next_log = 0, 0
                while chunk := r.read(1 << 20):
                    f.write(chunk)
                    done += len(chunk)
                    if total and done * 100 // total >= next_log:
                        print(f"[lama] downloading weights: {done * 100 // total}%", flush=True)
                        next_log += 10
            if total and tmp.stat().st_size != total:
                raise IOError(f"incomplete download: {tmp.stat().st_size}/{total} bytes")
            tmp.rename(dest)  # file appears only once fully downloaded
            return
        except Exception as e:
            last_err = e
            tmp.unlink(missing_ok=True)
            print(f"[lama] download attempt {attempt}/{retries} failed: {e}", flush=True)
            time.sleep(2 * attempt)
    raise RuntimeError(f"Cannot download LaMa weights from {url}: {last_err}")


def _ensure_weights() -> Path:
    if _WEIGHTS_PATH.exists():
        return _WEIGHTS_PATH
    with _download_lock:  # parallel jobs don't download twice
        if not _WEIGHTS_PATH.exists():
            _WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
            print("[lama] weights not found, downloading (~200 MB)...", flush=True)
            _download(_WEIGHTS_URL, _WEIGHTS_PATH)
    return _WEIGHTS_PATH


# ── model ──────────────────────────────────────────────────────────────────────

class LaMaInpainter(BaseInpainter):
    def __init__(self, dilation_frac: float = 0.01, max_side: int | None = 1024):
        """
        Args:
            dilation_frac: mask dilation as a fraction of the longer image side.
            max_side:      images larger than this are downscaled for inference
                           (result is upscaled back); None = never resize.
        """
        weights_path = _ensure_weights()
        self.dilation_frac = dilation_frac
        self.max_side = max_side
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
        mask = prepare_mask(mask, self.dilation_frac)
        h, w = image.shape[:2]

        scale = 1.0 if self.max_side is None else min(1.0, self.max_side / max(h, w))
        if scale < 1.0:
            sw, sh = round(w * scale), round(h * scale)
            img_in = cv2.resize(image, (sw, sh), interpolation=cv2.INTER_AREA)
            mask_in = (cv2.resize(mask, (sw, sh), interpolation=cv2.INTER_AREA) > 0).astype(np.uint8)
        else:
            img_in, mask_in = image, mask

        oh, ow = img_in.shape[:2]
        img_p, mask_p = self._pad(img_in, mask_in)
        img_t, mask_t = self._img_to_tensor(img_p), self._mask_to_tensor(mask_p)

        start = time.perf_counter()
        with torch.no_grad():
            out_t = self._model(img_t, mask_t)
        elapsed = time.perf_counter() - start

        out = self._tensor_to_img(out_t)[:oh, :ow]
        if scale < 1.0:
            out = cv2.resize(out, (w, h), interpolation=cv2.INTER_CUBIC)

        # original pixels outside the mask stay untouched
        final = np.where(mask[..., None] > 0, out, image).astype(np.uint8)
        return InpaintResult(image=final, inference_time=elapsed)

    # ── private helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _pad(image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Pad to the next multiple of 8 (image: reflect, mask: zeros)."""
        h, w = image.shape[:2]
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8
        if pad_h or pad_w:
            image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
            mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0)
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