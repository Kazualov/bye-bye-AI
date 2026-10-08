"""
OpenCV Telea inpainting — classical baseline.

Uses cv2.INPAINT_TELEA which propagates texture from the boundary
inward. No GPU or pretrained weights required.
"""

import time

import cv2
import numpy as np

from .base import BaseInpainter, InpaintResult


class OpenCVTelea(BaseInpainter):
    """
    Classical Telea inpainting via OpenCV.

    This is the fastest model and serves as the lower-bound baseline.
    It performs no semantic reasoning; quality degrades on large masks.
    """

    def __init__(self, inpaint_radius: int = 3):
        """
        Args:
            inpaint_radius: Neighbourhood radius used by the algorithm.
                            3–5 pixels is the standard range.
        """
        self.inpaint_radius = inpaint_radius

    @property
    def name(self) -> str:
        return "OpenCV Telea"

    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> InpaintResult:
        # OpenCV works in BGR; convert in, convert out
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # cv2.inpaint expects uint8 mask; non-zero = fill
        binary_mask = (mask > 0).astype(np.uint8)

        start = time.perf_counter()
        result_bgr = cv2.inpaint(bgr, binary_mask, self.inpaint_radius, cv2.INPAINT_TELEA)
        elapsed = time.perf_counter() - start

        result = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
        return InpaintResult(image=result, inference_time=elapsed)
