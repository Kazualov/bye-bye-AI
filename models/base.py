"""
Shared interface for all inpainting models.

Every model must implement `inpaint(image, mask) -> InpaintResult`
so the evaluation pipeline and backend can treat all four models
identically.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class InpaintResult:
    """
    Returned by every model's `inpaint` call.

    Attributes:
        image:          Inpainted RGB image, uint8, shape (H, W, 3).
        inference_time: Seconds spent inside model inference only
                        (excludes image loading/saving).
        metadata:       Optional dict for model-specific info
                        (e.g. VRAM usage, number of steps).
    """

    image: np.ndarray
    inference_time: float
    metadata: dict = field(default_factory=dict)


class BaseInpainter(ABC):
    """
    Abstract base class for all inpainting models.

    Subclasses only need to implement `inpaint`.
    The same instance may be called multiple times.
    """

    @abstractmethod
    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> InpaintResult:
        """
        Fill the masked region of `image`.

        Args:
            image:  RGB image, dtype uint8, shape (H, W, 3).
            mask:   Binary mask, dtype uint8, shape (H, W).
                    Pixels with value > 0 are the region to inpaint.

        Returns:
            InpaintResult containing the inpainted image and timing.
        """
        ...

    @property
    def name(self) -> str:
        """Human-readable model name used in reports and plots."""
        return type(self).__name__
