"""
Mask generation utilities for the sensitivity experiments.

Given a base binary mask, this module can produce derived masks:
  - native   : the mask as-is
  - dilated  : morphologically dilated by N pixels
  - bbox     : tight axis-aligned bounding box of the mask
  - ellipse  : inscribed ellipse inside the bounding box
  - rectangle: same as bbox but drawn as a filled rectangle

These are used in the benchmark to test how sensitive each model
is to mask precision, size, and shape.
"""

from enum import Enum

import cv2
import numpy as np


class MaskType(str, Enum):
    NATIVE = "native"
    DILATED = "dilated"
    BBOX = "bbox"
    ELLIPSE = "ellipse"
    RECTANGLE = "rectangle"


class MaskFactory:
    """Create derived masks from a base binary mask."""

    @staticmethod
    def native(mask: np.ndarray) -> np.ndarray:
        """Return the mask unchanged (uint8, non-zero = fill)."""
        return (mask > 0).astype(np.uint8) * 255

    @staticmethod
    def dilated(mask: np.ndarray, radius: int = 10) -> np.ndarray:
        """Dilate the mask by `radius` pixels using a circular kernel."""
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
        )
        binary = (mask > 0).astype(np.uint8)
        dilated = cv2.dilate(binary, kernel, iterations=1)
        return dilated * 255

    @staticmethod
    def bbox(mask: np.ndarray) -> np.ndarray:
        """Fill the tight axis-aligned bounding box of the mask."""
        binary = mask > 0
        if not np.any(binary):
            return np.zeros_like(mask, dtype=np.uint8)

        rows = np.any(binary, axis=1)
        cols = np.any(binary, axis=0)
        y0, y1 = np.where(rows)[0][[0, -1]]
        x0, x1 = np.where(cols)[0][[0, -1]]

        result = np.zeros(mask.shape, dtype=np.uint8)
        result[y0 : y1 + 1, x0 : x1 + 1] = 255
        return result

    @staticmethod
    def ellipse(mask: np.ndarray) -> np.ndarray:
        """Draw a filled ellipse inscribed in the mask's bounding box."""
        binary = mask > 0
        if not np.any(binary):
            return np.zeros_like(mask, dtype=np.uint8)

        rows = np.any(binary, axis=1)
        cols = np.any(binary, axis=0)
        y0, y1 = int(np.where(rows)[0][0]), int(np.where(rows)[0][-1])
        x0, x1 = int(np.where(cols)[0][0]), int(np.where(cols)[0][-1])

        center = ((x0 + x1) // 2, (y0 + y1) // 2)
        axes = ((x1 - x0) // 2, (y1 - y0) // 2)

        result = np.zeros(mask.shape, dtype=np.uint8)
        cv2.ellipse(result, center, axes, 0, 0, 360, 255, thickness=-1)
        return result

    @staticmethod
    def rectangle(mask: np.ndarray) -> np.ndarray:
        """Alias for bbox — explicit rectangle shape."""
        return MaskFactory.bbox(mask)

    @classmethod
    def make(cls, mask: np.ndarray, mask_type: MaskType, dilation_radius: int = 10) -> np.ndarray:
        """
        Dispatch helper.

        Args:
            mask:            Source mask, any dtype.
            mask_type:       Which derived mask to produce.
            dilation_radius: Only used when mask_type is DILATED.
        """
        if mask_type == MaskType.NATIVE:
            return cls.native(mask)
        elif mask_type == MaskType.DILATED:
            return cls.dilated(mask, radius=dilation_radius)
        elif mask_type == MaskType.BBOX:
            return cls.bbox(mask)
        elif mask_type == MaskType.ELLIPSE:
            return cls.ellipse(mask)
        elif mask_type == MaskType.RECTANGLE:
            return cls.rectangle(mask)
        else:
            raise ValueError(f"Unknown mask type: {mask_type}")
