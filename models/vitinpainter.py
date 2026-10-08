"""
ViTInpainter — Transformer-based inpainting model.

TODO (Alexander): implement this module.

Suggested approach
------------------
Use a pretrained ViT-based inpainting model, for example:
  - MAT  (https://github.com/fenglinglwb/MAT)
  - RePaint (https://github.com/andreas128/RePaint)

The class must follow the BaseInpainter contract:
    result = ViTInpainter().inpaint(image_rgb_uint8, mask_uint8)

Compatibility spike checklist (Python 3.12):
    1. import dependencies  ✓ / ✗
    2. load pretrained weights
    3. run one image + mask
    4. save one output
    5. record VRAM and inference time
"""

import numpy as np

from .base import BaseInpainter, InpaintResult


class ViTInpainter(BaseInpainter):
    """
    Transformer-based inpainting model.

    Implementation is owned by Alexander.
    Until implemented, calling `inpaint` raises NotImplementedError.
    """

    def __init__(self):
        raise NotImplementedError(
            "ViTInpainter is not yet implemented. "
            "Alexander: please fill in this class."
        )

    @property
    def name(self) -> str:
        return "ViTInpainter"

    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> InpaintResult:
        raise NotImplementedError("ViTInpainter is not yet implemented.")
