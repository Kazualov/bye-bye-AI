"""
FLUX.1-Fill — Diffusion-based inpainting model.

TODO (Egor): implement this module.

Suggested approach
------------------
Use the Black Forest Labs FLUX.1-Fill model via the Hugging Face
diffusers library:

    from diffusers import FluxFillPipeline
    pipe = FluxFillPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-Fill-dev",
        torch_dtype=torch.bfloat16,
    ).to("cuda")

The class must follow the BaseInpainter contract:
    result = FluxFillInpainter().inpaint(image_rgb_uint8, mask_uint8)

Compatibility spike checklist (Python 3.12):
    1. import dependencies  ✓ / ✗
    2. load pretrained weights
    3. run one image + mask
    4. save one output
    5. record VRAM and inference time

Notes
-----
- FLUX.1-Fill requires significant VRAM (>= 16 GB recommended).
- Investigate num_inference_steps trade-off for quality vs. speed.
- Document typical failure modes (hallucinations, style drift).
"""

import numpy as np

from .base import BaseInpainter, InpaintResult


class FluxFillInpainter(BaseInpainter):
    """
    Diffusion-based inpainting model using FLUX.1-Fill.

    Implementation is owned by Egor.
    Until implemented, calling `inpaint` raises NotImplementedError.
    """

    def __init__(self):
        raise NotImplementedError(
            "FluxFillInpainter is not yet implemented. "
            "Egor: please fill in this class."
        )

    @property
    def name(self) -> str:
        return "FLUX.1-Fill"

    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> InpaintResult:
        raise NotImplementedError("FluxFillInpainter is not yet implemented.")
