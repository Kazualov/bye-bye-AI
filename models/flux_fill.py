"""
FLUX.1-Fill — diffusion-based inpainting (OUT OF SCOPE).

FLUX.1-Fill was planned as the fourth (diffusion) method but was removed from
the project scope. This stub is kept so the shared model interface and the benchmark's model
list stay complete; constructing it raises NotImplementedError and the
benchmark skips it.
"""

import numpy as np

from .base import BaseInpainter, InpaintResult


class FluxFillInpainter(BaseInpainter):
    """Placeholder for FLUX.1-Fill; always raises NotImplementedError."""

    def __init__(self):
        raise NotImplementedError(
            "FluxFillInpainter is not implemented: FLUX.1-Fill is out of project scope."
        )

    @property
    def name(self) -> str:
        return "FLUX.1-Fill"

    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> InpaintResult:
        raise NotImplementedError("FluxFillInpainter is not implemented: FLUX.1-Fill is out of project scope.")
