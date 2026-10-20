from .base import BaseInpainter, InpaintResult
from .opencv_telea import OpenCVTelea
from .lama import LaMaInpainter

# ViTInpainter needs the vendored MAT code (models/mat/); keep the package
# importable for the other models when it is missing.
try:
    from .vitinpainter import ViTInpainter
except ImportError as e:  # pragma: no cover
    print(f"  [skip] ViTInpainter unavailable: {e}")
    ViTInpainter = None
from .flux_fill import FluxFillInpainter

__all__ = [
    "BaseInpainter",
    "InpaintResult",
    "OpenCVTelea",
    "LaMaInpainter",
    "ViTInpainter",
    "FluxFillInpainter",
]
