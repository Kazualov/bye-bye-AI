from .base import BaseInpainter, InpaintResult
from .opencv_telea import OpenCVTelea
from .lama import LaMaInpainter

__all__ = [
    "BaseInpainter", "InpaintResult",
    "OpenCVTelea", "LaMaInpainter",
    "ViTInpainter", "FluxFillInpainter",
]

def __getattr__(name):
    if name == "ViTInpainter":
        from .vitinpainter import ViTInpainter
        return ViTInpainter
    if name == "FluxFillInpainter":
        from .flux_fill import FluxFillInpainter
        return FluxFillInpainter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")