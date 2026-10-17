"""
FLUX.1-Fill — Diffusion-based inpainting model.
"""

import time
import numpy as np
import torch
from PIL import Image

from .base import BaseInpainter, InpaintResult


class FluxFillInpainter(BaseInpainter):
    """
    Diffusion-based inpainting model using FLUX.1-Fill-dev.
    Supports 4-bit NF4 quantization (via bitsandbytes) to fit into 16 GB VRAM GPUs (e.g. Kaggle T4).
    """

    def __init__(
        self,
        model_id: str = "black-forest-labs/FLUX.1-Fill-dev",
        num_inference_steps: int = 28,
        guidance_scale: float = 30.0,
        max_side: int = 768,
        prompt: str = "clean background, seamless continuation, high quality, natural texture",
        seed: int = 42,
        quantize: bool = True,
    ):
        """
        Args:
            model_id:            Hugging Face repo ID for FLUX.1-Fill.
            num_inference_steps: Denoising steps (20-28 is a good speed/quality sweet spot).
            guidance_scale:      Guidance scale (30.0 is recommended for FLUX.1-Fill).
            max_side:            Maximum resolution side (multiple of 16) to avoid OOM.
            prompt:              Fixed minimal prompt for generic object removal / inpainting.
            seed:                Random seed for reproducibility.
            quantize:            Use 4-bit NF4 quantization if CUDA + bitsandbytes are available.
        """
        from diffusers import FluxFillPipeline, FluxTransformer2DModel
        from transformers import T5EncoderModel

        self.model_id = model_id
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.max_side = max_side
        self.prompt = prompt
        self.seed = seed
        self.quantize = quantize

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # T4 (compute capability 7.5) works best with float16 compute; Ampere+ (8.0+) prefers bfloat16
        if self.device == "cuda" and torch.cuda.get_device_capability()[0] < 8:
            self.dtype = torch.float16
        else:
            self.dtype = torch.bfloat16

        if self.device == "cuda" and self.quantize:
            from diffusers import BitsAndBytesConfig as DiffusersBnBConfig
            from transformers import BitsAndBytesConfig as TransformersBnBConfig

            nf4_diffusers = DiffusersBnBConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=self.dtype,
            )
            nf4_transformers = TransformersBnBConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=self.dtype,
            )

            transformer = FluxTransformer2DModel.from_pretrained(
                self.model_id,
                subfolder="transformer",
                quantization_config=nf4_diffusers,
                torch_dtype=self.dtype,
            )
            text_encoder_2 = T5EncoderModel.from_pretrained(
                self.model_id,
                subfolder="text_encoder_2",
                quantization_config=nf4_transformers,
                torch_dtype=self.dtype,
            )

            self.pipe = FluxFillPipeline.from_pretrained(
                self.model_id,
                transformer=transformer,
                text_encoder_2=text_encoder_2,
                torch_dtype=self.dtype,
            )
        else:
            self.pipe = FluxFillPipeline.from_pretrained(
                self.model_id,
                torch_dtype=self.dtype,
            )

        # Memory optimizations
        if self.device == "cuda":
            self.pipe.enable_model_cpu_offload()
        self.pipe.vae.enable_slicing()
        self.pipe.vae.enable_tiling()
        self.pipe.set_progress_bar_config(disable=True)

    @property
    def name(self) -> str:
        return "FLUX.1-Fill"

    @staticmethod
    def _compute_target_size(orig_w: int, orig_h: int, max_side: int) -> tuple[int, int]:
        """Scale down if needed and align width/height to multiples of 16."""
        scale = min(1.0, max_side / max(orig_w, orig_h))
        new_w = max(16, int(round((orig_w * scale) / 16.0)) * 16)
        new_h = max(16, int(round((orig_h * scale) / 16.0)) * 16)
        return new_w, new_h

    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> InpaintResult:
        """
        Fill the masked region of `image`.

        Args:
            image: RGB image, dtype uint8, shape (H, W, 3).
            mask:  Binary mask, dtype uint8, shape (H, W). Pixels > 0 are inpainted.

        Returns:
            InpaintResult containing the reconstructed RGB image, timing, and VRAM metadata.
        """
        orig_h, orig_w = image.shape[:2]

        # Binary mask in {0, 255}
        binary_mask = ((mask > 0).astype(np.uint8)) * 255

        # Fast-path if mask is completely empty
        if not np.any(binary_mask):
            return InpaintResult(
                image=image.copy(),
                inference_time=0.0,
                metadata={"skipped": True},
            )

        target_w, target_h = self._compute_target_size(orig_w, orig_h, self.max_side)

        pil_image = Image.fromarray(image).convert("RGB")
        pil_mask = Image.fromarray(binary_mask).convert("L")

        if (target_w, target_h) != (orig_w, orig_h):
            pil_image_in = pil_image.resize((target_w, target_h), Image.Resampling.BICUBIC)
            pil_mask_in = pil_mask.resize((target_w, target_h), Image.Resampling.NEAREST)
        else:
            pil_image_in = pil_image
            pil_mask_in = pil_mask

        generator = torch.Generator(device="cpu").manual_seed(self.seed)

        if self.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        t0 = time.perf_counter()

        out_pil = self.pipe(
            prompt=self.prompt,
            image=pil_image_in,
            mask_image=pil_mask_in,
            height=target_h,
            width=target_w,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            max_sequence_length=256,
            generator=generator,
        ).images[0]

        if self.device == "cuda":
            torch.cuda.synchronize()

        inference_time = time.perf_counter() - t0

        # Resize back to original dimensions
        if out_pil.size != (orig_w, orig_h):
            out_pil = out_pil.resize((orig_w, orig_h), Image.Resampling.BICUBIC)

        pred_rgb = np.array(out_pil, dtype=np.uint8)

        # Composite: keep original unmasked pixels untouched for accurate full-image metrics
        mask_bool = (mask > 0)[..., None]
        final_rgb = np.where(mask_bool, pred_rgb, image).astype(np.uint8)

        metadata = {
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
            "inference_resolution": f"{target_w}x{target_h}",
            "quantized_4bit": self.quantize,
        }
        if self.device == "cuda":
            metadata["vram_allocated_mb"] = round(torch.cuda.max_memory_allocated() / (1024 ** 2), 2)
            metadata["vram_reserved_mb"] = round(torch.cuda.max_memory_reserved() / (1024 ** 2), 2)

        return InpaintResult(
            image=final_rgb,
            inference_time=inference_time,
            metadata=metadata,
        )
