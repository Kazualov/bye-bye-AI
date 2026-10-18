"""
FLUX.1-Fill — Diffusion-based inpainting and object removal model.
"""

import time
import cv2
import numpy as np
import torch
from PIL import Image

from .base import BaseInpainter, InpaintResult


class FluxFillInpainter(BaseInpainter):
    """
    Diffusion-based inpainting model using FLUX.1-Fill-dev,
    tuned specifically for object removal and background reconstruction.
    """

    def __init__(
        self,
        model_id: str = "black-forest-labs/FLUX.1-Fill-dev",
        num_inference_steps: int = 24,
        guidance_scale: float = 15.0,
        max_side: int = 768,
        prompt: str = (
            "empty background, nothing here, seamless background continuation, "
            "natural surface texture, clean scene without foreground object"
        ),
        seed: int = 42,
        quantize: bool = True,
        mask_expand_px: int = 8,
    ):
        """
        Args:
            model_id:            Hugging Face repo ID for FLUX.1-Fill.
            num_inference_steps: Denoising steps (20-28 balances speed and quality).
            guidance_scale:      Lower values (10-20) reduce object hallucinations.
            max_side:            Max image side during inference (multiple of 16).
            prompt:              Fixed prompt instructing background continuation.
            seed:                Random seed for reproducibility.
            quantize:            Use 4-bit NF4 quantization for 16 GB VRAM GPUs.
            mask_expand_px:      Slightly dilate input mask so object edges don't leak.
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
        self.mask_expand_px = mask_expand_px

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
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
        scale = min(1.0, max_side / max(orig_w, orig_h))
        new_w = max(16, int(round((orig_w * scale) / 16.0)) * 16)
        new_h = max(16, int(round((orig_h * scale) / 16.0)) * 16)
        return new_w, new_h

    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> InpaintResult:
        orig_h, orig_w = image.shape[:2]
        binary_mask = (mask > 0).astype(np.uint8)

        if not np.any(binary_mask):
            return InpaintResult(
                image=image.copy(),
                inference_time=0.0,
                metadata={"skipped": True},
            )

        # Expand mask slightly so residual object boundary pixels don't trigger re-generation
        if self.mask_expand_px > 0:
            k = 2 * self.mask_expand_px + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            feed_mask = cv2.dilate(binary_mask, kernel, iterations=1) * 255
        else:
            feed_mask = binary_mask * 255

        target_w, target_h = self._compute_target_size(orig_w, orig_h, self.max_side)

        pil_image = Image.fromarray(image).convert("RGB")
        pil_mask = Image.fromarray(feed_mask).convert("L")

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

        if out_pil.size != (orig_w, orig_h):
            out_pil = out_pil.resize((orig_w, orig_h), Image.Resampling.BICUBIC)

        pred_rgb = np.array(out_pil, dtype=np.uint8)

        # Composite inside the feed_mask so dilated object borders are also replaced cleanly
        comp_mask = (feed_mask > 0)[..., None]
        final_rgb = np.where(comp_mask, pred_rgb, image).astype(np.uint8)

        metadata = {
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
            "inference_resolution": f"{target_w}x{target_h}",
            "quantized_4bit": self.quantize,
            "mask_expand_px": self.mask_expand_px,
        }
        if self.device == "cuda":
            metadata["vram_allocated_mb"] = round(torch.cuda.max_memory_allocated() / (1024 ** 2), 2)
            metadata["vram_reserved_mb"] = round(torch.cuda.max_memory_reserved() / (1024 ** 2), 2)

        return InpaintResult(
            image=final_rgb,
            inference_time=inference_time,
            metadata=metadata,
        )
