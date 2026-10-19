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
    Diffusion-based inpainting model using FLUX.1-Fill-dev.

    Pre/post-processing follows ViTInpainter: no mask dilation by default,
    original pixels outside the mask, inward-only feathering.
    """

    def __init__(
        self,
        model_id: str = "black-forest-labs/FLUX.1-Fill-dev",
        num_inference_steps: int = 28,
        guidance_scale: float = 30.0,
        max_side: int = 768,
        prompt: str = "",
        seed: int = 42,
        quantize: bool = True,
        mask_expand_px: int = 0,
        feather_radius: float = 3.0,
    ):
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
        self.feather_radius = feather_radius

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # ВАЖНО: FLUX.1-Fill обучен строго в bfloat16 (в float16 дает артефакты и кашу)
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

        # 1. Опциональное расширение маски (по умолчанию выключено, как в ViTInpainter:
        # бенчмарк сам проверяет dilated-маски, а лишнее расширение портит метрики)
        if self.mask_expand_px > 0:
            k = 2 * self.mask_expand_px + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            working_mask = cv2.dilate(binary_mask, kernel, iterations=1)
        else:
            working_mask = binary_mask

        feed_mask = working_mask * 255

        # 2. Подаём оригинальное изображение без pre-fill: FluxFillPipeline сам
        # обнуляет пиксели под маской (masked_image = image * (1 - mask))
        target_w, target_h = self._compute_target_size(orig_w, orig_h, self.max_side)

        pil_image_in = Image.fromarray(image).convert("RGB")
        pil_mask_in = Image.fromarray(feed_mask).convert("L")

        if (target_w, target_h) != (orig_w, orig_h):
            pil_image_in = pil_image_in.resize((target_w, target_h), Image.Resampling.LANCZOS)
            # Консервативное уменьшение, чтобы тонкие края маски не съедались
            pil_mask_in = pil_mask_in.resize((target_w, target_h), Image.Resampling.BILINEAR)
            pil_mask_in = Image.fromarray(
                ((np.array(pil_mask_in) > 10).astype(np.uint8) * 255)
            )

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
            out_pil = out_pil.resize((orig_w, orig_h), Image.Resampling.LANCZOS)

        pred_rgb = np.array(out_pil, dtype=np.uint8)

        # 3. Сшивка как в ViTInpainter: снаружи маски пиксели строго оригинальные,
        # плавный переход только внутри маски на feather_radius пикселей
        if self.feather_radius > 0:
            dist = cv2.distanceTransform(working_mask, cv2.DIST_L2, 3)
            alpha = np.clip(dist / float(self.feather_radius), 0.0, 1.0)[..., None]
        else:
            alpha = working_mask[..., None].astype(np.float32)
        final_rgb = (
            image.astype(np.float32) * (1.0 - alpha) + pred_rgb.astype(np.float32) * alpha
        ).round().astype(np.uint8)

        metadata = {
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
            "inference_resolution": f"{target_w}x{target_h}",
            "quantized_4bit": self.quantize,
            "mask_expand_px": self.mask_expand_px,
            "feather_radius": self.feather_radius,
        }
        if self.device == "cuda":
            metadata["vram_allocated_mb"] = round(torch.cuda.max_memory_allocated() / (1024 ** 2), 2)
            metadata["vram_reserved_mb"] = round(torch.cuda.max_memory_reserved() / (1024 ** 2), 2)

        return InpaintResult(
            image=final_rgb,
            inference_time=inference_time,
            metadata=metadata,
        )
