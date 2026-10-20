"""
FLUX.1-Fill — Diffusion-based inpainting and object removal model.

Memory layout tuned for consumer GPUs (8 GB VRAM / 16 GB RAM):
  - transformer: pre-quantized GGUF checkpoint (Q6_K by default, ~9.9 GB,
    visually almost lossless). Loaded directly, so the 24 GB bf16 weights
    never have to fit into RAM (bitsandbytes nf4 quantizes on load and needs them).
  - text encoders: the prompt is fixed, so its embeddings are computed once
    on CPU, cached to disk and T5/CLIP are never kept in memory (lossless).
  - if the transformer does not fit into VRAM, it is streamed block by block
    (group offloading) instead of being moved to the GPU as a whole.

Env overrides (handy for Kaggle / CLI benchmark):
    FLUX_GGUF=Q4_K_S|Q5_K_S|Q6_K|Q8_0|none   (none = original bf16 weights)
    FLUX_OFFLOAD=auto|model|group
"""

import gc
import hashlib
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from .base import BaseInpainter, InpaintResult

_GGUF_REPO = "YarvixPA/FLUX.1-Fill-dev-GGUF"
_CACHE_DIR = Path.home() / ".cache" / "inpainting-bench"


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
        gguf_quant: str | None = "Q6_K",
        offload: str = "auto",
        mask_expand_px: int = 0,
        feather_radius: float = 3.0,
    ):
        from diffusers import FluxFillPipeline

        gguf_quant = os.environ.get("FLUX_GGUF", gguf_quant)
        if gguf_quant is not None and gguf_quant.lower() == "none":
            gguf_quant = None
        offload = os.environ.get("FLUX_OFFLOAD", offload)

        self.model_id = model_id
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.max_side = max_side
        self.prompt = prompt
        self.seed = seed
        self.gguf_quant = gguf_quant
        self.mask_expand_px = mask_expand_px
        self.feather_radius = feather_radius

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # ВАЖНО: FLUX.1-Fill обучен строго в bfloat16 (в float16 дает артефакты и кашу)
        self.dtype = torch.bfloat16

        # 1. Эмбеддинги промпта — до загрузки трансформера, чтобы T5 и FLUX
        # не лежали в RAM одновременно
        self.prompt_embeds, self.pooled_prompt_embeds = self._load_prompt_embeds()

        # 2. Трансформер
        transformer = self._load_transformer()

        # 3. Пайплайн без текстовых энкодеров (только VAE + трансформер)
        self.pipe = FluxFillPipeline.from_pretrained(
            self.model_id,
            transformer=transformer,
            text_encoder=None,
            text_encoder_2=None,
            tokenizer=None,
            tokenizer_2=None,
            torch_dtype=self.dtype,
        )

        self.offload = self._setup_offload(offload) if self.device == "cuda" else "none"
        self.pipe.vae.enable_slicing()
        self.pipe.vae.enable_tiling()
        self.pipe.set_progress_bar_config(disable=True)

    # ── loading helpers ────────────────────────────────────────────────────────

    def _load_prompt_embeds(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode the fixed prompt once on CPU and cache it to disk."""
        key = hashlib.sha1(f"{self.model_id}|{self.prompt}|256".encode()).hexdigest()[:12]
        cache_path = _CACHE_DIR / f"flux_prompt_{key}.pt"
        if cache_path.exists():
            cached = torch.load(cache_path)
            return cached["prompt_embeds"], cached["pooled_prompt_embeds"]

        from diffusers import FluxFillPipeline

        print(f"  [FLUX] encoding prompt {self.prompt!r} with T5/CLIP on CPU (one-time) ...")
        text_pipe = FluxFillPipeline.from_pretrained(
            self.model_id,
            transformer=None,
            vae=None,
            torch_dtype=self.dtype,
        )
        with torch.no_grad():
            prompt_embeds, pooled_prompt_embeds, _ = text_pipe.encode_prompt(
                prompt=self.prompt,
                prompt_2=None,
                device=torch.device("cpu"),
                max_sequence_length=256,
            )
        del text_pipe
        gc.collect()

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"prompt_embeds": prompt_embeds, "pooled_prompt_embeds": pooled_prompt_embeds},
            cache_path,
        )
        return prompt_embeds, pooled_prompt_embeds

    def _load_transformer(self):
        from diffusers import FluxTransformer2DModel

        if self.gguf_quant is None:
            return FluxTransformer2DModel.from_pretrained(
                self.model_id, subfolder="transformer", torch_dtype=self.dtype
            )

        from diffusers import GGUFQuantizationConfig

        url = (
            f"https://huggingface.co/{_GGUF_REPO}/blob/main/"
            f"flux1-fill-dev-{self.gguf_quant}.gguf"
        )
        print(f"  [FLUX] loading transformer {self.gguf_quant} from {_GGUF_REPO} ...")
        return FluxTransformer2DModel.from_single_file(
            url,
            quantization_config=GGUFQuantizationConfig(compute_dtype=self.dtype),
            config=self.model_id,
            subfolder="transformer",
            torch_dtype=self.dtype,
        )

    def _setup_offload(self, offload: str) -> str:
        if offload == "auto":
            weights_bytes = sum(
                p.numel() * p.element_size() for p in self.pipe.transformer.parameters()
            )
            total_vram = torch.cuda.get_device_properties(0).total_memory
            # ~2 GB запаса под активации 768px, VAE и CUDA-контекст
            offload = "model" if weights_bytes + 2 * 1024 ** 3 < total_vram else "group"

        if offload == "group":
            # Трансформер живёт в RAM и подгружается на GPU поблочно,
            # копирование следующего блока идёт параллельно с вычислением текущего
            self.pipe.transformer.enable_group_offload(
                onload_device=torch.device("cuda"),
                offload_device=torch.device("cpu"),
                offload_type="block_level",
                num_blocks_per_group=1,
                use_stream=True,
                low_cpu_mem_usage=True,
            )
            self.pipe.vae.to("cuda")
        else:
            self.pipe.enable_model_cpu_offload()
        print(f"  [FLUX] offload mode: {offload}")
        return offload

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
            prompt_embeds=self.prompt_embeds.to(self.device),
            pooled_prompt_embeds=self.pooled_prompt_embeds.to(self.device),
            image=pil_image_in,
            mask_image=pil_mask_in,
            height=target_h,
            width=target_w,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
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
            "transformer_weights": self.gguf_quant or "bf16",
            "offload": self.offload,
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
