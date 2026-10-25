
"""
PixArt-α — Diffusion Transformer (DiT) inpainting model.

Zero-shot inpainting: SDEdit-style partial noising + blended latent diffusion
(background latents are re-injected outside the mask on every step).

Pipeline:
  1. Crop a square window with context around the mask (no aspect distortion).
  2. Prefill the hole with OpenCV (coarse Telea + fine NS) for a sane init.
  3. Denoise the masked region with PixArt-α (prompt + CFG).
  4. Paste the result back with a feathered alpha mask.
"""

import gc
import ssl
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .base import BaseInpainter, InpaintResult

DEFAULT_MODEL_ID = "PixArt-alpha/PixArt-XL-2-512x512"
_CACHE_DIR = Path.home() / ".cache" / "inpainting-bench"
# v3: new file name -> old (empty-prompt) cache is ignored
_PROMPT_CACHE_FILE = _CACHE_DIR / "pixart_alpha_prompt_v3.pt"

PROMPT = "a clean photo, empty background, seamless natural texture"
NEG_PROMPT = "object, text, watermark, blurry, smudge, low quality"


def _patch_ssl_if_needed() -> None:
    try:
        import certifi

        ssl._create_default_https_context = (
            lambda: ssl.create_default_context(cafile=certifi.where())
        )
    except ImportError:
        pass


class PixArtInpainter(BaseInpainter):
    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str | torch.device | None = None,
        num_inference_steps: int = 50,
        strength: float = 1,
        guidance_scale: float = 7.5,
        seed: int = 42,
        context: float = 2.5,
        min_crop: int = 256,
    ):
        _patch_ssl_if_needed()

        self.model_id = model_id
        self.device = self._pick_device(device)
        self.dtype = (
            torch.float16
            if self.device.type in ("cuda", "mps")
            else torch.float32
        )
        self.num_inference_steps = num_inference_steps
        self.strength = float(np.clip(strength, 0.05, 1.0))
        self.guidance_scale = guidance_scale
        self.seed = seed
        self.context = context      # crop side = mask size * context
        self.min_crop = min_crop    # minimal crop side in pixels

        from diffusers import DDIMScheduler, PixArtAlphaPipeline

        (
            self._prompt_embeds,
            self._prompt_attn_mask,
            self._neg_prompt_embeds,
            self._neg_prompt_attn_mask,
        ) = self._load_or_encode_prompt(model_id)

        self.pipe = PixArtAlphaPipeline.from_pretrained(
            model_id,
            text_encoder=None,
            tokenizer=None,
            torch_dtype=self.dtype,
        )
        self.pipe.scheduler = DDIMScheduler.from_config(
            self.pipe.scheduler.config
        )
        self.pipe.to(self.device)

        if hasattr(self.pipe, "enable_attention_slicing"):
            self.pipe.enable_attention_slicing()
        if hasattr(self.pipe.vae, "enable_slicing"):
            self.pipe.vae.enable_slicing()

        self.pipe.transformer.eval()
        self.pipe.vae.eval()

    @staticmethod
    def _pick_device(device: str | torch.device | None = None) -> torch.device:
        if device is not None:
            return torch.device(device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    @staticmethod
    def _flush_memory() -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            try:
                torch.mps.empty_cache()
            except Exception:
                pass

    def _load_or_encode_prompt(
        self, model_id: str
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if _PROMPT_CACHE_FILE.exists():
            try:
                cached = torch.load(
                    _PROMPT_CACHE_FILE, map_location="cpu", weights_only=True
                )
                if (
                    cached.get("model_id") == model_id
                    and cached.get("prompt") == PROMPT
                    and cached.get("neg_prompt") == NEG_PROMPT
                ):
                    return (
                        cached["prompt_embeds"].to(
                            device=self.device, dtype=self.dtype
                        ),
                        cached["prompt_attn_mask"].to(device=self.device),
                        cached["neg_prompt_embeds"].to(
                            device=self.device, dtype=self.dtype
                        ),
                        cached["neg_prompt_attn_mask"].to(device=self.device),
                    )
            except Exception:
                pass

        from diffusers import PixArtAlphaPipeline

        encoder_pipe = PixArtAlphaPipeline.from_pretrained(
            model_id,
            transformer=None,
            vae=None,
            torch_dtype=self.dtype,
        ).to(self.device)

        with torch.no_grad():
            (
                prompt_embeds,
                prompt_attn_mask,
                neg_prompt_embeds,
                neg_prompt_attn_mask,
            ) = encoder_pipe.encode_prompt(
                prompt=PROMPT,
                do_classifier_free_guidance=True,
                negative_prompt=NEG_PROMPT,
                num_images_per_prompt=1,
                device=self.device,
                clean_caption=False,
            )

        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        tmp_path = _PROMPT_CACHE_FILE.with_suffix(".part")
        torch.save(
            {
                "model_id": model_id,
                "prompt": PROMPT,
                "neg_prompt": NEG_PROMPT,
                "prompt_embeds": prompt_embeds.cpu(),
                "prompt_attn_mask": prompt_attn_mask.cpu(),
                "neg_prompt_embeds": neg_prompt_embeds.cpu(),
                "neg_prompt_attn_mask": neg_prompt_attn_mask.cpu(),
            },
            tmp_path,
        )
        tmp_path.replace(_PROMPT_CACHE_FILE)

        del encoder_pipe
        self._flush_memory()

        return (
            prompt_embeds.to(device=self.device, dtype=self.dtype),
            prompt_attn_mask.to(device=self.device),
            neg_prompt_embeds.to(device=self.device, dtype=self.dtype),
            neg_prompt_attn_mask.to(device=self.device),
        )

    # ── context crop ───────────────────────────────────────────────────────────

    @staticmethod
    def _context_crop(
        mask: np.ndarray, h: int, w: int, context: float, min_side: int
    ) -> tuple[int, int, int]:
        """Square window centred on the mask, `context`x larger than the mask."""
        ys, xs = np.where(mask > 0)
        y0, y1 = ys.min(), ys.max() + 1
        x0, x1 = xs.min(), xs.max() + 1
        side = int(max(y1 - y0, x1 - x0) * context)
        side = min(max(side, min_side), h, w)
        top = int(np.clip((y0 + y1) / 2 - side / 2, 0, h - side))
        left = int(np.clip((x0 + x1) / 2 - side / 2, 0, w - side))
        return top, left, side

    # ── prefill ────────────────────────────────────────────────────────────────

    @staticmethod
    def _multiscale_prefill(
        image: np.ndarray, binary_mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        h, w = image.shape[:2]

        # Dilation depends on the OBJECT size (not image size): covers
        # antialiasing / halo on the object edge.
        ys, xs = np.where(binary_mask > 0)
        obj_size = max(ys.max() - ys.min(), xs.max() - xs.min())
        dil_radius = int(np.clip(obj_size * 0.08, 8, 60))
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * dil_radius + 1, 2 * dil_radius + 1)
        )
        work_mask = cv2.dilate(binary_mask, kernel, iterations=1)

        # Coarse pyramid fill at 1/4 resolution for clean global color gradient
        small_w, small_h = max(64, w // 4), max(64, h // 4)
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        small_bgr = cv2.resize(
            bgr, (small_w, small_h), interpolation=cv2.INTER_AREA
        )
        small_mask = cv2.resize(
            work_mask, (small_w, small_h), interpolation=cv2.INTER_NEAREST
        )
        coarse_bgr = cv2.inpaint(small_bgr, small_mask, 15, cv2.INPAINT_TELEA)
        coarse_up_bgr = cv2.resize(
            coarse_bgr, (w, h), interpolation=cv2.INTER_CUBIC
        )

        # Full-res Navier-Stokes fill for boundary structure
        fine_bgr = cv2.inpaint(bgr, work_mask, 15, cv2.INPAINT_NS)
        blended_bgr = cv2.addWeighted(coarse_up_bgr, 0.70, fine_bgr, 0.30, 0)

        # Feathered alpha mask for seamless transition
        soft_alpha = cv2.GaussianBlur(
            work_mask.astype(np.float32),
            (0, 0),
            sigmaX=max(3.0, dil_radius * 0.45),
        )[..., None]
        soft_alpha = np.maximum(
            soft_alpha, binary_mask[..., None].astype(np.float32)
        )

        orig_f = bgr.astype(np.float32)
        prefilled_bgr = (
            orig_f * (1.0 - soft_alpha) + blended_bgr.astype(np.float32) * soft_alpha
        ).clip(0, 255).astype(np.uint8)

        prefilled_rgb = cv2.cvtColor(prefilled_bgr, cv2.COLOR_BGR2RGB)
        return prefilled_rgb, work_mask, soft_alpha

    @property
    def name(self) -> str:
        return "PixArt-α"

    # ── public API: crop wrapper ───────────────────────────────────────────────

    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> InpaintResult:
        binary = (mask > 0).astype(np.uint8)

        if not np.any(binary):
            return InpaintResult(
                image=image.copy(),
                inference_time=0.0,
                metadata={"device": str(self.device), "steps": 0},
            )

        h, w = image.shape[:2]
        top, left, side = self._context_crop(
            binary, h, w, self.context, self.min_crop
        )
        sl = (slice(top, top + side), slice(left, left + side))

        res = self._inpaint_square(
            np.ascontiguousarray(image[sl]),
            np.ascontiguousarray(binary[sl]),
        )

        out = image.copy()
        out[sl] = res.image
        meta = dict(res.metadata)
        meta["crop"] = (top, left, side)
        return InpaintResult(
            image=out, inference_time=res.inference_time, metadata=meta
        )

    # ── core: square crop inpainting ───────────────────────────────────────────

    def _inpaint_square(
        self, image: np.ndarray, binary_mask: np.ndarray
    ) -> InpaintResult:
        orig_h, orig_w = image.shape[:2]

        start_time = time.perf_counter()

        prefilled_rgb, work_mask, soft_alpha = self._multiscale_prefill(
            image, binary_mask
        )

        target_res = 512
        img_512 = np.array(
            Image.fromarray(prefilled_rgb).resize(
                (target_res, target_res), Image.LANCZOS
            )
        )
        mask_512 = np.array(
            Image.fromarray(work_mask * 255).resize(
                (target_res, target_res), Image.NEAREST
            )
        )

        img_tensor = (
            torch.from_numpy(img_512)
            .permute(2, 0, 1)
            .float()
            .unsqueeze(0)
            .to(device=self.device, dtype=self.dtype)
        )
        img_tensor = (img_tensor / 127.5) - 1.0

        latent_size = target_res // self.pipe.vae_scale_factor
        mask_tensor = (
            torch.from_numpy((mask_512 > 0).astype(np.float32))
            .unsqueeze(0)
            .unsqueeze(0)
            .to(device=self.device, dtype=self.dtype)
        )
        latent_mask = F.interpolate(
            mask_tensor,
            size=(latent_size, latent_size),
            mode="bilinear",
            align_corners=False,
        )
        latent_mask = F.avg_pool2d(
            latent_mask, kernel_size=5, stride=1, padding=2
        )

        with torch.no_grad():
            posterior = self.pipe.vae.encode(img_tensor).latent_dist
            init_latents = posterior.mean * self.pipe.vae.config.scaling_factor

            self.pipe.scheduler.set_timesteps(
                self.num_inference_steps, device=self.device
            )
            timesteps = self.pipe.scheduler.timesteps
            init_step = min(
                int(len(timesteps) * (1.0 - self.strength)),
                len(timesteps) - 1,
            )
            active_timesteps = timesteps[init_step:]

            generator = torch.Generator(device="cpu").manual_seed(self.seed)
            noise = torch.randn(
                init_latents.shape,
                generator=generator,
                dtype=self.dtype,
            ).to(self.device)

            latents = self.pipe.scheduler.add_noise(
                init_latents, noise, active_timesteps[:1]
            )

            do_cfg = self.guidance_scale > 1.0
            if do_cfg:
                prompt_embeds = torch.cat(
                    [self._neg_prompt_embeds, self._prompt_embeds], dim=0
                )
                prompt_attn_mask = torch.cat(
                    [self._neg_prompt_attn_mask, self._prompt_attn_mask], dim=0
                )
            else:
                prompt_embeds = self._prompt_embeds
                prompt_attn_mask = self._prompt_attn_mask

            if self.pipe.transformer.config.sample_size == 128:
                res_cond = torch.tensor(
                    [[target_res, target_res]],
                    dtype=self.dtype,
                    device=self.device,
                )
                ar_cond = torch.tensor(
                    [[1.0]], dtype=self.dtype, device=self.device
                )
                if do_cfg:
                    res_cond = torch.cat([res_cond, res_cond], dim=0)
                    ar_cond = torch.cat([ar_cond, ar_cond], dim=0)
                added_cond_kwargs = {
                    "resolution": res_cond,
                    "aspect_ratio": ar_cond,
                }
            else:
                added_cond_kwargs = {"resolution": None, "aspect_ratio": None}

            latent_channels = init_latents.shape[1]

            for i, t in enumerate(active_timesteps):
                latent_model_input = (
                    torch.cat([latents] * 2) if do_cfg else latents
                )
                latent_model_input = self.pipe.scheduler.scale_model_input(
                    latent_model_input, t
                )

                current_timestep = t
                if not torch.is_tensor(current_timestep):
                    current_timestep = torch.tensor(
                        [current_timestep],
                        dtype=torch.long,
                        device=self.device,
                    )
                elif current_timestep.ndim == 0:
                    current_timestep = current_timestep.unsqueeze(0).to(
                        self.device
                    )
                current_timestep = current_timestep.expand(
                    latent_model_input.shape[0]
                )

                noise_pred = self.pipe.transformer(
                    latent_model_input,
                    encoder_hidden_states=prompt_embeds,
                    encoder_attention_mask=prompt_attn_mask,
                    timestep=current_timestep,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]

                if do_cfg:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                if (
                    self.pipe.transformer.config.out_channels // 2
                    == latent_channels
                ):
                    noise_pred = noise_pred.chunk(2, dim=1)[0]

                latents = self.pipe.scheduler.step(
                    noise_pred, t, latents, return_dict=False
                )[0]

                if i < len(active_timesteps) - 1:
                    next_t = active_timesteps[i + 1]
                    bg_latents = self.pipe.scheduler.add_noise(
                        init_latents, noise, next_t.unsqueeze(0)
                    )
                    latents = (
                        bg_latents * (1.0 - latent_mask)
                        + latents * latent_mask
                    )
                else:
                    latents = (
                        init_latents * (1.0 - latent_mask)
                        + latents * latent_mask
                    )

            decoded = self.pipe.vae.decode(
                latents / self.pipe.vae.config.scaling_factor,
                return_dict=False,
            )[0]

        inference_time = time.perf_counter() - start_time

        out_img = decoded.squeeze(0).permute(1, 2, 0).cpu().float().numpy()
        out_img = ((out_img + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
        out_img = np.array(
            Image.fromarray(out_img).resize((orig_w, orig_h), Image.LANCZOS)
        )

        # Blend with the feathered dilated mask: outside it the original
        # pixels are untouched, so there is no seam when pasting the crop back.
        final_img = (
            image.astype(np.float32) * (1.0 - soft_alpha)
            + out_img.astype(np.float32) * soft_alpha
        ).clip(0, 255).astype(np.uint8)

        return InpaintResult(
            image=final_img,
            inference_time=inference_time,
            metadata={
                "device": str(self.device),
                "model_id": self.model_id,
                "steps": len(active_timesteps),
            },
        )