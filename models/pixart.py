"""
PixArt-α — Diffusion Transformer (DiT) inpainting model.

Replaces FLUX.1-Fill as a lightweight diffusion-based inpainter (~0.6B
parameters in the DiT backbone vs. 12B in FLUX.1-Fill) that runs comfortably
on consumer GPUs and Apple Silicon (MPS).

Because the official PixArt-α release in `diffusers` (`PixArtAlphaPipeline`)
is a text-to-image model and is not registered in `AutoPipelineForInpainting`,
this module implements **Coarse-to-Fine Blended Latent Diffusion** (following
Avrahami et al., "Blended Latent Diffusion" / RePaint) on top of
`PixArt-alpha/PixArt-XL-2-512x512`:

1. **Coarse object removal (pre-fill)**: The masked region (`mask > 0`) is
   first pre-filled in pixel space using OpenCV Telea (`cv2.inpaint`) so the
   object to be erased does not leak through the VAE's 8x8 downsampling
   receptive field or leave ghost silhouettes.
2. **Memory-optimized T5 prompt caching**: Following the official Hugging Face
   low-VRAM guide for PixArt-α, the neutral background prompt is encoded once
   with `T5EncoderModel` and cached to `~/.cache/inpainting-bench/`. The 4.7B
   T5 text encoder is then immediately unloaded, leaving only the 0.6B
   `PixArtTransformer2DModel` and `AutoencoderKL` (~1.4 GB in float16) in
   memory during inference.
3. **Latent-space diffusion blending**: At each DDIM denoising step, the
   known background latents (`mask == 0`) are re-noised to the current
   timestep and blended with the denoised foreground latents (`mask > 0`),
   allowing DiT global self-attention to synthesize coherent background
   texture that seamlessly matches the surrounding scene.
4. **Pixel-space compositing**: The decoded result is resized back to the
   original image resolution and composited strictly inside `mask > 0`.
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
_PROMPT_CACHE_FILE = _CACHE_DIR / "pixart_alpha_prompt_embeds_v1.pt"

# Neutral background restoration prompts for object removal
_DEFAULT_PROMPT = (
    "clean empty background, natural scene texture, "
    "seamless continuation, high quality photo"
)
_DEFAULT_NEGATIVE_PROMPT = (
    "object, foreground object, person, text, watermark, "
    "logo, artifacts, blurry, distorted, unnatural seam"
)


def _patch_ssl_if_needed() -> None:
    """Use certifi CA bundle on macOS so Hugging Face downloads do not fail."""
    try:
        import certifi

        ssl._create_default_https_context = (
            lambda: ssl.create_default_context(cafile=certifi.where())
        )
    except ImportError:
        pass


class PixArtInpainter(BaseInpainter):
    """
    Diffusion Transformer inpainting model based on PixArt-α (0.6B).

    Implements the `BaseInpainter` contract:
        result = PixArtInpainter().inpaint(image_rgb_uint8, mask_uint8)
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str | torch.device | None = None,
        num_inference_steps: int = 20,
        strength: float = 0.80,
        guidance_scale: float = 3.5,
        seed: int = 42,
    ):
        """
        Args:
            model_id:            Hugging Face repo ID for PixArt-α weights.
            device:              Target torch device ('cuda', 'mps', or 'cpu').
                                 Auto-detected when None.
            num_inference_steps: Total DDIM schedule steps (default: 20).
            strength:            Fraction of diffusion schedule to run on the
                                 Telea-prefilled image (0 < strength <= 1.0).
            guidance_scale:      Classifier-free guidance scale (default: 3.5).
            seed:                Random seed for reproducible latent noise.
        """
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

        from diffusers import DDIMScheduler, PixArtAlphaPipeline

        # 1. Load or compute cached T5 prompt embeddings so we don't keep the
        #    heavy 4.7B T5 text encoder in memory during image inpainting.
        (
            self._prompt_embeds,
            self._prompt_attn_mask,
            self._neg_prompt_embeds,
            self._neg_prompt_attn_mask,
        ) = self._load_or_encode_prompt(model_id)

        # 2. Load only the 0.6B DiT transformer and VAE (~1.4 GB in fp16).
        self.pipe = PixArtAlphaPipeline.from_pretrained(
            model_id,
            text_encoder=None,
            tokenizer=None,
            torch_dtype=self.dtype,
        )
        # Single-step Markovian DDIM scheduler works best for RePaint-style
        # per-step latent blending.
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
        """
        Return cached T5 embeddings for the background restoration prompt,
        computing and saving them once on first use, then freeing T5.
        """
        if _PROMPT_CACHE_FILE.exists():
            try:
                cached = torch.load(
                    _PROMPT_CACHE_FILE, map_location="cpu", weights_only=True
                )
                if cached.get("model_id") == model_id:
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

        # Encode once with the text encoder only (transformer=None, vae=None)
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
                prompt=_DEFAULT_PROMPT,
                do_classifier_free_guidance=True,
                negative_prompt=_DEFAULT_NEGATIVE_PROMPT,
                num_images_per_prompt=1,
                device=self.device,
                clean_caption=False,
            )

        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        tmp_path = _PROMPT_CACHE_FILE.with_suffix(".part")
        torch.save(
            {
                "model_id": model_id,
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

    @property
    def name(self) -> str:
        return "PixArt-α"

    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> InpaintResult:
        """
        Remove the masked object from `image` and reconstruct the background.

        Args:
            image: RGB image, dtype uint8, shape (H, W, 3).
            mask:  Binary mask, dtype uint8, shape (H, W).
                   Pixels with value > 0 are the region to inpaint.

        Returns:
            InpaintResult containing the inpainted RGB image and inference time.
        """
        orig_h, orig_w = image.shape[:2]
        binary_mask = (mask > 0).astype(np.uint8)

        # Fast path when mask is completely empty
        if not np.any(binary_mask):
            return InpaintResult(
                image=image.copy(),
                inference_time=0.0,
                metadata={"device": str(self.device), "steps": 0},
            )

        # 1. Coarse pre-fill in pixel space using OpenCV Telea so the object
        #    inside the mask is erased before VAE encoding.
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        prefilled_bgr = cv2.inpaint(bgr, binary_mask, 5, cv2.INPAINT_TELEA)
        prefilled_rgb = cv2.cvtColor(prefilled_bgr, cv2.COLOR_BGR2RGB)

        # 2. Resize to PixArt-α native resolution (512x512)
        target_res = 512
        img_512 = np.array(
            Image.fromarray(prefilled_rgb).resize(
                (target_res, target_res), Image.LANCZOS
            )
        )
        mask_512 = np.array(
            Image.fromarray(binary_mask * 255).resize(
                (target_res, target_res), Image.NEAREST
            )
        )

        # Convert image to [-1, 1] tensor: (1, 3, 512, 512)
        img_tensor = (
            torch.from_numpy(img_512)
            .permute(2, 0, 1)
            .float()
            .unsqueeze(0)
            .to(device=self.device, dtype=self.dtype)
        )
        img_tensor = (img_tensor / 127.5) - 1.0

        # Build latent-space mask: (1, 1, 64, 64) with 1 = region to generate
        latent_size = target_res // self.pipe.vae_scale_factor
        mask_tensor = (
            torch.from_numpy((mask_512 > 0).astype(np.float32))
            .unsqueeze(0)
            .unsqueeze(0)
            .to(device=self.device, dtype=self.dtype)
        )
        latent_mask = F.interpolate(
            mask_tensor, size=(latent_size, latent_size), mode="nearest"
        )
        # Dilate by 1 latent cell and feather slightly to avoid boundary seams
        latent_mask = F.max_pool2d(
            latent_mask, kernel_size=3, stride=1, padding=1
        )
        latent_mask = F.avg_pool2d(
            latent_mask, kernel_size=3, stride=1, padding=1
        )

        start_time = time.perf_counter()
        with torch.no_grad():
            # Encode coarse-filled image into VAE latent space
            posterior = self.pipe.vae.encode(img_tensor).latent_dist
            init_latents = posterior.mean * self.pipe.vae.config.scaling_factor

            # Set up DDIM timesteps and slice according to `self.strength`
            self.pipe.scheduler.set_timesteps(
                self.num_inference_steps, device=self.device
            )
            timesteps = self.pipe.scheduler.timesteps
            init_step = min(
                int(len(timesteps) * (1.0 - self.strength)),
                len(timesteps) - 1,
            )
            active_timesteps = timesteps[init_step:]

            # Sample deterministic Gaussian noise
            generator = torch.Generator(device="cpu").manual_seed(self.seed)
            noise = torch.randn(
                init_latents.shape,
                generator=generator,
                dtype=self.dtype,
            ).to(self.device)

            # Noise the initial latents to the starting timestep
            latents = self.pipe.scheduler.add_noise(
                init_latents, noise, active_timesteps[:1]
            )

            # Prepare classifier-free guidance embeddings and micro-conditions
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

            # 3. RePaint / Blended Latent Diffusion loop
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

                # PixArt-α DiT outputs 8 channels (4 noise + 4 learned variance)
                if (
                    self.pipe.transformer.config.out_channels // 2
                    == latent_channels
                ):
                    noise_pred = noise_pred.chunk(2, dim=1)[0]

                latents = self.pipe.scheduler.step(
                    noise_pred, t, latents, return_dict=False
                )[0]

                # Blend known background latents with denoised hole latents
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

            # Decode latents back to pixel space
            decoded = self.pipe.vae.decode(
                latents / self.pipe.vae.config.scaling_factor,
                return_dict=False,
            )[0]

        inference_time = time.perf_counter() - start_time

        # 4. Postprocess: [-1, 1] -> uint8 RGB, resize to original (orig_w, orig_h)
        out_img = decoded.squeeze(0).permute(1, 2, 0).cpu().float().numpy()
        out_img = ((out_img + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
        out_img = np.array(
            Image.fromarray(out_img).resize((orig_w, orig_h), Image.LANCZOS)
        )

        # Blend strictly inside the masked region so background pixels stay untouched
        final_img = (
            image * (binary_mask[..., None] == 0)
            + out_img * (binary_mask[..., None] > 0)
        ).astype(np.uint8)

        return InpaintResult(
            image=final_img,
            inference_time=inference_time,
            metadata={
                "device": str(self.device),
                "model_id": self.model_id,
                "steps": len(active_timesteps),
            },
        )
