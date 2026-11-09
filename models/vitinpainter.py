"""
ViTInpainter — Transformer-based inpainting using MAT (Mask-Aware Transformer).

Wraps the official MAT generator (Li et al., CVPR 2022), vendored in
`models/mat/`, with weights trained on Places365 at 512×512. The weights are
downloaded from the Hugging Face Hub (`Acly/MAT`) on first use and cached in
~/.cache/huggingface/.

MAT is a generative model: it takes a random latent vector `z`, so different
`z` values give different fills for the same input. For reproducible results
`inpaint` uses a fixed seed (DEFAULT_SEED) unless told otherwise.
"""

import os
import sys
import time

import cv2
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from safetensors.torch import load_file

# The vendored MAT code imports its helper packages as top-level modules
# (`import dnnlib`, `from torch_utils import ...`), so `models/mat` must be on
# sys.path before importing the network.
sys.path.append(os.path.join(os.path.dirname(__file__), "mat"))

from .base import BaseInpainter, InpaintResult  # noqa: E402
from .mat.networks.mat import Generator  # noqa: E402

# MAT is trained at exactly 512×512. Other resolutions break its pre-allocated
# noise buffers and attention windows, so every input is resized to this size.
MAT_RESOLUTION = 512

# Seed for the latent vector `z`. Fixed by default so that benchmark numbers
# and demo outputs are identical from run to run.
DEFAULT_SEED = 0


class ViTInpainter(BaseInpainter):
    """Transformer-based inpainting model using MAT (Mask-Aware Transformer)."""

    def __init__(self, device: torch.device | str | None = None):
        """
        Args:
            device: Torch device to run on. If None, picks CUDA, then Apple MPS,
                    then CPU.
        """
        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        self.device = torch.device(device)

        # 1. Download pretrained MAT weights. We use the safetensors version so
        #    that no pickle (arbitrary code) is executed when loading.
        model_path = hf_hub_download(repo_id="Acly/MAT", filename="MAT_Places512_G_fp16.safetensors")

        # 2. Build the generator with the architecture the weights were trained with.
        self.generator = Generator(
            z_dim=512, c_dim=0, w_dim=512,
            img_resolution=MAT_RESOLUTION, img_channels=3,
        ).to(self.device)

        # 3. Load weights. strict=False because this checkpoint omits the 22
        #    per-layer noise tensors (`noise_const`, `noise_strength`) of the
        #    decoder. `noise_strength` therefore stays at its initial value 0,
        #    so the random `noise_const` buffers have no effect on the output.
        state_dict = load_file(model_path)
        self.generator.load_state_dict(state_dict, strict=False)
        self.generator.eval()

    @property
    def name(self) -> str:
        return "ViTInpainter"

    def inpaint(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        truncation_psi: float = 0.7,
        auto_dilate: int = 5,
        feather_radius: float = 3.0,
        seed: int | None = DEFAULT_SEED,
    ) -> InpaintResult:
        """
        Fill the masked region of `image`.

        Args:
            image:          RGB image, uint8, shape (H, W, 3).
            mask:           Mask, uint8, shape (H, W); pixels > 0 are filled.
            truncation_psi: StyleGAN truncation. Lower = more conservative,
                            less varied fills; 1.0 = no truncation.
            auto_dilate:    Grow the mask by this many pixels before
                            inpainting so tight user masks don't leave object
                            edges behind. 0 disables it.
            feather_radius: Width (px) of the soft blend between original and
                            generated pixels at the mask border. 0 = hard paste.
            seed:           Seed for the latent vector. None = random (outputs
                            change between calls).

        Returns:
            InpaintResult with the inpainted image at the original resolution.
        """
        h, w = image.shape[:2]

        # ── mask preparation ─────────────────────────────────────────────────
        if auto_dilate > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * auto_dilate + 1, 2 * auto_dilate + 1))
            working_mask = cv2.dilate((mask > 0).astype(np.uint8) * 255, k)
        else:
            working_mask = (mask > 0).astype(np.uint8) * 255

        # ── resize to MAT's fixed input size ─────────────────────────────────
        img_512 = np.array(Image.fromarray(image).resize((MAT_RESOLUTION, MAT_RESOLUTION), Image.LANCZOS))
        # Bilinear resize + a low threshold (10/255) keeps thin mask parts from
        # disappearing when the image is downscaled: any pixel even partly
        # covered by the mask stays in the hole.
        mask_pil = Image.fromarray(working_mask).resize((MAT_RESOLUTION, MAT_RESOLUTION), Image.BILINEAR)
        mask_512 = (np.array(mask_pil) > 10).astype(np.uint8) * 255

        # Image to (1, 3, 512, 512) float in [-1, 1]
        img_tensor = torch.from_numpy(img_512).permute(2, 0, 1).float().unsqueeze(0).to(self.device)
        img_tensor = (img_tensor / 127.5) - 1.0

        # MAT's mask convention is the opposite of ours:
        # 1 = valid pixel to keep, 0 = hole to fill.
        mask_tensor = torch.from_numpy(mask_512).float().unsqueeze(0).unsqueeze(0).to(self.device)
        mask_tensor = (mask_tensor == 0).float()

        # ── randomness ───────────────────────────────────────────────────────
        # MAT has two sources of randomness: the latent vector z, and dropout
        # that the network applies *even at inference* (F.dropout(...,
        # training=True) in models/mat/networks/mat.py, a deliberate part of
        # MAT's design for diverse fills). Dropout uses torch's global RNG, so
        # we seed that too. Note: this resets the global torch seed.
        # z is sampled on the CPU so it is identical on every device; the
        # dropout masks are drawn on the model's device, so outputs are
        # reproducible per device type (CUDA / MPS / CPU), not across them.
        if seed is not None:
            torch.manual_seed(seed)
        gen = torch.Generator().manual_seed(seed) if seed is not None else None
        z = torch.randn(1, 512, generator=gen).to(self.device)
        c = torch.empty(1, 0, device=self.device)  # unconditional model: no class label

        # ── inference ────────────────────────────────────────────────────────
        start = time.perf_counter()
        with torch.no_grad():
            # noise_mode="const" uses fixed per-layer noise instead of fresh random noise
            output = self.generator(img_tensor, mask_tensor, z, c, truncation_psi=truncation_psi, noise_mode="const")
        inference_time = time.perf_counter() - start

        # ── postprocessing ───────────────────────────────────────────────────
        # Decode [-1, 1] back to uint8 and resize to the original resolution
        out_img = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
        out_img = np.clip((out_img + 1.0) * 127.5, 0, 255).astype(np.uint8)
        out_img = np.array(Image.fromarray(out_img).resize((w, h), Image.LANCZOS))

        # Paste the result back: pixels outside working_mask stay untouched;
        # inside, the weight ramps from 0 to 1 over `feather_radius` pixels.
        binary_mask = (working_mask > 0).astype(np.uint8)
        if feather_radius > 0:
            dist = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 3)
            weight = np.clip(dist / float(feather_radius), 0.0, 1.0)[..., None]
            final_img = (image.astype(np.float32) * (1.0 - weight) + out_img.astype(np.float32) * weight).round()
        else:
            final_img = np.where(binary_mask[..., None] > 0, out_img, image)

        return InpaintResult(
            image=final_img.astype(np.uint8),
            inference_time=inference_time,
            metadata={"device": str(self.device), "truncation_psi": truncation_psi, "seed": seed},
        )
