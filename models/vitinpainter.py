import os
import sys
import time
import numpy as np
import torch
from PIL import Image
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

# Add mat directory to sys.path so its internal absolute imports work
sys.path.append(os.path.join(os.path.dirname(__file__), 'mat'))

from .base import BaseInpainter, InpaintResult
from .mat.networks.mat import Generator

class ViTInpainter(BaseInpainter):
    """
    Transformer-based inpainting model using MAT (Mask-Aware Transformer).
    """

    def __init__(self, device=None):
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = device
                
        # 1. Download pretrained MAT weights
        # We use safetensors to avoid downloading and executing unverified pickle files.
        model_path = hf_hub_download(repo_id="Acly/MAT", filename="MAT_Places512_G_fp16.safetensors")
        
        # 2. Initialize generator
        self.generator = Generator(z_dim=512, c_dim=0, w_dim=512, img_resolution=512, img_channels=3).to(self.device)
        
        # 3. Load weights
        state_dict = load_file(model_path)
        # MAT state_dicts might need strict=False depending on if they include EMA buffers, but safetensors generally work directly
        self.generator.load_state_dict(state_dict, strict=False)
        self.generator.eval()

    @property
    def name(self) -> str:
        return "ViTInpainter"

    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> InpaintResult:
        # Preprocessing
        h, w = image.shape[:2]

        # MAT is trained at exactly 512×512. Any other resolution breaks the
        # pre-allocated noise_const buffers and transformer attention maps.
        # Resize to 512×512, inpaint, then resize the result back.
        TARGET = 512
        img_512 = np.array(Image.fromarray(image).resize((TARGET, TARGET), Image.LANCZOS))
        # Nearest-neighbour for the mask to preserve binary edges
        mask_512 = np.array(Image.fromarray(mask).resize((TARGET, TARGET), Image.NEAREST))

        # Normalize image to [-1, 1]
        img_tensor = torch.from_numpy(img_512).permute(2, 0, 1).float().unsqueeze(0).to(self.device)
        img_tensor = (img_tensor / 127.5) - 1.0

        # Prepare mask for MAT
        # MAT convention: 0 = hole/inpaint region, 1 = valid context to keep.
        # (Opposite of BaseInpainter convention where mask > 0 is the hole).
        mask_tensor = torch.from_numpy(mask_512).float().unsqueeze(0).unsqueeze(0).to(self.device)
        mask_tensor = (mask_tensor == 0).float()

        # Random latent code
        z = torch.randn(1, 512, device=self.device)
        c = torch.empty(1, 0, device=self.device)

        # Inference
        start_time = time.time()
        with torch.no_grad():
            output = self.generator(img_tensor, mask_tensor, z, c, truncation_psi=1, noise_mode='const')
        inference_time = time.time() - start_time

        # Postprocessing: decode from [-1, 1] back to uint8
        out_img = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
        out_img = (out_img + 1.0) * 127.5
        out_img = np.clip(out_img, 0, 255).astype(np.uint8)

        # Resize the inpainted result back to the original resolution
        out_img = np.array(Image.fromarray(out_img).resize((w, h), Image.LANCZOS))

        # Blend: keep original pixels where mask==0, use inpainted pixels where mask>0
        final_img = image * (mask[..., None] == 0) + out_img * (mask[..., None] > 0)

        return InpaintResult(
            image=final_img.astype(np.uint8),
            inference_time=inference_time,
            metadata={"device": str(self.device)}
        )
