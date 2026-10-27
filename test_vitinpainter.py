import sys
import os

# Add the project root to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.vitinpainter import ViTInpainter
import numpy as np

def main():
    print("Initializing ViTInpainter...")
    inpainter = ViTInpainter()
    
    print("Creating test image and mask...")
    # Create an image with varying pixel values (gradient + pattern)
    np.random.seed(42)
    image = np.random.randint(50, 200, size=(256, 256, 3), dtype=np.uint8)
    
    # Create a mask (1 for hole to inpaint, 0 for background)
    mask = np.zeros((256, 256), dtype=np.uint8)
    mask[80:160, 80:160] = 255
    
    print("Running inpaint...")
    result = inpainter.inpaint(image, mask)
    
    print(f"Result shape: {result.image.shape}")
    print(f"Inference time: {result.inference_time:.2f}s")
    print(f"Device: {result.metadata.get('device', 'unknown')}")
    
    # Check that output matches expected shape and dtype
    assert result.image.shape == image.shape, f"Shape mismatch: {result.image.shape} vs {image.shape}"
    assert result.image.dtype == np.uint8, f"Dtype mismatch: {result.image.dtype}"
    
    # With default auto_dilate=5, unmasked pixels outside dilation boundary are preserved exactly
    import cv2
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    dilated_mask = cv2.dilate((mask > 0).astype(np.uint8) * 255, k5)
    np.testing.assert_array_equal(
        result.image[dilated_mask == 0],
        image[dilated_mask == 0],
        err_msg="Pixels outside dilation boundary were modified by inpainting!"
    )

    # When auto_dilate=0, original unmasked pixels must be preserved exactly
    result_exact = inpainter.inpaint(image, mask, auto_dilate=0)
    np.testing.assert_array_equal(
        result_exact.image[mask == 0],
        image[mask == 0],
        err_msg="Unmasked pixels were modified when auto_dilate=0!"
    )
    
    # Masked region must be modified by the inpainter (not an exact copy)
    mask_diff = np.abs(result.image[mask > 0].astype(int) - image[mask > 0].astype(int))
    assert mask_diff.max() > 0, "Inpainted region is an exact copy of the input image!"
    print(f"Masked region mean absolute pixel change: {mask_diff.mean():.2f}")
    print("All assertions passed successfully!")

if __name__ == "__main__":
    main()
