import io
import sys
from pathlib import Path
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

# Add project root to sys.path
root = Path(__file__).resolve().parent.parent
sys.path.append(str(root))

from backend.app import app

client = TestClient(app)

def make_test_png(w=256, h=256, color=(120, 150, 180)):
    arr = np.full((h, w, 3), color, dtype=np.uint8)
    # Add some pattern so it's not a flat image
    arr[20:60, 20:60] = [200, 100, 50]
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue(), arr

def make_test_mask(w=256, h=256, hole=(80, 80, 160, 160)):
    arr = np.zeros((h, w), dtype=np.uint8)
    y0, x0, y1, x1 = hole
    arr[y0:y1, x0:x1] = 255
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue(), arr

def test_health():
    res = client.get("/api/v1/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"
    print("✓ test_health passed")

def test_models():
    res = client.get("/api/v1/models")
    assert res.status_code == 200
    model_ids = [m["id"] for m in res.json()["models"]]
    assert "vit" in model_ids
    assert "lama" in model_ids
    assert "telea" in model_ids
    assert "flux" in model_ids
    print("✓ test_models passed")

def test_validation():
    img_bytes, _ = make_test_png(100, 100)
    msk_bytes, _ = make_test_mask(50, 50)  # Mismatched size!
    
    # 1. Size mismatch
    res = client.post(
        "/api/v1/jobs",
        files={"image": ("input.png", img_bytes, "image/png"), "mask": ("mask.png", msk_bytes, "image/png")},
        data={"models": "vit"},
    )
    assert res.status_code == 400
    assert res.json()["error"] == "invalid_mask"

    # 2. Unknown model
    msk_bytes_ok, _ = make_test_mask(100, 100)
    res = client.post(
        "/api/v1/jobs",
        files={"image": ("input.png", img_bytes, "image/png"), "mask": ("mask.png", msk_bytes_ok, "image/png")},
        data={"models": "unknown_model"},
    )
    assert res.status_code == 400
    assert res.json()["error"] == "invalid_model"
    print("✓ test_validation passed")

def test_vit_job_execution():
    img_bytes, img_arr = make_test_png(256, 256)
    msk_bytes, msk_arr = make_test_mask(256, 256)

    # Submit job
    res = client.post(
        "/api/v1/jobs",
        files={"image": ("input.png", img_bytes, "image/png"), "mask": ("mask.png", msk_bytes, "image/png")},
        data={"models": "vit"},
    )
    assert res.status_code == 202
    job_id = res.json()["job_id"]
    print(f"Created job {job_id}")

    # In TestClient, BackgroundTasks are executed synchronously during the request
    res = client.get(f"/api/v1/jobs/{job_id}")
    assert res.status_code == 200
    job = res.json()
    assert job["status"] == "succeeded"

    vit_res = next(r for r in job["results"] if r["model"] == "vit")
    assert vit_res["status"] == "succeeded"
    assert vit_res["error"] is None
    assert vit_res["runtime_ms"] > 0
    assert vit_res["output_url"] == f"/files/{job_id}/vit.png"
    print(f"ViT completed in {vit_res['runtime_ms']} ms")

    # Fetch output file
    file_res = client.get(vit_res["output_url"])
    assert file_res.status_code == 200
    out_img = Image.open(io.BytesIO(file_res.content))
    out_arr = np.array(out_img)
    assert out_arr.shape == (256, 256, 3)

    # Check that inpainting modified the masked area
    mask_diff = np.abs(out_arr[msk_arr > 0].astype(int) - img_arr[msk_arr > 0].astype(int))
    assert mask_diff.max() > 0, "Masked region was not inpainted!"
    print(f"Masked region mean absolute pixel change: {mask_diff.mean():.2f}")
    print("✓ test_vit_job_execution passed")

def test_multi_model_execution():
    img_bytes, _ = make_test_png(128, 128)
    msk_bytes, _ = make_test_mask(128, 128)

    res = client.post(
        "/api/v1/jobs",
        files={"image": ("input.png", img_bytes, "image/png"), "mask": ("mask.png", msk_bytes, "image/png")},
        data={"models": "telea,vit,flux"},
    )
    assert res.status_code == 202
    job_id = res.json()["job_id"]

    res = client.get(f"/api/v1/jobs/{job_id}")
    assert res.status_code == 200
    job = res.json()
    # telea and vit succeeded, flux failed (not implemented) -> overall job is partial
    assert job["status"] == "partial"
    
    telea_res = next(r for r in job["results"] if r["model"] == "telea")
    vit_res = next(r for r in job["results"] if r["model"] == "vit")
    flux_res = next(r for r in job["results"] if r["model"] == "flux")

    assert telea_res["status"] == "succeeded"
    assert vit_res["status"] == "succeeded"
    assert flux_res["status"] == "failed"
    assert "not yet implemented" in flux_res["error"]
    print("✓ test_multi_model_execution passed (telea + vit succeeded, flux failed gracefully)")

if __name__ == "__main__":
    print("--- Running Backend Integration Tests ---")
    test_health()
    test_models()
    test_validation()
    test_vit_job_execution()
    test_multi_model_execution()
    print("--- ALL TESTS PASSED SUCCESSFULLY ---")
