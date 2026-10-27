import io
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

import numpy as np
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

APP_VERSION = "0.1.0"
STORAGE = Path("storage")
STORAGE.mkdir(exist_ok=True)

MODELS = [
    {"id": "telea", "name": "OpenCV Telea", "type": "classical",   "available": True},
    {"id": "lama",  "name": "LaMa",          "type": "cnn",         "available": True},
    {"id": "vit",   "name": "ViTInpainter",  "type": "transformer", "available": True},
    {"id": "flux",  "name": "FLUX.1-Fill",   "type": "diffusion",   "available": True},
]

# Models to load in the background at server start (so the first user
# request doesn't wait for the weights download).
WARMUP_MODELS = ("lama",)

_MODEL_CACHE = {}
_LOAD_LOCK = threading.Lock()   # warmup and first request must not load a model twice
_RUN_LOCK = threading.Lock()    # one GPU: inference runs one at a time


def get_model(mid: str):
    """Lazy-load and cache inpainter instances."""
    with _LOAD_LOCK:
        if mid not in _MODEL_CACHE:
            if mid == "vit":
                from models.vitinpainter import ViTInpainter
                _MODEL_CACHE[mid] = ViTInpainter()
            elif mid == "telea":
                from models.opencv_telea import OpenCVTelea
                _MODEL_CACHE[mid] = OpenCVTelea()
            elif mid == "lama":
                from models.lama import LaMaInpainter
                _MODEL_CACHE[mid] = LaMaInpainter()
            elif mid == "flux":
                from models.flux_fill import FluxFillInpainter
                _MODEL_CACHE[mid] = FluxFillInpainter()
            else:
                raise ValueError(f"Unknown model ID: {mid}")
        return _MODEL_CACHE[mid]


def _free_gpu() -> None:
    """After a CUDA OOM the next model in the same job shouldn't fail too."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


JOBS = {}


def log(event: str, **fields) -> None:
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    kv = " ".join(f"{k}={v}" for k, v in fields.items())
    print(f"[{ts}] {event} {kv}", flush=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    def warmup():
        for mid in WARMUP_MODELS:
            try:
                get_model(mid)
                log("model_ready", model=mid)
            except Exception as e:
                log("model_load_failed", model=mid, error=str(e))

    threading.Thread(target=warmup, daemon=True).start()   # server starts immediately
    yield


app = FastAPI(title="Inpainting API", version=APP_VERSION, lifespan=lifespan)


@app.get("/api/v1/health")
def health():
    return {"status": "ok", "version": APP_VERSION}


@app.get("/api/v1/models")
def list_models():
    return {"models": MODELS}


def _run_models(job_id: str, image_path: str, mask_path: str, model_ids: List[str]) -> None:
    job = JOBS[job_id]
    job["status"] = "running"

    img_arr = np.array(Image.open(image_path).convert("RGB"))
    msk_arr = np.array(Image.open(mask_path).convert("L"))

    for mid in model_ids:
        entry = next(r for r in job["results"] if r["model"] == mid)
        entry["status"] = "running"
        log("model_start", job_id=job_id, model=mid)

        t0 = time.time()
        try:
            model = get_model(mid)
            with _RUN_LOCK:
                result = model.inpaint(img_arr, msk_arr)
            out_name = f"{mid}.png"
            out_path = STORAGE / job_id / out_name
            Image.fromarray(result.image).save(out_path)
            entry["status"] = "succeeded"
            entry["output_url"] = f"/files/{job_id}/{out_name}"
            runtime_sec = getattr(result, "inference_time", time.time() - t0)
            entry["runtime_ms"] = int(runtime_sec * 1000)
            log("model_done", job_id=job_id, model=mid, runtime_ms=entry["runtime_ms"])
        except Exception as exc:
            _free_gpu()
            msg = str(exc) or type(exc).__name__
            entry["status"] = "failed"
            entry["error"] = msg
            log("model_failed", job_id=job_id, model=mid, error=msg)

    statuses = [r["status"] for r in job["results"]]
    if all(s == "succeeded" for s in statuses):
        job["status"] = "succeeded"
    elif all(s == "failed" for s in statuses):
        job["status"] = "failed"
    else:
        job["status"] = "partial"
    log("job_done", job_id=job_id, status=job["status"])


@app.post("/api/v1/jobs", status_code=202)
async def create_job(
        background: BackgroundTasks,
        image: UploadFile = File(...),
        mask: UploadFile = File(...),
        models: str = Form(...),
):
    img_bytes = await image.read()
    mask_bytes = await mask.read()

    if len(img_bytes) > 10 * 1024 * 1024:
        raise HTTPException(400, detail={"error": "invalid_image", "message": "image > 10 MB"})
    if len(mask_bytes) > 10 * 1024 * 1024:
        raise HTTPException(400, detail={"error": "invalid_mask", "message": "mask > 10 MB"})

    try:
        img = Image.open(io.BytesIO(img_bytes))
        img.load()
        msk = Image.open(io.BytesIO(mask_bytes))
        msk.load()
    except Exception:
        raise HTTPException(400, detail={"error": "invalid_image", "message": "cannot decode"})

    if img.size != msk.size:
        raise HTTPException(400, detail={
            "error": "invalid_mask",
            "message": f"mask size {msk.size} != image size {img.size}",
        })

    msk_l = msk.convert("L")
    if not np.array(msk_l).any():
        raise HTTPException(400, detail={
            "error": "invalid_mask",
            "message": "mask is empty (nothing to inpaint)",
        })

    requested = [m.strip() for m in models.split(",") if m.strip()]
    if not requested:
        raise HTTPException(400, detail={"error": "invalid_model", "message": "no models selected"})

    valid_ids = {m["id"] for m in MODELS}
    for mid in requested:
        if mid not in valid_ids:
            raise HTTPException(400, detail={"error": "invalid_model", "message": mid})

    job_id = uuid.uuid4().hex[:12]
    job_dir = STORAGE / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    image_path = job_dir / "input.png"
    mask_path = job_dir / "mask.png"
    img.convert("RGB").save(image_path)
    msk_l.save(mask_path)

    JOBS[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "input_image_url": f"/files/{job_id}/input.png",
        "mask_url": f"/files/{job_id}/mask.png",
        "results": [
            {
                "model": mid,
                "status": "pending",
                "output_url": None,
                "runtime_ms": None,
                "error": None,
            }
            for mid in requested
        ],
    }
    log("job_created", job_id=job_id, models=",".join(requested))

    background.add_task(_run_models, job_id, str(image_path), str(mask_path), requested)
    return {"job_id": job_id, "status": "pending"}


@app.get("/api/v1/jobs/{job_id}")
def get_job(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, detail={"error": "job_not_found", "message": job_id})
    return job


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    if isinstance(exc.detail, dict):
        body = exc.detail
    else:
        body = {"error": "http_error", "message": str(exc.detail)}
    return JSONResponse(status_code=exc.status_code, content=body)


# Serve generated files (input, mask, outputs)
app.mount("/files", StaticFiles(directory=str(STORAGE)), name="files")

# Serve the frontend if the folder exists (added later, not yet)
FRONTEND_DIR = Path("frontend")
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")