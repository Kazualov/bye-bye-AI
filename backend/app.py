import io
import shutil
import time
import uuid
from pathlib import Path
from typing import List
import asyncio
import numpy as np


from fastapi import (BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile,
                     WebSocket, WebSocketDisconnect)
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from models.lama import LaMaInpainter
from models.opencv_telea import OpenCVTelea
from models.vitinpainter import ViTInpainter

APP_VERSION = "0.1.0"
STORAGE = Path("storage")
STORAGE.mkdir(exist_ok=True)

MODELS = [
    {"id": "telea", "name": "OpenCV Telea", "type": "classical",   "available": True},
    {"id": "lama",  "name": "LaMa",          "type": "cnn",         "available": True},
    {"id": "vit",   "name": "ViTInpainter",  "type": "transformer", "available": True},
    {"id": "flux",  "name": "FLUX.1-Fill",   "type": "diffusion",   "available": True},
]

# Mock inference times per model (seconds). Real models will replace this.
MOCK_DELAYS = {"telea": 0.2, "lama": 1.0, "vit": 1.5, "flux": 3.0}

JOBS = {}

app = FastAPI(title="Inpainting API", version=APP_VERSION)

print("Loading models (this may take a while on first run)...")
try:
    telea_model = OpenCVTelea()
    print("✓ Telea loaded")
except Exception as e:
    telea_model = None
    print(f"✗ Telea failed to load: {e}")

try:
    lama_model = LaMaInpainter()
    print("✓ LaMa loaded")
except Exception as e:
    lama_model = None
    print(f"✗ LaMa failed to load: {e}")

try:
    vit_model = ViTInpainter()
    print("✓ ViT loaded")
except Exception as e:
    vit_model = None
    print(f"✗ ViT failed to load: {e}")

class ConnectionManager:
    def __init__(self):
        self.active_connections: dict[str, list[WebSocket]] = {}

    async def connect(self, job_id: str, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.setdefault(job_id, []).append(websocket)

    def disconnect(self, job_id: str, websocket: WebSocket):
        if job_id in self.active_connections:
            self.active_connections[job_id].remove(websocket)
            if not self.active_connections[job_id]:
                del self.active_connections[job_id]

    async def broadcast(self, job_id: str, message: dict):
        if job_id in self.active_connections:
            for connection in self.active_connections[job_id]:
                try:
                    await connection.send_json(message)
                except Exception:
                    pass

manager = ConnectionManager()


def log(event: str, **fields) -> None:
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    kv = " ".join(f"{k}={v}" for k, v in fields.items())
    print(f"[{ts}] {event} {kv}", flush=True)


@app.get("/api/v1/health")
def health():
    return {"status": "ok", "version": APP_VERSION}


@app.get("/api/v1/models")
def list_models():
    return {"models": MODELS}


async def _run_models(job_id: str, image_path: str, mask_path: str, model_ids: List[str]) -> None:
    job = JOBS[job_id]
    job["status"] = "running"
    await manager.broadcast(job_id, job)

    try:
        img_np = np.array(Image.open(image_path).convert("RGB"))
        mask_np = np.array(Image.open(mask_path).convert("L"))
    except Exception as e:
        log("job_failed", job_id=job_id, error=f"Failed to load image/mask: {e}")
        job["status"] = "failed"
        await manager.broadcast(job_id, job)
        return

    for mid in model_ids:
        entry = next(r for r in job["results"] if r["model"] == mid)
        entry["status"] = "running"
        await manager.broadcast(job_id, job)
        log("model_start", job_id=job_id, model=mid)

        try:
            out_name = f"{mid}.png"
            out_path = STORAGE / job_id / out_name

            if mid == "telea":
                if not telea_model: raise Exception("Telea model not loaded")
                result = telea_model.inpaint(img_np, mask_np)
                Image.fromarray(result.image).save(out_path)
                entry["runtime_ms"] = int(result.inference_time * 1000)

            elif mid == "lama":
                if not lama_model: raise Exception("LaMa model not loaded")
                result = await asyncio.to_thread(lama_model.inpaint, img_np, mask_np)
                Image.fromarray(result.image).save(out_path)
                entry["runtime_ms"] = int(result.inference_time * 1000)

            elif mid == "vit":
                if not vit_model: raise Exception("ViT model not loaded")
                result = await asyncio.to_thread(vit_model.inpaint, img_np, mask_np)
                Image.fromarray(result.image).save(out_path)
                entry["runtime_ms"] = int(result.inference_time * 1000)

            elif mid == "flux":
                await asyncio.sleep(1.0)
                shutil.copy(image_path, out_path)
                entry["runtime_ms"] = 1000

            else:
                raise Exception(f"Unknown model ID: {mid}")

            entry["status"] = "succeeded"
            entry["output_url"] = f"/files/{job_id}/{out_name}"
            log("model_done", job_id=job_id, model=mid, runtime_ms=entry["runtime_ms"])

        except Exception as exc:
            entry["status"] = "failed"
            entry["error"] = str(exc)
            log("model_failed", job_id=job_id, model=mid, error=str(exc))

        await manager.broadcast(job_id, job)

    statuses = [r["status"] for r in job["results"]]
    if all(s == "succeeded" for s in statuses):
        job["status"] = "succeeded"
    elif all(s == "failed" for s in statuses):
        job["status"] = "failed"
    else:
        job["status"] = "partial"

    log("job_done", job_id=job_id, status=job["status"])
    await manager.broadcast(job_id, job)


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
    img.convert("RGB").save(image_path)
    msk = msk.convert("L").point(lambda p: 255 if p > 127 else 0)
    msk.save(mask_path)

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


@app.websocket("/api/v1/ws/jobs/{job_id}")
async def websocket_endpoint(websocket: WebSocket, job_id: str):
    if job_id not in JOBS:
        await websocket.close(code=1008)
        return

    await manager.connect(job_id, websocket)
    try:
        # Сразу отправляем текущий статус задачи
        await websocket.send_json(JOBS[job_id])
        # Держим соединение открытым, пока клиент не отключится
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(job_id, websocket)

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