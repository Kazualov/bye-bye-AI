# API Contract

Version: 0.1
Date: 2026-10-06
Owner: Tatiana (Frontend / Backend)

## 1. What the application does

The user uploads an image and draws a mask (the region to be removed).
The backend sends the image and the mask to the inpainting models.
The models return their results.
The frontend displays a gallery of the results.

Diagram:

```
Frontend  →  Backend  →  Models (Telea, LaMa, ViT, FLUX)
   ↑            ↓
   └── results ──┘
```

## 2. Model IDs

Model IDs are identical on the frontend, the backend, and in the ML code.

| ID      | Model               | Type          |
|---------|---------------------|---------------|
| `telea` | OpenCV Telea        | classical     |
| `lama`  | LaMa                | CNN           |
| `vit`   | ViTInpainter        | Transformer   |
| `flux`  | FLUX.1-Fill         | diffusion     |

## 3. Endpoints

### GET /api/v1/health

Health check.

Response 200:
```json
{ "status": "ok", "version": "0.1.0" }
```

### GET /api/v1/models

List of available models.

Response 200:
```json
{
  "models": [
    { "id": "telea", "name": "OpenCV Telea", "type": "classical",   "available": true },
    { "id": "lama",  "name": "LaMa",          "type": "cnn",         "available": true },
    { "id": "vit",   "name": "ViTInpainter",  "type": "transformer", "available": true },
    { "id": "flux",  "name": "FLUX.1-Fill",   "type": "diffusion",   "available": true }
  ]
}
```

### POST /api/v1/jobs

Accept an image and a mask, create a job.

Content-Type: `multipart/form-data`

Fields:
- `image`  — file, PNG or JPG, up to 10 MB
- `mask`   — file, PNG, same size as image
- `models` — comma-separated string of model IDs, e.g. `"telea,lama,flux"`

Response 202:
```json
{ "job_id": "abc123", "status": "pending" }
```

### GET /api/v1/jobs/{job_id}

Get job status and results.

Response 200:
```json
{
  "job_id": "abc123",
  "status": "running",
  "created_at": "2026-10-06T12:00:00Z",
  "input_image_url": "/files/abc123/input.png",
  "mask_url": "/files/abc123/mask.png",
  "results": [
    {
      "model": "telea",
      "status": "succeeded",
      "output_url": "/files/abc123/telea.png",
      "runtime_ms": 42,
      "error": null
    },
    {
      "model": "lama",
      "status": "running",
      "output_url": null,
      "runtime_ms": null,
      "error": null
    },
    {
      "model": "flux",
      "status": "failed",
      "output_url": null,
      "runtime_ms": null,
      "error": "CUDA OOM"
    }
  ]
}
```

Response 404 (job_id not found):
```json
{ "error": "job_not_found", "message": "abc123" }
```

### GET /files/{job_id}/{filename}

Serves a stored file (input, mask, or model output).

Response 200: the file itself (image/png).

## 4. Image and mask format

### Image
- Format: PNG or JPG.
- Max size: 10 MB.
- Color: RGB.

### Mask
- Format: PNG.
- Size: exactly matches the image. If not — 400.
- Color: grayscale.
  - `0` (black) — keep the pixel as is.
  - `255` (white) — inpaint (remove and reconstruct).
- The frontend may use a brush, rectangle, or ellipse, but the backend always receives a PNG mask.

## 5. Job statuses

| Status      | Meaning                                            |
|-------------|----------------------------------------------------|
| `pending`   | job created, not started yet                       |
| `running`   | at least one model is running                      |
| `succeeded` | all models finished successfully                   |
| `partial`   | some models failed, some succeeded                 |
| `failed`    | all models failed                                  |

## 6. Model statuses

| Status      | Meaning                        |
|-------------|--------------------------------|
| `pending`   | queued                         |
| `running`   | in progress                    |
| `succeeded` | completed successfully         |
| `failed`    | failed (see `error` field)     |

## 7. Errors

| HTTP | error            | When                                          |
|------|------------------|-----------------------------------------------|
| 400  | `invalid_image`  | not an image / larger than 10 MB / cannot decode |
| 400  | `invalid_mask`   | not PNG / size does not match the image       |
| 400  | `invalid_model`  | unknown model ID                              |
| 404  | `job_not_found`  | no job with this job_id                       |
| 500  | `internal_error` | internal server error                         |

Error format:
```json
{ "error": "invalid_mask", "message": "mask size 512x512 != image size 640x480" }
```

## 8. Common model interface (backend ↔ ML)

The backend calls every model through a single interface:

```python
result = model.inpaint(image, mask)
```

- `image`: numpy array of shape H×W×3, dtype uint8, RGB.
- `mask`:  numpy array of shape H×W, dtype uint8. `0` = keep, `255` = remove.
- `result`: numpy array of shape H×W×3, dtype uint8, RGB.

The backend does not know model implementation details. It only calls `inpaint`.
The ML team is responsible for ensuring every model (`telea`, `lama`, `vit`, `flux`) implements this interface.

## 9. Logging

The backend writes structured logs to stdout:

```
[2026-10-06T12:00:00Z] job_created job_id=abc123 models=telea,lama
[2026-10-06T12:00:01Z] model_start job_id=abc123 model=telea
[2026-10-06T12:00:01Z] model_done  job_id=abc123 model=telea runtime_ms=42
[2026-10-06T12:00:05Z] model_failed job_id=abc123 model=flux error=CUDA OOM
[2026-10-06T12:00:05Z] job_done    job_id=abc123 status=partial
```

Required fields: `job_id`, `model`, `runtime_ms`, `error`, `status`.

## 10. Open questions(to be resolved on team meeting)

1. Is polling every 1.5 s enough, or do we need WebSocket / SSE for FLUX?
2. Do we store files locally in `storage/` or in S3-compatible storage?
3. Who writes the adapter layer between the backend and the models — backend or ML team?
4. Do we need job history across sessions, or is in-memory state enough?
5. Do we need authentication for the MVP? (Default: no.)

## 11. Out of scope for this contract

- PSNR / SSIM / LPIPS metrics — computed in benchmark mode, not in the UI.
- Model training — ML team, separate.
- Test mask generation — separate `masks/` module.