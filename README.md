# bye-bye-AI

Image inpainting and object removal — CV 2026 course project.

We compare three pretrained inpainting methods on removing objects from photos:

| Method | Type | Code |
|---|---|---|
| **OpenCV Telea** | Classical (fast marching) | `models/opencv_telea.py` |
| **LaMa** (big-lama) | CNN with Fourier convolutions | `models/lama.py` |
| **ViTInpainter** (MAT) | Mask-Aware Transformer | `models/vitinpainter.py` |

FLUX.1-Fill (diffusion) was planned as a fourth method but was removed from scope. `models/flux_fill.py` is kept only as a placeholder.

No model is trained or fine-tuned; all methods use public pretrained weights. The project has two modes:

1. **Evaluation Mode**: a command-line benchmark on the CORNE-Val dataset (PSNR / SSIM / LPIPS / inference time under different masks).
2. **Application Mode**: a web app where you upload an image, draw a mask, and compare the outputs of all models side by side.

---

## Repository structure

```
bye-bye-AI/
├── models/                   # one class per method, all share BaseInpainter
│   ├── base.py               # BaseInpainter interface + InpaintResult
│   ├── opencv_telea.py       # classical baseline
│   ├── lama.py               # LaMa (CNN)
│   ├── vitinpainter.py       # MAT (Transformer)
│   ├── flux_fill.py          # placeholder, out of scope
│   └── mat/                  # vendored MAT network code (NVIDIA / Li et al.)
├── evaluation/
│   ├── benchmark.py          # Evaluation Mode CLI
│   ├── metrics.py            # PSNR / SSIM / LPIPS (masked region + full image)
│   ├── masks.py              # derived masks: dilated, bbox, ellipse
│   └── visualization.py      # comparison grids, failure-case figures, plots
├── data/
│   └── dataset.py            # downloads CORNE-Val and converts it to our layout
├── backend/                  # FastAPI server for the web app
│   ├── app.py
│   └── Dockerfile
├── frontend/                 # React + Vite web UI (served by nginx in Docker)
├── tests/                    # backend and ViTInpainter tests
├── docs/api_contract.md      # backend REST/WebSocket API
├── reports/                  # weekly progress reports (PDF)
├── docker-compose.yml
├── requirements.txt          # Python dependencies (minimum versions)
└── requirements-lock.txt     # exact versions used for the reported results
```

---

## Requirements

- **Python 3.12** (for the benchmark and local backend)
- **Node.js 20+** (only for running the frontend locally)
- **Docker Desktop** (only for the one-command demo)
- Disk space for the dataset (~550 MB) and model weights (a few hundred MB per model), plus the PyTorch install
- A GPU is optional. Models automatically use CUDA, then Apple Silicon (MPS), then CPU.

Model weights download automatically the first time each model is used:

| Weights | Source | Cached in |
|---|---|---|
| LaMa `big-lama.pt` (~200 MB) | GitHub release of the IOPaint project | `~/.cache/inpainting-bench/` |
| MAT `MAT_Places512_G_fp16.safetensors` | Hugging Face `Acly/MAT` | `~/.cache/huggingface/` |
| LPIPS VGG backbone (metrics only) | torchvision | `~/.cache/torch/` |

---

## Installation

```bash
git clone https://github.com/Kazualov/bye-bye-AI.git
cd bye-bye-AI
python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements-lock.txt
```

`requirements-lock.txt` pins the exact versions we used (recorded on macOS / Apple Silicon). If it fails to install on your platform, use `pip install -r requirements.txt` instead, which only sets minimum versions.

---

## Evaluation Mode (benchmark)

### 1. Download the dataset

```bash
python data/dataset.py
```

This downloads [CORNE-Val](https://huggingface.co/datasets/QinmingZhou/CORNE-Val) from Hugging Face and writes it to `data/corne-val/` (218 samples), one folder per sample:

```
data/corne-val/<sample_id>/
├── shot.*       # photo with the object
├── bg.*         # same scene without the object (ground truth)
├── mask_sam.*   # object-core mask
└── mask_eff.*   # effect-aware mask (object + shadow/reflection)
```

### 2. Run the benchmark

```bash
# All models
python -m evaluation.benchmark --dataset data/corne-val --output results/

# Specific model(s)
python -m evaluation.benchmark --dataset data/corne-val --output results/ LaMa
python -m evaluation.benchmark --dataset data/corne-val --output results/ LaMa "OpenCV Telea" ViTInpainter
```

Each model is evaluated on six mask experiments per sample:

| Experiment | Mask |
|---|---|
| `sam_native` | object-core mask as given |
| `sam_dilated10`, `sam_dilated20` | object-core mask grown by 10 / 20 px |
| `sam_bbox` | bounding box of the object-core mask |
| `sam_ellipse` | ellipse inscribed in that bounding box |
| `eff_native` | effect-aware mask as given |

Metrics are computed against `bg` on the masked region (`*_region`, the main numbers) and on the full image (`*_full`).

### 3. Outputs

```
results/<ModelName>/
├── metrics.csv          # one row per (sample, mask experiment)
├── summary.csv          # mean and std per mask experiment
├── best_per_sample.csv  # best mask experiment per sample (by full-image LPIPS)
├── mask_ranking.csv     # mask experiments ranked by mean full-image LPIPS
├── qualitative_best/    # comparison grid: shot | mask | GT | prediction
└── plots/               # bar charts for sam_native
```

### Reproducibility

- **Determinism.** Re-running on the same machine gives identical outputs. Telea and LaMa contain no randomness. ViTInpainter (MAT) is generative: it samples a latent vector and applies dropout even at inference. Both are seeded (`DEFAULT_SEED = 0` in `models/vitinpainter.py`). Pass `seed=None` to `inpaint` for random fills.
- **Hardware.** ViTInpainter's random draws happen on the device it runs on, so its outputs differ between CUDA, MPS and CPU (each is reproducible on its own). LaMa can also differ slightly across devices because floating-point kernels differ. Inference times depend entirely on the hardware. Report which device produced your numbers.
- **Versions.** The exact package versions are in `requirements-lock.txt`.
- **Dataset.** CORNE-Val is used only as a held-out evaluation set. There is no training, so there is no train/val/test split.

---

## Application Mode (web demo)

### Option 1: Docker (one command)

Requires Docker Desktop to be running.

```bash
docker compose up --build
```

Then open **http://localhost:5173**. Stop it with `Ctrl+C` or `docker compose down`.

- The first build takes several minutes.
- The first request downloads the model weights. They are stored in the `model-cache` Docker volume, so later starts are fast.
- Inside Docker the models run on **CPU only**, including on Macs with Apple Silicon, so ViT and LaMa are slower than when running locally.

### Option 2: Local development (two terminals)

Terminal 1, backend (after [Installation](#installation)):

```bash
source .venv/bin/activate
uvicorn backend.app:app --reload
```

Terminal 2, frontend:

```bash
cd frontend
npm install
npm run dev
```

Open **http://localhost:5173**. The Vite dev server forwards `/api` and `/files` to the backend on port 8000. The API is described in `docs/api_contract.md`.

---

## Using a model from Python

All models share one interface:

```python
from models import OpenCVTelea, LaMaInpainter, ViTInpainter

model = LaMaInpainter()                 # or OpenCVTelea(), ViTInpainter()
result = model.inpaint(image, mask)     # image: RGB uint8 (H, W, 3); mask: uint8 (H, W), >0 = fill
result.image                            # inpainted RGB uint8 (H, W, 3)
result.inference_time                   # seconds spent in the model only
```

---

## Tests

```bash
pytest tests/ -s
```

The tests run ViTInpainter on synthetic images (first run downloads the MAT weights) and test the backend API end to end.

---

## Team responsibilities

| Member | Scope |
|---|---|
| Ivan | LaMa, OpenCV Telea, shared interface, evaluation pipeline |
| Alexander | ViTInpainter / MAT (`models/vitinpainter.py`, `models/mat/`) |
| Egor | FLUX.1-Fill (removed from scope) |
| Tatiana | Backend (`backend/`), frontend (`frontend/`) |
