# bye-bye-AI

Image inpainting and object-removal system — CV 2026 course project.

Compares four methods: **OpenCV Telea** (classical), **LaMa** (CNN), **ViTInpainter** (Transformer), **FLUX.1-Fill** (Diffusion).

---

## Project structure

```
bye-bye-AI/
├── models/
│   ├── base.py           # shared BaseInpainter interface
│   ├── opencv_telea.py   # classical baseline (Ivan)
│   ├── lama.py           # CNN model (Ivan)
│   ├── vitinpainter.py   # Transformer stub (Alexander)
│   └── flux_fill.py      # Diffusion stub (Egor)
│
├── evaluation/
│   ├── benchmark.py      # CLI evaluation pipeline
│   ├── metrics.py        # PSNR / SSIM / LPIPS
│   ├── masks.py          # mask generation utilities
│   └── visualization.py  # plots and comparison grids
│
├── app/
│   ├── backend/          # FastAPI backend (Tatiana)
│   └── frontend/         # Web UI (Tatiana)
│
├── results/              # generated outputs (git-ignored)
├── requirements.txt
└── README.md
```

---

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python data/dataset.py
```

LaMa weights are downloaded automatically on first use.

---

## Evaluation Mode

Runs the benchmark against the CORNE-Val dataset and writes per-model
subdirectories under the output folder, each containing `metrics.csv`,
`summary.csv`, comparison grids, and plots.

```bash
# All models
python -m evaluation.benchmark \
    --dataset data/corne-val \
    --output  results/

# Specific model(s)
python -m evaluation.benchmark --dataset data/corne-val --output results/ LaMa
python -m evaluation.benchmark --dataset data/corne-val --output results/ LaMa "OpenCV Telea"
```

## Frontend mode
```bash

uvicorn backend.app:app --reload

```

### CORNE-Val directory layout

```
corne-val/
└── <sample_id>/
    ├── shot.jpg       # image with the object
    ├── bg.jpg         # clean background (ground truth)
    ├── mask_sam.png   # object-core mask
    └── mask_eff.png   # effect-aware mask
```

---

## Model interface

Every model exposes the same contract:

```python
from models import OpenCVTelea, LaMaInpainter

model = LaMaInpainter()  # or OpenCVTelea()
result = model.inpaint(image_rgb_uint8, mask_uint8)
# result.image           → inpainted RGB ndarray
# result.inference_time  → seconds (float)
```

---

## Team responsibilities

| Member | Scope |
|---|---|
| Ivan | LaMa, OpenCV Telea, shared interface, evaluation pipeline |
| Alexander | ViTInpainter (`models/vitinpainter.py`) |
| Egor | FLUX.1-Fill (`models/flux_fill.py`) |
| Tatiana | Frontend + Backend (`app/`) |

---

## 🚀 Quick Start

There are two ways to run the project: using **Docker** (recommended, one command) or running the **Local Development** servers separately.

### Option 1: Docker (One-Command Launch)

This is the easiest way to run the entire stack (Frontend, Backend, and Models) on any operating system.

**Prerequisites:** [Docker Desktop](https://www.docker.com/products/docker-desktop/) must be installed and running.

1. Clone the repository:
   ```bash
   git clone https://github.com/Kazualov/bye-bye-AI.git
   cd bye-bye-AI
   ```

2. Make sure **Docker Desktop** is running in the background.

3. Build and start the containers:
   ```bash
   docker compose up --build
   ```
   *(Note: The first build may take 5-10 minutes as Docker downloads base images and installs dependencies).*

4. Open the application in your browser:
   **http://localhost:5173**

5. **To stop the application:**
   Press `Ctrl + C` in the terminal, or run the following command in a new terminal tab:
   ```bash
   docker compose down
   ```

---

### Option 2: Local Development (Separate Terminals)

Use this method if you want to modify the code and see changes instantly (Hot Reload).

**Prerequisites:** Python 3.12 and Node.js 20+ must be installed.

#### Terminal 1: Backend
```bash
cd bye-bye-AI
python3.12 -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn backend.app:app --reload
```

#### Terminal 2: Frontend
```bash
cd frontend
npm install
npm run dev
```

**Access:**
*   Frontend: **http://localhost:5173**
*   Backend API: **http://127.0.0.1:8000**

> ⚠️ **Important Note:** On the very first request, the backend will automatically download the neural network weights (LaMa and ViT models). This may take a few minutes depending on your internet speed. Please wait until the models are downloaded and inference completes.
