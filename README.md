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

Runs the full benchmark against the CORNE-Val dataset and writes
`results/metrics.csv`, `results/summary.csv`, plots, and comparison grids.

```bash
python -m evaluation.benchmark \
    --dataset data/corne-val \
    --output  results/
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
