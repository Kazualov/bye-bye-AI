# Image Inpainting and Object Removal — Project Context

## 1. Project overview

We are building an image inpainting and object-removal system for the Computer Vision 2026 course.

The task is:

> Build an application that removes masked objects or regions and fills the missing area. Evaluate several mask sizes/shapes using quantitative and perceptual measures and document typical semantic failures.

The project has two separate modes:

1. **Evaluation Mode** — a console application used only for reproducible quantitative benchmarking.
2. **Application Mode** — an interactive web application where a user uploads any image, draws/selects a mask, and compares the four model outputs in a carousel.

We are **not training or fine-tuning the models from scratch** for the main project. All ML methods use pretrained weights. The contribution is the controlled comparison, evaluation, application pipeline, and failure analysis.

---

## 2. Research question

### Main question

> **How do classical, CNN-based, Transformer-based, and diffusion-based inpainting methods compare in reconstruction quality and inference speed under different mask conditions?**

The project compares four approaches:

| Approach | Type | Role |
|---|---|---|
| OpenCV Telea | Classical | Baseline |
| LaMa | CNN | Learned inpainting |
| ViTInpainter | Transformer | Transformer-based comparison |
| FLUX.1-Fill | Diffusion | Generative comparison |

The same input image and mask should be passed to all four approaches whenever the model interface allows it.

---

## 3. Dataset

### Primary evaluation dataset: CORNE-Val

CORNE-Val is the fixed held-out benchmark used for the quantitative evaluation.

Each sample provides the logical components:

- `shot` — image containing the object to remove;
- `bg` — object-free background / ground truth;
- `mask_sam` — object-core mask;
- `mask_eff` — effect-aware mask covering the object and associated visual effects.

The basic evaluation task is:

```text
shot + mask
    ↓
model
    ↓
prediction
    ↓
compare with bg
```

This is important: `bg` is the clean target, so we can evaluate actual object removal rather than simply comparing against an image that still contains the removed object.

### No custom train/validation/test split

Because the project uses pretrained models and performs no model training/fine-tuning in the main experiment, CORNE-Val is treated as a fixed **held-out evaluation benchmark** rather than a conventional training dataset.

Do not introduce a fake train split unless the project scope changes to include fine-tuning.

---

## 4. Mask experiments

The main native mask comparison is:

### A. Object-core mask

`mask_sam` covers the main object region.

### B. Effect-aware mask

`mask_eff` additionally covers effects associated with the object, such as shadows or other visible remnants.

### C. Derived sensitivity masks

We may generate additional controlled masks from the object mask:

- exact/native object mask;
- dilated mask (different dilation sizes);
- bounding-box mask;
- potentially simple geometric masks such as ellipse/rectangle when useful for the course requirement about mask shapes.

The goal is to measure how sensitive each inpainting method is to the precision, size, and shape of the supplied mask.

Important: all models receive a mask as an input. The project is **not an object-detection project**; the system is not primarily responsible for discovering the object automatically.

---

## 5. Evaluation metrics

The primary quantitative metrics are:

### PSNR ↑

Pixel-level reconstruction similarity.

### SSIM ↑

Structural similarity between prediction and clean background.

### LPIPS ↓

Perceptual similarity using learned visual features.

### Inference time ↓

Time required to produce one result under the fixed evaluation hardware and image-resolution setup.

Whenever appropriate, metrics should focus on the masked/affected region so that the large unchanged part of the image does not dominate the score.

A useful secondary check is full-image perceptual similarity, but the main analysis should be connected to the edited region.

---

## 6. Failure analysis

The project must document representative semantic/visual failures. Examples to investigate:

1. **Residual object content** — part of the object remains near the mask boundary.
2. **Shadow/reflection residue** — the object disappears but its shadow/reflection remains.
3. **Structural artifacts** — geometry, textures, lines, or repeated patterns become inconsistent.
4. **Large/complex masked regions** — quality degrades when too much information must be reconstructed.
5. **Generative hallucination** — a generative model creates plausible but incorrect content.

At least three clear failure cases should be shown and explained rather than just displayed.

---

## 7. Evaluation Mode

Evaluation Mode is a **console application**, not part of the user-facing web UI.

### Intended workflow

```text
Load CORNE-Val
      ↓
Load shot + mask + bg
      ↓
Run OpenCV Telea
Run LaMa
Run ViTInpainter
Run FLUX.1-Fill
      ↓
Save predictions
      ↓
Calculate PSNR / SSIM / LPIPS / inference time
      ↓
Aggregate results
      ↓
Save CSV + figures + failure examples
```

### Expected outputs

```text
results/
├── metrics.csv
├── summary.csv
├── qualitative_comparison/
├── failure_cases/
└── plots/
```

Potential CLI structure:

```bash
python benchmark.py \
    --dataset path/to/corne-val \
    --output results/
```

The exact CLI can be changed during implementation; keep it simple and reproducible.

---

## 8. Application Mode

Application Mode is the interactive web application.

### User flow

```text
Upload arbitrary photo
        ↓
Draw/select mask
        ↓
Click Remove
        ↓
Send image + mask to backend
        ↓
Run four models
        ↓
Display results progressively in carousel
```

### Carousel behavior

The UI should use a **carousel**, not a static gallery.

There are four carousel positions, one per method:

```text
[ OpenCV ] [ LaMa ] [ ViTInpainter ] [ FLUX ]
```

When a model has not finished yet, its carousel item shows a loading placeholder.

When that model finishes, the placeholder is replaced with the generated image **without waiting for the other models**.

Example:

```text
OpenCV      -> ready image
LaMa        -> ready image
ViTInpainter-> loading placeholder
FLUX        -> loading placeholder
```

Later:

```text
OpenCV      -> ready image
LaMa        -> ready image
ViTInpainter-> ready image
FLUX        -> ready image
```

### Important distinction

The Application Mode does **not** calculate PSNR/SSIM/LPIPS because arbitrary user images do not have ground-truth clean backgrounds.

Application Mode is for qualitative comparison and demo.

Evaluation Mode is for quantitative comparison.

The same model implementations must be shared between the two modes so that benchmark results and demo behavior use the same inference code.

---

## 9. Model responsibilities

### Ivan — LaMa + team lead

Primary responsibilities:

- implement and validate LaMa inference;
- implement OpenCV Telea baseline;
- maintain the shared model interface;
- coordinate the evaluation pipeline;
- integrate results from all ML models;
- coordinate experiments and final benchmark;
- aggregate results and help prepare the final report/demo.

### Alexander — ViTInpainter

Primary responsibilities:

- implement ViTInpainter inference;
- handle preprocessing/postprocessing;
- validate output quality and image/mask compatibility;
- measure runtime and memory behavior;
- integrate the model into the shared inference interface.

### Egor — FLUX.1-Fill

Primary responsibilities:

- implement FLUX.1-Fill inference;
- configure image/mask preprocessing and inference parameters;
- investigate VRAM usage and runtime;
- integrate the model into the shared inference interface;
- analyze semantic/generative failures.

### Tatiana — Frontend / Backend

Primary responsibilities:

- implement the web application UI;
- implement image upload and mask interaction;
- implement the four-slot carousel;
- implement loading placeholders and progressive result replacement;
- connect the frontend to the backend API;
- connect the backend to the shared model inference interface.

No detailed frontend implementation plan is required in the project report; the ML team provides a stable inference API/schema to Tatiana.

---

## 10. Shared ML interface

All four approaches should expose the same conceptual interface:

```python
result = model.inpaint(image, mask)
```

The exact class/function name can change, but the contract should remain stable:

**Input**

- RGB image;
- binary mask.

**Output**

- inpainted RGB image;
- optionally inference time / model metadata if the caller needs it.

The frontend/backend should not know model-specific preprocessing details.

---

## 11. Suggested repository structure

```text
project/
├── models/
│   ├── base.py
│   ├── opencv_telea.py
│   ├── lama.py
│   ├── vitinpainter.py
│   └── flux_fill.py
│
├── evaluation/
│   ├── benchmark.py
│   ├── metrics.py
│   ├── masks.py
│   └── visualization.py
│
├── app/
│   ├── backend/
│   └── frontend/
│
├── configs/
├── results/
├── scripts/
├── requirements.txt
└── README.md
```

Keep the model code and benchmark code independent from the web UI.

---

## 12. Python/runtime constraint

Use **Python 3.12** as the project runtime target.

The goal is one coherent modern environment rather than separate legacy environments for individual models.

Before full integration, each ML model must pass a small compatibility spike in the Python 3.12 environment:

1. import dependencies;
2. load pretrained weights;
3. run one image + mask;
4. save one output;
5. record approximate VRAM and inference time.

Do not spend excessive project time porting an incompatible legacy implementation. If a model cannot reasonably run in the common environment, flag it early and discuss a replacement before implementing the full benchmark around it.

---

## 13. First implementation priority

The first working end-to-end milestone should be:

```text
one input image
+
one mask
        ↓
OpenCV Telea
        ↓
LaMa
        ↓
ViTInpainter
        ↓
FLUX.1-Fill
        ↓
four saved outputs
```

After that:

1. unify model interfaces;
2. build CORNE-Val benchmark;
3. implement metrics;
4. implement mask sensitivity experiments;
5. build application mode;
6. add failure-case analysis;
7. finalize tables/figures/report.

---

## 14. Things we are deliberately NOT doing

- We are not training the large pretrained models from scratch.
- We are not making automatic object detection the main task.
- We are not using the original object-containing image as the ground truth for object removal.
- We are not calculating objective metrics for arbitrary user-uploaded photos.
- We are not making the frontend responsible for model-specific inference logic.
- We are not requiring all four models to finish before displaying the first completed carousel result.

---

## 15. Final project concept in one diagram

```text
                         PROJECT
                            │
              ┌─────────────┴─────────────┐
              │                           │
       EVALUATION MODE              APPLICATION MODE
       console benchmark                 web app
              │                           │
          CORNE-Val                 arbitrary image
              │                           │
          shot + mask                 user mask
              │                           │
              └─────────────┬─────────────┘
                            │
                ┌───────────┼───────────┐
                │           │           │
             OpenCV       LaMa       ViTInpainter     FLUX
             Telea         CNN       Transformer     Diffusion
                │           │           │              │
                └───────────┴───────────┴──────────────┘
                            │
                    generated result
                            │
                ┌───────────┴───────────┐
                │                       │
          quantitative eval       visual comparison
          PSNR/SSIM/LPIPS         carousel + loading
          + inference time
```

## 16. Course alignment

The project is designed around the course requirements:

- clear research/engineering question;
- baseline before advanced methods;
- controlled model/mask comparison;
- at least two quantitative metrics;
- sensitivity experiment;
- failure analysis;
- reproducible evaluation;
- individual understanding of the chosen methods.

The course guidelines explicitly encourage pretrained models and emphasize experimental reasoning, comparison, quantitative evaluation, and failure analysis rather than large-scale training.
