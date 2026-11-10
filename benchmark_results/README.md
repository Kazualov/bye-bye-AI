# Final benchmark results

Results of the full CORNE-Val benchmark (218 samples × 6 mask experiments = 1,308 runs per model). These are the numbers used in the final report.

## How they were produced

- **Code:** commit `34c3366` (branch `chore/reproducibility`). Run from a fresh clone, following the main README step by step.
- **Command:** `python -m evaluation.benchmark --dataset data/corne-val --output results/`
- **Environment:** `requirements-lock.txt`, Python 3.12.10.
- **Hardware:** Apple M5, 16 GB RAM, macOS 26.6. LaMa and ViTInpainter ran on the GPU (MPS); Telea ran on the CPU.
- **Date and duration:** 2026-11-09. The full run took about 4 h 48 min, including metric computation.
- **ViTInpainter settings:** `seed=0`, `auto_dilate=5` (the defaults; the same settings the web demo uses).

Telea and LaMa match the Week 5 report exactly. ViTInpainter differs from Week 5 because those numbers were produced before its default `auto_dilate` changed from 0 to 5 px.

## Region metrics (mean ± std over 218 samples)

Best value per mask in bold. PSNR and SSIM: higher is better. LPIPS and time: lower is better. Inference times depend on the hardware above.

| Mask | Model | PSNR reg ↑ | SSIM reg ↑ | LPIPS reg ↓ | Time, s ↓ |
|---|---|---|---|---|---|
| `sam_native` | OpenCV Telea | 12.74 ± 4.01 | 0.603 ± 0.146 | 0.487 ± 0.111 | **0.015** |
| `sam_native` | LaMa | **15.94** ± 4.96 | **0.657** ± 0.172 | **0.359** ± 0.099 | 0.373 |
| `sam_native` | ViTInpainter | 13.91 ± 4.52 | 0.609 ± 0.167 | 0.405 ± 0.111 | 0.268 |
| `sam_dilated10` | OpenCV Telea | 14.65 ± 4.48 | 0.664 ± 0.147 | 0.446 ± 0.099 | **0.021** |
| `sam_dilated10` | LaMa | **16.83** ± 5.34 | **0.670** ± 0.177 | **0.357** ± 0.096 | 0.362 |
| `sam_dilated10` | ViTInpainter | 14.88 ± 4.68 | 0.630 ± 0.168 | 0.396 ± 0.099 | 0.269 |
| `sam_dilated20` | OpenCV Telea | 15.26 ± 4.75 | 0.671 ± 0.149 | 0.453 ± 0.095 | **0.024** |
| `sam_dilated20` | LaMa | **17.53** ± 5.79 | **0.672** ± 0.181 | **0.358** ± 0.092 | 0.352 |
| `sam_dilated20` | ViTInpainter | 15.78 ± 5.36 | 0.636 ± 0.173 | 0.396 ± 0.097 | 0.280 |
| `sam_bbox` | OpenCV Telea | 14.45 ± 4.09 | 0.545 ± 0.196 | 0.608 ± 0.091 | **0.029** |
| `sam_bbox` | LaMa | **17.26** ± 5.13 | **0.588** ± 0.226 | **0.456** ± 0.110 | 0.360 |
| `sam_bbox` | ViTInpainter | 15.28 ± 4.82 | 0.498 ± 0.224 | 0.517 ± 0.101 | 0.278 |
| `sam_ellipse` | OpenCV Telea | 12.49 ± 3.57 | 0.506 ± 0.162 | 0.605 ± 0.077 | **0.026** |
| `sam_ellipse` | LaMa | **14.94** ± 4.46 | **0.545** ± 0.198 | **0.483** ± 0.092 | 0.357 |
| `sam_ellipse` | ViTInpainter | 11.86 ± 3.70 | 0.421 ± 0.167 | 0.565 ± 0.087 | 0.273 |
| `eff_native` | OpenCV Telea | 17.42 ± 5.40 | **0.722** ± 0.146 | 0.420 ± 0.099 | **0.027** |
| `eff_native` | LaMa | **19.75** ± 6.06 | 0.711 ± 0.180 | **0.327** ± 0.090 | 0.359 |
| `eff_native` | ViTInpainter | 18.75 ± 5.83 | 0.690 ± 0.176 | 0.353 ± 0.092 | 0.273 |

## Files

| File | Contents |
|---|---|
| `final_summary.csv` | All models and masks in one table: mean and std of every metric (region and full image) and inference time |
| `<Model>/metrics.csv` | One row per (sample, mask experiment) |
| `<Model>/summary.csv` | Mean and std per mask experiment |
| `<Model>/best_per_sample.csv` | Best mask experiment per sample (by full-image LPIPS) |
| `<Model>/mask_ranking.csv` | Mask experiments ranked by mean full-image LPIPS |
| `<Model>/plots/` | Bar charts for `sam_native` |

The qualitative comparison images (`qualitative_best/`, ~450 MB) are not committed. Re-run the benchmark to regenerate them.
