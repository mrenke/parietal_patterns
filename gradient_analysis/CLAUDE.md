# gradient_analysis — ARCHIVED

Status: **ARCHIVED** — first full gradient pipeline on DNumRisk. Superseded by `gradients_noHalo/` (preprocessing experiments) and ultimately by `nets_PFM/`.

> Earlier gradient work (first pass on a smaller NumRisk dataset) lives in a separate `numrisk` git repo. This folder is the first full-scale gradient analysis on DNumRisk.

## Goal
Fit individual-subject connectivity gradients (BrainSpace Laplacian eigenmap) and measure network dispersion — how spread-out a given network is in gradient space. Compare dyscalculia vs controls on gradient position and network dispersion.

## Pipeline
1. Connectivity matrix: fsaverage5 space, **36P** confound model (6 motion + derivatives + power² + WM/CSF/GS), scrubbing FD > 0.3 mm, ≥104 frames/run
2. Gradient fitting: `GradientMaps` (BrainSpace), aligned to group reference
3. Network dispersion: variance of gradient values within each network ROI

## Scripts
- `fit_gradients_aligned.py` — Fit per-subject gradients aligned to a group reference
- `utils.py`, `utils_02.py` — Shared utilities

## Notebooks
| Notebook | What it does |
|----------|-------------|
| `average_gradients_01.ipynb` | Group-average gradient maps |
| `vis_gradients_01.ipynb` | Gradient visualization |
| `networkDispersion_01.ipynb` | Network dispersion analysis |
| `networkDispersion_NPC_01.ipynb` | Network dispersion for NPC (numerosity processing cortex) |
| `networkDispersion_miguel.ipynb` | Network dispersion for Miguel's data |
| `betaGMs_groupDiffs.ipynb` | Group differences in beta-weighted gradient maps |
| `grad-ND_behave_corr_01.ipynb` | Gradient / network dispersion vs behavioral scores |
| `debug_fit_gradients_aligned.ipynb` | Debugging gradient alignment |

## Datasets
- Primary: DNumRisk (`/mnt_AdaBD_largefiles/Data/SMILE_Data/DNumRisk/ds-dnumrisk`)
- Data derivatives: `derivatives/gradients/`, `derivatives/correlation_matrices/`

## Why archived
The "halo" artifact in the CC mask, and questions about 32P vs 36P denoising, led to the systematic exploration in `gradients_noHalo/`. The final pipeline choice (Friston-24P, CoV masking) is implemented in `nets_PFM/` using fsLR 32k.
