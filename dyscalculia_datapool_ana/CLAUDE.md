# dyscalculia_datapool_ana — Group Comparisons (DD vs TD)

Analyses comparing functional network organization between children/adults with **dyscalculia (DD)** and typically-developing controls (TD), pooling data across the available datasets with dyscalculic participants.

## Goal
Test whether individual-specific functional network topology (gradients + precision functional maps) differs systematically between DD and TD groups.

## Datasets used
| Dataset | Task | Sessions | Notes |
|---------|------|----------|-------|
| DNumRisk | magnitude judgment | 1 | N=66, ~half DD / half TD — primary dataset |
| Numberline (`ds-numberline`) | numberline task | multiple | not all subjects have all sessions |
| SMILE (`ds-smile`) | arithmetic | multiple | |

## Pipeline

### 1. Correlation matrix generation (multi-session)
`genCM_concatTS.py` — Concatenates time series across sessions and task conditions before computing the correlation matrix. Uses 36P confounds, scrubbing (FD > 0.3 mm), ≥104 retained frames/run.

> For DNumRisk (single session), CMs come from `common_connectivity_stuff/genCM_01.py`.

### 2. Gradient fitting
`fit_gradients_01.py` — Per-subject BrainSpace gradients aligned to group reference.

### 3. PFM fitting
`fit_precFuncMap_01.py` — Per-subject precision functional maps.

## Notebooks
| Notebook | What it does |
|----------|-------------|
| `data_completness_01.ipynb` | Check subject/session completeness across datasets |
| `gradients_interSubReliability_01.ipynb` | Inter-subject reliability of gradient measures |
| `groupComp_01_GMs.ipynb` | Group differences in gradient maps (DD vs TD) |
| `groupComp_02_PFMs.ipynb` | Group differences in precision functional maps |
| `precFuncMap_01.ipynb` | PFM overview and visualization |
| `std_effects_01.ipynb` | Standardized effect sizes for group differences |

## Status / next steps
- Group comparisons done for the existing (fsaverage5, 36P) pipeline
- Will need updating when `nets_PFM/` results (fsLR 32k, Friston-24P) are complete for DNumRisk — those will be the publishable PFM results
- Numberline: not all subjects have all sessions (see `data_completness_01.ipynb`)

## Related folders
- `gradient_analysis/` — earlier gradient pipeline these analyses built on
- `numberline_data/` — numberline-specific preprocessing
- `smile_data/` — SMILE-specific preprocessing
- `nets_PFM/` — the new (correct) PFM pipeline whose outputs will replace `fit_precFuncMap_01.py` results
