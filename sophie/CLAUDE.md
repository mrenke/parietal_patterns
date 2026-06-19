# sophie — ARCHIVED (GLMsingle Collaboration)

Status: **ARCHIVED** — collaboration with Sophie on GLMsingle-based connectivity for the NumRisk dataset.

## What this was
GLMsingle (Kay et al.) estimates improved single-trial response amplitudes using a custom HRF and cross-validated noise ceiling. This folder explored using GLMsingle beta estimates (rather than raw denoised time series) as the basis for connectivity matrix computation and gradient fitting on NumRisk.

## Scripts
| Script | What it does |
|--------|-------------|
| `fit_glmsingle_myattempt.py` | Run GLMsingle on NumRisk fMRI data |
| `get_cm.py` | Standard correlation matrix generation |
| `get_cm_glmsingle.py` | GLMsingle-based correlation matrices |
| `get_cm_new.py` | Updated CM generation |
| `subject_gradients.py` | Subject-level gradient fitting (standard) |
| `subject_gradients_glmsingle.py` | Subject-level gradient fitting (GLMsingle betas) |
| `corr_scan-len_dispersion.py` | Correlation between scan length (usable frames) and gradient dispersion |
| `surface_transformation_script.py` | Surface projection pipeline |
| `my_utils.py`, `utils_old.py` | Utilities |

## Notebooks
| Notebook | What it does |
|----------|-------------|
| `glmsingle_analysis_1.ipynb` | GLMsingle results and QC |
| `average_gradient.ipynb` | Group-average gradients (standard) |
| `average_gradient_glmsingle.ipynb` | Group-average gradients (GLMsingle) |
| `debug_glmsingle.ipynb`, `debuging_glmsinge_numrisk.ipynb` | Debugging |
| `fit_glmsingle_myattempt.py` (script) | Fitting |
| `surface_transformation.ipynb` | Surface projection |
| `NPC_mask_gradients_stimuli.ipynb` | NPC mask + gradients + stimulus analysis |

## Dataset
NumRisk (`/mnt_AdaBD_largefiles/Data/SMILE_Data/DNumRisk/ds-numrisk`)
Related derivatives: `derivatives/correlation_matrices.glmsingle/`, `derivatives/gradients.glmsingle/`

## Status
Archived. The GLMsingle approach was explored but the standard connectivity pipeline (resting-state style) was adopted as the primary method.
