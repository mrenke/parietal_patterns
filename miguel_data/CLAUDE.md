# miguel_data — ds-numrisk Data Preparation & GLM

Data preparation and GLM fitting for the **NumRisk** dataset (`ds-numrisk`), including GLMDenoise/GLMSingle and behavioral analyses.

## Dataset
NumRisk (`/mnt_AdaBD_largefiles/Data/SMILE_Data/DNumRisk/ds-numrisk`)

## Contents
| File | What it does |
|------|-------------|
| `fit_glmDenoise_bothStim.py` | Run GLMDenoise on both stimulus conditions |
| `try_fit_glmSingle.ipynb` | Explore GLMSingle fitting |
| `reorga_fmri_BIDS_sophie.ipynb` | BIDS reorganization of NumRisk fMRI data |
| `behavior_bauer.ipynb` | Behavioral data analysis |
| `behavior_probit.ipynb` | Probit model for behavioral responses |

## Status
Analysis done. 

## Notes
- GLMsingle outputs for NumRisk live in `ds-numrisk/derivatives/correlation_matrices.glmsingle/` and `gradients.glmsingle/`
- Related GLMsingle work on NumRisk also in `sophie/`
