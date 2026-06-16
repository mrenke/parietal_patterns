# dti_analysis — DTI / Structural Connectivity

Structural connectivity analyses using diffusion-weighted imaging (DWI) from the DNumRisk dataset.

## Dataset
DNumRisk (`/mnt_AdaBD_largefiles/Data/SMILE_Data/DNumRisk/ds-dnumrisk`)

DWI derivatives:
- `derivatives/dwi_preproc/` — preprocessed DWI
- `derivatives/dwi_FA2ndLevel/` — FA 2nd-level analysis
- `derivatives/dwi_connectome/` — structural connectome matrices

## Analysis components
| Notebook / script | What it does |
|-------------------|-------------|
| `ants_reg_forFAana.ipynb` | ANTs registration: subject FA → MNI template |
| `FAnan_2ndLevel_01.ipynb` | 2nd-level FA analysis — group comparisons (DD vs TD) |
| `atlas_stuff.ipynb` | Atlas preparation, JHU white-matter atlas parcellation |
| `connectome_sanityChecks.ipynb` | QC for structural connectome matrices |
| `connectome_2ndLevel_01.ipynb` | 2nd-level structural connectome analysis |
| `connectome_2ndLevel_02.ipynb` | Further structural connectome analysis |
| `connectome_gradients_01.ipynb` | Structural connectivity gradients |
| `connectome_surface_plots.ipynb` | Surface visualization of connectome results |
| `inspect_connectome_results_01.ipynb` | Inspect and summarize results |
| `debug_connectomePipeline_01.ipynb` | Pipeline debugging notes |

## Status
Analysis complete. Contributes to the dyscalculia group comparison.
