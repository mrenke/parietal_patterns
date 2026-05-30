# smile_data — SMILE Dataset Analysis

Connectivity and gradient analyses for the SMILE dataset (dyscalculics + controls, multiple sessions, 2 tasks 20 mins each (magjudge & placevalue) + rest ~ 10 mins).

## Dataset
Caroline copied raw Niftis to `/mnt_AdaBD_largefiles/Data/SMILE_Data/measurements/mri_files` - for each subject's session a seperate .zip

--> working with the bids-folder in : `/mnt_AdaBD_largefiles/Data/DNumRisk_Data/ds-smile`

## Notebooks
| Notebook | What it does |
|----------|-------------|
| `prepare_data.ipynb` | Data preparation and organization |
| `check_completness.ipynb` | Subject/session completeness check |
| `cm_gm_tasks_01.ipynb` | Correlation matrices and gradient maps across task conditions |

## Status
- ongoing project --> Caroline incrementely adds newly aquired data 
- Sophie did  multi-task analysis with the dataset status ~ November 2025
- Maike: Part of the dyscalculia datapool comparison (`dyscalculia_datapool_ana/`).
