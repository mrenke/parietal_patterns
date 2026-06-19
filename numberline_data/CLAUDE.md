# numberline_data — Numberline Dataset Analysis

Connectivity and network analyses for the numberline task dataset (dyscalculics + controls, multiple sessions).

## Dataset
`/mnt_AdaBD_largefiles/Data/SMILE_Data/DNumRisk/ds-numberline`
Derivatives: `derivatives/gradients/`

> **Note:** Not all subjects have all sessions — check `dyscalculia_datapool_ana/data_completness_01.ipynb`.

## Scripts
| Script | What it does |
|--------|-------------|
| `getCM_parcel.py` | Generate parcel-based correlation matrices |
| `getCM_vertex.py` | Generate vertex-wise correlation matrices |
| `utils.py` | Shared utilities |

## Notebooks
| Notebook | What it does |
|----------|-------------|
| `prep_data_into_BIDS.ipynb` | Data preparation / reorganization into BIDS format |
| `group_comparisons_01.ipynb` | Group comparisons: DD vs TD |
| `participCoeff_01.ipynb` | Participation coefficient analysis |

## Notes
- For multi-session pooled analysis: `dyscalculia_datapool_ana/genCM_concatTS.py`
- Method details in `analysis_idea_overview.md` (Gordon 2017 consensus assignment, participation coefficient, Wang 2022 network segregation approach)

## Status
Initial analysis done; part of the dyscalculia datapool comparison.
