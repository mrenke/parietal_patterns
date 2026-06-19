# common_connectivity_stuff — Shared Connectivity Utilities

Shared scripts for connectivity matrix generation and network-mapping analyses, reused across multiple datasets and analysis branches.

## Scripts
| Script | What it does | Used by |
|--------|-------------|---------|
| `genCM_01.py` | Generate correlation matrices from fMRI time series (36P, scrubbing, fsaverage5) — single-dataset version | DNumRisk analysis |
| `fitNetMaps_assign_consens_plot.py` | Fit Infomap network maps, assign to reference atlas, run consensus procedure, and plot | StressRisk reliability analysis |
| `utils.py` | Shared utility functions | all |

## Notebooks
| Notebook | What it does |
|----------|-------------|
| `genCM_01.py` → run as script | See above |
| `inter_intra_net_conn_01.ipynb` | Inter- and intra-network connectivity analysis |
| `participation_coef_01.ipynb` | Participation coefficient analysis |
| `netMapping_reliability_stressrisk.ipynb` | Network mapping reliability across subjects in StressRisk dataset |

## Datasets
- DNumRisk (CM generation, inter/intra connectivity)
- StressRisk (network mapping reliability — `ds-stressrisk/derivatives/networks_infomap_full/`)

## Related folders
- `dyscalculia_datapool_ana/genCM_concatTS.py` — extended multi-session/task version of `genCM_01.py`
- `networks_indTopology/` — uses similar network assignment pipeline
