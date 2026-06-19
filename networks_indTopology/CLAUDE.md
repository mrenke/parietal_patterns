# networks_indTopology — ARCHIVED (superseded by nets_PFM)

Status: **ARCHIVED** — first implementation of individual network topology via Infomap community detection. **Superseded by `nets_PFM/`** which uses the correct Gordon 2017 pipeline in fsLR 32k space.

See `individual_nework_maps_overview.md` in this folder for detailed method notes on the Gordon 2017 consensus assignment procedure and references.

## What it did
Individual-specific network maps using Infomap on whole-brain connectivity matrices, in fsaverage5 space with 36P denoising. Explored network assignment, consensus across thresholds, and group comparisons.

## Key differences from `nets_PFM/`
| | `networks_indTopology/` | `nets_PFM/` |
|--|------------------------|-------------|
| Surface space | fsaverage5 | fsLR 32k (CIFTI) |
| Denoising | 36P | Friston-24P (Gordon 2017) |
| Subcortical | No | Yes (CIFTI includes subcortex) |
| Smoothing | Volumetric | Surface geodesic |
| Scrubbing | interpolate-then-excise | interpolate-then-excise (corrected) |
| Consensus | partial | Full Gordon 2017 procedure |

## Scripts
| Script | What it does |
|--------|-------------|
| `fit_networks_wholeBrain.py` | Run Infomap on whole-brain CM |
| `fit_assign_consens_plot_nets.py` | Assign + consensus across thresholds + plot |
| `fit_singleThresh_assign_plot_nets.py` | Single-threshold version |
| `fit_to_EG17nets.py` | Match communities to EG17 atlas |
| `net_assign_plots.py` | Network assignment plots |
| `sum_figures_nets-gradients.py` | Summary figures combining nets + gradients |
| `import_fs_sub_pycortex.py` | Import FreeSurfer subject to pycortex |

## Notebooks
| Notebook | What it does |
|----------|-------------|
| `net_assignment_01/02.ipynb` | Network assignment exploration |
| `prec_func_mapping_infomap_01.ipynb` | Initial PFM/Infomap exploration |
| `average_nets_01.ipynb` | Average network maps across subjects |
| `nets_and_gradients_01.ipynb` | Combined network + gradient analysis |
| `nets_and_grads_groupComp_01.ipynb` | Group comparison: nets + gradients |
| `groupComp_DD_01.ipynb` | DD vs TD group comparison |
| `remove_smallNetPieces_01.ipynb` | Remove small network fragments |
| `salienceNet_01.ipynb` | Salience network analysis |
| `vis_atlases_01.ipynb` | Atlas visualization |
| `vis_pycortex_scloud_01.ipynb` | pycortex visualization |

## Related data (ds-dnumrisk/derivatives/)
- `networks_infomap/` — older runs
- `networks_infomap_EG17nets/` — matched to EG17 atlas
- `networks_infomap_singleThresh/` — single threshold
- `networks_infomap_full/` — earlier full run
- `networks_infomap_full_01/` — most recent run (keep this one)
