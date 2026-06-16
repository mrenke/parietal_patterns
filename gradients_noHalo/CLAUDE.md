# gradients_noHalo — ARCHIVED (post-preprocessing pipeline exploration)

Status: **ARCHIVED** — systematic exploration of post-preprocessing parameters for gradient analysis on DNumRisk. Results informed the final pipeline in `nets_PFM/`.

## What this investigated
The gradient maps showed a "halo" artifact: instead of the expected anchoring along the full longitudinal motor cortex, the first gradient component anchored to the **most dorsal part of the brain only** — an abnormal pattern. This was caused by the denoising pipeline. This folder explored fixes:

- **Changing confound model:** 32P vs 36P, and ultimately Friston-24P
- **Adding / fixing bandpass filtering** — a key factor in gradient anchoring
- **z-transformation method:** switched to **arctanh (hyperbolic tangent)** for the correlation values before gradient fitting
- Old vs new CC mask variants
- Other post-processing parameters (scrubbing thresholds, FD cutoffs per run)

## Scripts
| Script | What it does |
|--------|-------------|
| `getCM_parcel.py` | Generate parcel-based correlation matrices |
| `getCM_specConf.py` | Generate CMs with a specified confound configuration |
| `fit_gradients.py` | Standard gradient fitting |
| `fit_gradients_ccMask.py` | Gradient fitting with custom CC mask |
| `fit_gradients_dParams.py` | Gradient fitting across different parameter sets |

## Notebooks
| Notebook | What it does |
|----------|-------------|
| `tryRemoveHalo_01.ipynb` | Initial halo exploration and diagnosis |
| `averageGMs_01.ipynb` | Average gradient maps across pipeline variants |
| `rep_groupDiffs_01/02/03.ipynb` | Group difference reports across pipeline variants |
| `rep_groupDiffs_dParams.ipynb` | Group diffs across parameter sets |
| `rep_correlations.ipynb` | Correlation-based reliability across variants |
| `compare_subwiseParams_across∂PostPreproc.ipynb` | Subject-wise comparison across post-preprocessing specs |
| `space_transforms.ipynb` | Surface space transformations |
| `vis_01.ipynb`, `vis_pycortex_cloud.ipynb` | Visualization |

## Datasets
- Primary: DNumRisk
- Also: StressRisk (`ds-stressrisk`) — network reliability check

## Related derivative folders (ds-dnumrisk)
All experimental output from this exploration (see `DATA_OVERVIEW.md` for deletion candidates):

| Folder | Confound spec |
|--------|--------------|
| `correlation_matrices.tryNoHalo/` | CMs from the fixed pipeline |
| `gradients.tryNoHalo/` | Gradients with halo resolved |
| `gradients.tryParams/`, `gradients.tryParams.36P/` | Parameter grid search |
| `gradients.32Pscrub3BPfilterrunFD104/` | 32P + scrub + BPF |
| `gradients.32Pscrub3BPfilterrunFD104_oldCCmask/` | 32P + old CC mask |
| `gradients.32scrub3BPfilterrunFD104/` | 32P only |
| `gradients.36Pscrub3BPfilterrunFD104/` | **36P + BPF** (canonical 36P result) |
| `gradients.36Pscrub3BPfilterrunFD104_oldCCmask/` | 36P + old CC mask |
| `gradients.36Pscrub3runFD104/` | 36P, no BPF |
| `gradients.glmsingle/` | GLMsingle-based gradients (Sophie collab) |
| `gradients.glmsingle_wrong_arctan/` | **Buggy** (wrong arctanh application) — delete |

## Outcome
Key fixes adopted going forward:
1. Correct bandpass filtering resolves the dorsal anchoring artifact
2. arctanh (hyperbolic tangent) for z-transformation of correlations
3. Friston-24P confound model aligns with Gordon 2017 — implemented in `nets_PFM/`

Most derivative folders here are experimental intermediates and can be deleted (see `DATA_OVERVIEW.md`).
