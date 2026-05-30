# Data Server Overview — DNumRisk storage

Base path: `/mnt_AdaBD_largefiles/Data/SMILE_Data/DNumRisk/`

This file documents all derivative folders on the data server so we can decide what to keep vs. delete.

---

## Dataset folders

| Folder | Description | N subjects |
|--------|-------------|-----------|
| `ds-dnumrisk/` | **Main dataset** — magnitude judgment, fmriprep + all analysis derivatives | 66 |
| `ds-numrisk/` | Replication dataset — same task, all healthy controls | 64 |
| `ds-numberline/` | Numberline task — DD + TD, multi-session | — |
| `ds-smile/` | SMILE arithmetic task — DD + TD, multi-session | — |
| `ds-stressrisk/` | Stress-risk cohort — network reliability only | — |
| `sub-04/` | Stray subject folder at root — contents unclear | — |

## Shared reference data (root level)

| Folder / file | Contents | Keep? |
|---------------|----------|-------|
| `atlases_parcellations/` | Glasser, HCPMMP1, ColeAnticevic parcellations (fsaverage/fsaverage5) | ✅ KEEP |
| `netAtlas_Gordon_17/` | Gordon 2017 network atlas | ✅ KEEP |
| `gradients_references/` | Reference gradient maps | ✅ KEEP |
| `AAL3/`, `AAL3.zip` | AAL3 atlas | ✅ KEEP |
| `GlasserParcellationAtlas_MNI/` | Glasser atlas in MNI | ✅ KEEP |
| `GlasserParcellationAtlas_MNI_hemiUnique/` | Hemi-unique version | ✅ KEEP |
| `JHU-ICBM-labels-1mm_MNI152NLin2009cAsym.nii.gz` | JHU DTI atlas | ✅ KEEP |
| `for_Valerina/`, `from_Karin/`, `dwi_forAnnasCP/` | Data shares for collaborators | ❓ Check if still needed |
| `test_write.txt`, `try.txt` | Test files | 🗑️ DELETE |

---

## ds-dnumrisk/derivatives/

### Core / irreplaceable (fMRIPrep outputs)

| Folder | Description | Status |
|--------|-------------|--------|
| `fmriprep/` | fMRIPrep preprocessing | ✅ KEEP — cannot re-run cheaply |
| `phenotype/` | Behavioral / phenotype data | ✅ KEEP |
| `behavioral_files/` | Task timing + behavioral scores | ✅ KEEP |
| `pupil/` | Pupillometry data | ✅ KEEP |

### GLM outputs

| Folder | Description | Status |
|--------|-------------|--------|
| `glm_stim1.denoise/` | GLM (1st stimulus type, denoised) | ✅ KEEP |
| `glm_stim2.denoise/` | GLM (2nd stimulus type, denoised) | ✅ KEEP |
| `average_act.glm/` | Average activation maps | ✅ KEEP |
| `activation_maps_bins.glmav/` | Activation binned by gradient | ✅ KEEP |
| `binning_results/` | Gradient-bin analysis results | ✅ KEEP |

### Correlation matrices

| Folder | Description | Status |
|--------|-------------|--------|
| `correlation_matrices/` | **Final** CMs (36P, scrubbing) — standard pipeline | ✅ KEEP |
| `correlation_matrices.parcel/` | Parcel-level CMs | ✅ KEEP |
| `correlation_matrices.tryNoHalo/` | Experimental: CoV halo removal | ❓ Keep if `gradients_noHalo` results needed |
| `correlation_matrices.glmsingle/` | GLMsingle-based CMs (Sophie's approach) | ❓ Probably dispensable |
| `SD_corr_matrix_dfs/` | SD of correlation matrices (QC) | ❓ Check if used |
| `corr_connectivity_grads/` | Correlation between connectivity and gradients | ✅ KEEP (results) |
| `corr_usable-frames_grad-range_cm-sd/` | QC: scan length vs gradient range | ✅ KEEP (QC results) |

### Gradients — **FINAL pipeline** (use these)

| Folder | Description | Status |
|--------|-------------|--------|
| `gradients/` | Main gradients (36P, scrubbing, fsaverage5) | ✅ KEEP — **main analysis** |
| `gradients.36Pscrub3BPfilterrunFD104-6runs/` | 36P, all 6 runs variant | ✅ KEEP |

### Gradients — **EXPERIMENTAL** (preprocessing exploration, `gradients_noHalo/`)

| Folder | Description | Status |
|--------|-------------|--------|
| `gradients.tryNoHalo/` | Halo removal experiment | 🗑️ Can delete after noting results |
| `gradients.tryParams/` | Parameter grid search | 🗑️ Can delete |
| `gradients.tryParams.36P/` | 36P parameter grid search | 🗑️ Can delete |
| `gradients.32Pscrub3BPfilterrunFD104/` | 32P model variant | 🗑️ Can delete |
| `gradients.32Pscrub3BPfilterrunFD104_oldCCmask/` | 32P + old CC mask | 🗑️ Can delete |
| `gradients.32scrub3BPfilterrunFD104/` | 32P no-BP variant | 🗑️ Can delete |
| `gradients.36Pscrub3BPfilterrunFD104/` | 36P standard run | ✅ KEEP — likely the "canonical" 36P result |
| `gradients.36Pscrub3BPfilterrunFD104_oldCCmask/` | 36P + old CC mask | 🗑️ Can delete |
| `gradients.36Pscrub3runFD104/` | 36P, no BPF | 🗑️ Can delete |
| `gradients.glmsingle/` | GLMsingle-based gradients | ❓ Keep if comparing with standard |
| `gradients.glmsingle_wrong_arctan/` | Buggy version (wrong arctan) | 🗑️ DELETE |
| `marg_gradients/` | Marginal/averaged gradients | ❓ Check if used |
| `sensitivity_analyses/` | Sensitivity analysis outputs | ✅ KEEP |

### Networks (Infomap) — `networks_indTopology/` and `nets_PFM/`

| Folder | Description | Pipeline | Status |
|--------|-------------|----------|--------|
| `networks_infomap_full_01/` | **Most recent** Infomap (fsaverage5) | `networks_indTopology/` | ✅ KEEP — final version |
| `networks_infomap_full/` | Earlier full Infomap | `networks_indTopology/` | ❓ May be superseded by _01 |
| `networks_infomap/` | Older Infomap run | `networks_indTopology/` | 🗑️ Superseded |
| `networks_infomap_EG17nets/` | Infomap matched to EG17 nets | `networks_indTopology/` | ✅ KEEP if used in analysis |
| `networks_infomap_singleThresh/` | Single-threshold version | `networks_indTopology/` | 🗑️ Superseded by consensus |

### DTI

| Folder | Description | Status |
|--------|-------------|--------|
| `dwi_preproc/` | DWI preprocessing | ✅ KEEP |
| `dwi_FA2ndLevel/` | FA 2nd-level analysis | ✅ KEEP |
| `dwi_connectome/` | Structural connectome | ✅ KEEP |

---

## ds-numrisk/derivatives/

| Folder | Description | Status |
|--------|-------------|--------|
| `fmriprep/` | fMRIPrep (NumRisk) | ✅ KEEP |
| `freesurfer/` | FreeSurfer recons | ✅ KEEP |
| `correlation_matrices/` | Main CMs | ✅ KEEP |
| `correlation_matrices.glmsingle/` | GLMsingle CMs | ❓ Keep if comparing methods |
| `glm_stim.denoise/` | GLM denoised | ✅ KEEP |
| `glm_stim.denoise.coOccCV/` | Co-occurrence CV variant | ❓ Check if used |
| `glm_stim.denoise.fsaverage5/` | GLM in fsaverage5 | ❓ Check if used |
| `glm_stim1.denoise_wrong/`, `glm_stim2.denoise_wrong/` | Buggy GLM versions | 🗑️ DELETE |
| `gradients/` | Main gradients | ✅ KEEP |
| `gradients.glmsingle/` | GLMsingle gradients | ❓ Keep if comparing methods |
| `gradients.glmsingle_wrong/` | Buggy GLMsingle version | 🗑️ DELETE |
| `SD_corr_matrix_dfs/`, `binning_results/`, `activation_maps_bins.glmav/`, `average_act.glm/` | Analysis outputs | ✅ KEEP |

---

## ds-smile/derivatives/

| Folder | Description | Status |
|--------|-------------|--------|
| `correlation_matrices/` | CMs for SMILE dataset | ✅ KEEP |

---

## ds-numberline/derivatives/

| Folder | Description | Status |
|--------|-------------|--------|
| `gradients/` | Gradients for numberline dataset | ✅ KEEP |

---

## ds-stressrisk/derivatives/

| Folder | Description | Status |
|--------|-------------|--------|
| `fmriprep/`, `freesurfer/` | Preprocessing | ✅ KEEP |
| `correlation_matrices.tryNoHalo/` | Experimental CM (halo removal) | ❓ Keep for network reliability analysis |
| `networks_infomap_full/` | Infomap networks | ✅ KEEP — used in `common_connectivity_stuff/netMapping_reliability_stressrisk.ipynb` |

---

## Summary — Deletion candidates

Items marked 🗑️ above are candidates for deletion. Before deleting, confirm:

1. `gradients.glmsingle_wrong_arctan/` (ds-dnumrisk) — clearly buggy, delete
2. `glm_stim1.denoise_wrong/`, `glm_stim2.denoise_wrong/` (ds-numrisk) — clearly buggy, delete
3. `gradients.glmsingle_wrong/` (ds-numrisk) — clearly buggy, delete
4. All experimental gradient parameter folders (see list above) — but first confirm which 36P version was the "canonical" one used in publications
5. `test_write.txt`, `try.txt` at root — delete
6. Old Infomap runs superseded by `networks_infomap_full_01/`

> **Before any deletion:** run `du -sh` on each candidate folder to estimate space savings.
