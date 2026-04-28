# Precision Functional Mapping (PFM) — fsLR 32k Pipeline

Replication of Gordon et al. 2017 (Neuron, *Precision Functional Mapping of Individual Human Brains*), previously implemented in fsaverage5 space. This version follows the original pipeline more closely, using CIFTI format and fsLR 32k space.

## Goal

Generate individual-specific functional connectomes (parcel-to-parcel correlation matrices) in fsLR 32k space, then run Infomap-based network parcellation.

## Dataset

- **BIDS root:** `/mnt_03/ds-dnumrisk`
- **Subjects:** sub-01 through sub-15
- **Sessions:** ses-1 only (for now)
- **Task:** task-magjudge, 6 runs per subject
- **BOLD:** 188 volumes, TR = 2.298 s, voxels = 2.5 × 2.5 × 3.0 mm, space = T1w (59 × 71 × 49)

## Key File Paths

### Per-run inputs (replace `sub-XX` and `run-N`)
```
BIDS_FUNC = /mnt_03/ds-dnumrisk/derivatives/fmriprep/sub-XX/ses-1/func/

bold_T1w  = sub-XX_ses-1_task-magjudge_run-N_space-T1w_desc-preproc_bold.nii.gz
confounds = sub-XX_ses-1_task-magjudge_run-N_desc-confounds_timeseries.tsv
aparcaseg = sub-XX_ses-1_task-magjudge_run-N_space-T1w_desc-aparcaseg_dseg.nii.gz
```

### Session-level anatomical (fMRIPrep — surfaces in T1w space, fsnative density)
```
ANAT = /mnt_03/ds-dnumrisk/derivatives/fmriprep/sub-XX/ses-1/anat/

smoothwm_L   = sub-XX_ses-1_hemi-L_smoothwm.surf.gii     # ribbon inner boundary
pial_L       = sub-XX_ses-1_hemi-L_pial.surf.gii          # ribbon outer boundary
midthick_L   = sub-XX_ses-1_hemi-L_midthickness.surf.gii  # output surface
(same for hemi-R)
```

### Registration / template surfaces
```
# Subject's fsnative → fsaverage registration sphere (FreeSurfer)
sphere_reg_L = /mnt_03/ds-dnumrisk/derivatives/freesurfer/sub-XX/surf/lh.sphere.reg.surf.gii

# fsLR 32k target sphere expressed in fsaverage coordinates (HCPpipelines)
HCP_RESAMPLE = /home/ubuntu/git/HCPpipelines/global/templates/standard_mesh_atlases/resample_fsaverage/
fs_LR_sphere_L = fs_LR-deformed_to-fsaverage.L.sphere.32k_fs_LR.surf.gii
fs_LR_sphere_R = fs_LR-deformed_to-fsaverage.R.sphere.32k_fs_LR.surf.gii

# fsLR 32k template midthickness (neuromaps)
NEUROMAPS_FSLR = /home/ubuntu/neuromaps-data/atlases/fsLR/
tpl_midthick_L = tpl-fsLR_den-32k_hemi-L_midthickness.surf.gii
tpl_sphere_L   = tpl-fsLR_den-32k_hemi-L_sphere.surf.gii
(same for R)
```

### Tools
```
wb_command  = /home/ubuntu/workbench/bin_linux64/wb_command  (v2.0.1)
FreeSurfer  = /home/ubuntu/freesurfer  (v7.3.2)
Python env  = conda run -n numrefields python  (nilearn 0.10.4, nibabel 5.2.1,
                                                numpy 1.26.4, scipy 1.13.1,
                                                infomap 2.8.0)
```

## Pipeline Steps

### Step 1 — Denoise (T1w volume space, per run)

Gordon uses the **Friston 1996 Volterra expansion** for motion + 3 physiological signals.
This is NOT the same as the 36-parameter model (see `pipeline_comparison.md`).

**Motion regressors — Friston 24-parameter (from fMRIPrep confounds TSV):**
- 6 motion params at time t (`trans_x/y/z`, `rot_x/y/z`)
- 6 motion params at time t−1 (shift columns by 1 TR — not directly in TSV, compute manually)
- 6 squared motion params at t (power2 columns: `trans_x_power2` etc.)
- 6 squared motion params at t−1 (shift the power2 columns by 1 TR)

**Physiological signals (3 regressors, no derivatives/power2):**
- `global_signal`, `white_matter`, `csf`

**Scrubbing — interpolate-then-excise (Power et al. 2014):**
1. Flag frames with FD > 0.2 mm (`framewise_displacement` column)
2. Regress out confounds with flagged frames excluded from beta estimation
3. **Interpolate** across flagged frames using spectral method before bandpass
4. Bandpass filter 0.009–0.08 Hz
5. **Excise** flagged frames from the data

**Pre-surface masking:**
- Before ribbon-constrained sampling, exclude voxels with temporal CoV > 0.5 SD
  above mean CoV of nearby voxels (5 mm Gaussian neighbourhood) — removes dropout/artefact voxels

### Step 2 — Ribbon-constrained volume → fsnative surface (per run, per hemi)
```bash
wb_command -volume-to-surface-mapping \
    bold_denoised.nii.gz midthickness.surf.gii out_fsnative.func.gii \
    -ribbon-constrained smoothwm.surf.gii pial.surf.gii
```
- Input volume and surfaces must share the same coordinate space (T1w) ✓
- Note: fMRIPrep outputs `smoothwm` not `white`; this is fine as inner ribbon boundary

### Step 3 — Resample fsnative → fsLR 32k (per run, per hemi)
```bash
wb_command -metric-resample \
    out_fsnative.func.gii \
    lh.sphere.reg.surf.gii \
    fs_LR-deformed_to-fsaverage.L.sphere.32k_fs_LR.surf.gii \
    ADAP_BARY_AREA \
    out_fsLR32k.func.gii \
    -area-surfs native_midthickness.surf.gii tpl-fsLR_den-32k_hemi-L_midthickness.surf.gii
```
- Resampling path: fsnative → fsaverage (via `sphere.reg`) → fsLR 32k (via HCP deformed sphere)

### Step 4 — Extract subcortical + cerebellar time series (per run)
- Source: T1w-space `aparcaseg_dseg.nii.gz` (FreeSurfer segmentation of native-space T1)
- Structures: bilateral accumbens, amygdala, caudate, hippocampus, pallidum, putamen, thalamus + **cerebellar voxels**
- Keep as individual voxels (not averaged into ROIs) — voxel-level data goes into CIFTI

### Step 5 — Assemble CIFTI dtseries (per run)
```bash
wb_command -cifti-create-dense-timeseries \
    out.dtseries.nii \
    -left-metric L.32k.func.gii -right-metric R.32k.func.gii \
    -volume subcortical_ts.nii.gz structure_labels.nii.gz \
    -timestep 2.298
```

### Step 6 — Smooth CIFTI (per run)
```bash
wb_command -cifti-smoothing \
    out.dtseries.nii \
    2.55 2.55 COLUMN out_smooth.dtseries.nii \
    -left-surface tpl-fsLR_den-32k_hemi-L_midthickness.surf.gii \
    -right-surface tpl-fsLR_den-32k_hemi-R_midthickness.surf.gii
```
- σ = 2.55 mm (geodesic on surface, Euclidean on volume), as in Gordon 2017

### Step 7 — Concatenate runs
```bash
wb_command -cifti-merge out_concat.dtseries.nii \
    -cifti run1.dtseries.nii -cifti run2.dtseries.nii ...
```

### Step 8 — Vertex-wise correlation matrix (`03_vertex_cm.py`)
All-to-all cross-correlation of CIFTI timeseries (cortical vertices + subcortical voxels).
The full ~90k × 90k dense matrix is never materialised — computed in row-chunks of 200 nodes.

- **30 mm proximity zeroing:** same-hemisphere surface pairs within 30 mm set to 0.
  Distance approximated as Euclidean on midthickness coordinates (conservative:
  Euclidean ≤ geodesic, so slightly fewer pairs are zeroed than with true geodesic).
  Inter-hemispheric and subcortical pairs are retained regardless of distance.
- **Threshold estimation:** random sample of 100k pairs → empirical r-value distribution
  → compute r_cutoff for each target density.
- **Target densities:** 0.3%, 0.5%, 1%, 2%, 3%, 5%  (fraction of all n*(n-1)/2 pairs kept)
- **Output:** one `scipy.sparse` CSR matrix (`.npz`) per density threshold.
  Symmetrised (upper + lower triangle stored). Plus a `_cm_meta.npz` with node
  coordinates and hemisphere IDs (needed for Infomap).
- **Disk estimate:** ~120 MB (0.3%) to ~2 GB (5%) per subject.

### Step 9 — Infomap network detection (`04_infomap.py`)
- Two-level Infomap (undirected, weighted) on each thresholded sparse graph.
- Communities < 400 nodes removed (as in Gordon 2017).
- Consensus assignment: mode community label across all density thresholds.
- Requires `pip install infomap` (not yet installed on this system).
- **Network labelling:** raw integer module IDs are output; mapping to named networks
  (DMN, FPN, DAN, etc.) by spatial overlap with a group-average atlas is a
  separate step needing the Gordon 2016 network atlas in fsLR 32k space.

### Optional utility: `03_corr_matrix.py`
Parcellated CM (parcel-to-parcel, using a `.dlabel.nii` atlas). Not used for
Infomap network detection, but useful for seed-based connectivity analyses,
QC, or comparison with other datasets.

## Output Structure (planned)
```
nets_PFM/
  outputs/
    sub-XX/
      denoised/   run-level denoised T1w volumes (temp)
      surface/    run-level fsLR 32k .func.gii
      cifti/      run-level .dtseries.nii
      concat/     session-level concatenated .dtseries.nii
      matrices/   parcel × parcel correlation matrices (.npy or .nii)
```

## Status

- [x] Environment checked, paths confirmed
- [x] Pipeline design agreed (corrected vs. original 36-param plan — see pipeline_comparison.md)
- [x] `01_denoise.py` — written
- [x] `02_surface_cifti.py` — written (ribbon map → fsLR 32k → CIFTI → smooth → concat)
- [x] `03_vertex_cm.py` — written (chunked vertex-wise sparse CM)
- [x] `04_infomap.py` — written (Infomap + consensus)
- [ ] Test on sub-01 (run all 4 scripts end-to-end)
- [x] infomap 2.8.0 available in `numrefields` conda env (all deps confirmed)
- [ ] Obtain network labelling atlas (Gordon 2016 networks in fsLR 32k)
- [ ] Apply to all subjects

## References

- Gordon et al. 2017, Neuron — original PFM paper
- HCPpipelines resampling templates: `/home/ubuntu/git/HCPpipelines/global/templates/standard_mesh_atlases/`
- Infomap wrapper: https://github.com/MidnightScanClub/MSCcodebase/tree/master/Utilities/Infomap_wrapper
