# asd_pfm_ana — ASD cohort, parietal/DAN-patch follow-up

**Status: 🟡 PLANNED** — sibling to `charles_01/` (archived). New, more relevant notebooks/scripts for this dataset go here, written in the same style as the rest of the `parietal_patterns` pipeline (cf. `nets_PFM/`).

## Background

`charles_01/` holds the archived BIDS reorganization + early gradient work Charles (master student) did on this dataset before splitting off into his own repo: [fMRI-autism-analysis](https://github.com/charles-neuro/fMRI-autism-analysis/tree/main). That folder is left as-is; this one is for picking the dataset back up with new questions (see `~/obsidian-wiki/concepts/Autism.md`).

## Dataset

- **BIDS root:** `/mnt_AdaBD_largefiles/Data/DNumRisk_Data/ds-asd` (per `charles_01/fit_PFM_av01.py`)

## To-do

- [ ] **Replicate the parietal DAN-patch finding in the ASD cohort.** In `nets_PFM/` (DNumRisk, DD vs TD), the bilateral parietal-lateral DAN patch was significantly *smaller* in DD (p_bonf = 0.043; see `nets_PFM/CLAUDE.md` → `npc_net_ana.ipynb`). Test whether this patch is comparatively *larger* (or otherwise different) in this ASD cohort, using the same PFM pipeline (fsLR 32k, Infomap parcellation, patch decomposition by nearest reference-patch centroid) applied to `ds-asd`.
