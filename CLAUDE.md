# parietal_patterns — Project Overview

**Research theme:** Parietal network topology in dyscalculia and numerical cognition.
How is individual-specific functional network organization in parietal cortex different between dyscalculic and typically-developing individuals? Approaches: precision functional mapping (Infomap-based, Gordon 2017 style) and connectivity gradients (BrainSpace).

> **Note:** Earlier gradient analysis work predates this repo and lives in a separate `numrisk` git repository. This repo consolidates the evolved pipeline and all group-level analyses.

---

## Datasets

| Name | BIDS root | N | Task | Notes |
|------|-----------|---|------|-------|
| **DNumRisk** | `/mnt_AdaBD_largefiles/Data/SMILE_Data/DNumRisk/ds-dnumrisk` , `/mnt_03/ds-dnumrisk` | 66 | `magjudge` (magnitude comparison, 6 runs/subject) | **Main dataset** — ~half dyscalculia, ~half controls |
| **NumRisk** | `/mnt_AdaBD_largefiles/Data/SMILE_Data/DNumRisk/ds-numrisk` , `/mnt_04/ds-numrisk` | 64 | `magjudge` | All healthy controls — replication cohort |
| **Numberline** | `/mnt_AdaBD_largefiles/Data/SMILE_Data/DNumRisk/ds-numberline` | — | numberline task | Dyscalculics + controls, multiple sessions |
| **SMILE** | `/mnt_AdaBD_largefiles/Data/SMILE_Data/DNumRisk/ds-smile` | ~ 30 (ongoing!) |`magjudge` , `placevalue`, `rest` | Dyscalculics + controls, multiple sessions |
| **StressRisk** | `/mnt_AdaBD_largefiles/Data/SMILE_Data/DNumRisk/ds-stressrisk` | 50 | risk | Separate cohort, network reliability check |

Atlases and parcellations:
`/mnt_AdaBD_largefiles/Data/SMILE_Data/DNumRisk/atlases_parcellations/`

See `DATA_OVERVIEW.md` for a full audit of derivative folders on the data server.

---

## Analysis Folders

### 🟢 ACTIVE

| Folder | Dataset(s) | Method | Notes |
|--------|-----------|--------|-------|
| `nets_PFM/` | DNumRisk | Precision functional mapping (Gordon 2017, fsLR 32k, Infomap) + group-level NPC/DAN-patch analysis | Pipeline complete (65 subjects); NPC network sizes and DAN patch topology compared DD vs TD — **see `nets_PFM/CLAUDE.md`** |
| `asd_pfm_ana/` | ASD (`ds-asd`) | PFM (fsLR 32k, Infomap), parietal DAN-patch comparison | 🟡 Planned — sibling to `charles_01/`; testing whether the DD-vs-TD parietal DAN-patch finding from `nets_PFM/` replicates in this cohort — **see `asd_pfm_ana/CLAUDE.md`** |

### 🔵 ANALYSIS COMPLETE (archived or pending group-level work)

| Folder | Dataset(s) | Method | Notes |
|--------|-----------|--------|-------|
| `dyscalculia_datapool_ana/` | DNumRisk + Numberline + SMILE | Gradients + PFMs, group comparisons | DD vs TD; will need updating when nets_PFM (fsLR 32k) results are ready |
| `gradient_analysis/` | DNumRisk (+ others) | BrainSpace gradients, network dispersion | First gradient pipeline (fsaverage5, 36P); archived |
| `gradients_noHalo/` | DNumRisk, StressRisk | Preprocessing parameter exploration | 32P vs 36P, CoV halo removal, CC mask variants; informed `nets_PFM/` |
| `networks_indTopology/` | DNumRisk | Infomap on fsaverage5 CM | **Superseded by `nets_PFM/`** |
| `dti_analysis/` | DNumRisk | DTI FA analysis + structural connectome | DWI derivatives done |
| `numberline_data/` | Numberline | Gradients, participation coefficient | Multi-session analysis |
| `smile_data/` | SMILE | CMs, gradients | Data prep + CM computation done |
| `common_connectivity_stuff/` | DNumRisk, StressRisk | Shared CM / network-mapping utilities | Helper scripts |
| `activity_and_connectivity/` | DNumRisk | Gradient-bin vs activation | Single exploratory notebook |
| `com_nPRF_pfm/` | DNumRisk | nPRF vs DAN PFM spatial comparison | Single comparison notebook |

### 🔴 ARCHIVED COLLABORATIONS

| Folder | Collaborator | What it was |
|--------|-------------|-------------|
| `sophie/` | Sophie | GLMsingle-based connectivity on NumRisk |
| `charles_01/` | Charles | BIDS reorganization + early gradient fitting |
| `miguel_data/` | Miguel | GLMDenoise/GLMSingle + behavioral analysis |

### 📄 ROOT-LEVEL FILES

| File | Purpose |
|------|---------|
| `ohbm26_poster_plotsAstats_01.ipynb` | OHBM 2026 poster — plots and statistics |
| `check_EPIasymmetry_EllasRequest.ipynb` | One-off EPI asymmetry check for Ella |
| `try_netCorresToolbox.ipynb` | Exploratory: network correspondence toolbox |
| `utils/` | Shared Python package: `surfaces.py`, `statistics.py` |

---

## Key Tools & Paths

```
wb_command       /home/ubuntu/workbench/bin_linux64/wb_command  (v2.0.1)
FreeSurfer       /home/ubuntu/freesurfer  (v7.3.2)
HCPpipelines     /home/ubuntu/git/HCPpipelines/global/templates/
neuromaps        /home/ubuntu/neuromaps-data/atlases/
Python env       conda run -n numrefields python
                 → nilearn 0.10.4, nibabel 5.2.1, numpy 1.26.4,
                   scipy 1.13.1, brainspace, infomap 2.8.0
```

---

## Analysis Lineage

```
numrisk repo (older)
  └─ first gradient fits (fsaverage5, 36P denoising)
       ↓
gradient_analysis/          ← first full gradient pipeline on DNumRisk
       ↓
gradients_noHalo/           ← preprocessing experiments (halo, 32P/36P, CC mask)
       ↓
networks_indTopology/       ← first Infomap attempt (fsaverage5) — SUPERSEDED
       ↓
nets_PFM/                   ← CURRENT: correct Gordon 2017 pipeline (fsLR 32k)
```
