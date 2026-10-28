from pathlib import Path
import re

# --- Data paths ---
BIDS_ROOT   = Path('/mnt_asd/ds-asd')
FMRIPREP    = BIDS_ROOT / 'derivatives' / 'fmriprep'
FREESURFER  = BIDS_ROOT / 'derivatives' / 'freesurfer'
# Only {lh,rh}.sphere.reg per subject are kept here (2026-10-26), synced from
# the recon-all run inside fMRIPrep on the cluster
# (sciencecluster:/shares/zne.uzh/mrenke/ds-asd/derivatives/freesurfer) —
# same run as fMRIPrep's anat/*.surf.gii, vertex counts verified to match.

# No overrides needed — kept for interface parity with config_dnumrisk.py.
FREESURFER_OVERRIDE = {}
ANAT_DIR_OVERRIDE   = {}

# /mnt_asd is nearly full (~28 GB free) — redirect pfm_fslr outputs to the
# largefiles mount instead of writing next to the input data
# (~4-5 GB/subject x 98). /mnt_asd/ds-asd/derivatives/pfm_fslr is a symlink here.
OUTPUT_ROOT = Path('/mnt_AdaBD_largefiles/Data/DNumRisk_Data/ds-asd'
                    '/derivatives/pfm_fslr')

# Reference atlases (gordon17, ca19magjudgeDDnr) are dataset-independent labelling
# references already in fsLR 32k space — reuse DNumRisk's copy rather than
# duplicating them here.
ATLAS_DIR = Path('/mnt_03/ds-dnumrisk/derivatives/pfm_fslr/atlases')
# 04b / 05 / 08 run for each (names: see config.py; files: config.atlas_path)
REFERENCES = ['gordon17', 'ca19', 'ca19magjudgeDDnr', 'gordon17chaseASDC', 'ca19chaseASDC']

PLOT_DIR = Path('/mnt_AdaBD_largefiles/Data/DNumRisk_Data/ds-asd'
                 '/plots_and_ims/nets_PFM')

# --- Tools ---
WB_COMMAND    = '/home/ubuntu/workbench/bin_linux64/wb_command'
MRIS_CONVERT  = '/home/ubuntu/freesurfer/bin/mris_convert'

# --- Template surfaces ---
HCP_ATLASES  = Path('/home/ubuntu/git/HCPpipelines/global/templates/standard_mesh_atlases')
HCP_RESAMPLE = HCP_ATLASES / 'resample_fsaverage'
NEUROMAPS_FSLR = Path('/home/ubuntu/neuromaps-data/atlases/fsLR')

FSLR_SPHERE = {
    'L': HCP_RESAMPLE / 'fs_LR-deformed_to-fsaverage.L.sphere.32k_fs_LR.surf.gii',
    'R': HCP_RESAMPLE / 'fs_LR-deformed_to-fsaverage.R.sphere.32k_fs_LR.surf.gii',
}
FSLR_MIDTHICK = {
    'L': NEUROMAPS_FSLR / 'tpl-fsLR_den-32k_hemi-L_midthickness.surf.gii',
    'R': NEUROMAPS_FSLR / 'tpl-fsLR_den-32k_hemi-R_midthickness.surf.gii',
}
FSLR_ROI = {
    'L': HCP_ATLASES / 'L.atlasroi.32k_fs_LR.shape.gii',
    'R': HCP_ATLASES / 'R.atlasroi.32k_fs_LR.shape.gii',
}

# --- Vertex-wise CM / Infomap ---
INFOMAP_DENSITIES = [0.003, 0.005, 0.01, 0.02, 0.03, 0.05]
CM_CHUNK_SIZE     = 200
CM_DIST_CUTOFF_MM = 30.0

# --- Acquisition parameters ---
SESSION = 'ses-1'
TASK    = 'chase'
TR      = 2.3338   # seconds

ALL_SUBJECTS = list(range(1, 99))


def get_runs(subject: str) -> list[int]:
    """Run numbers for a subject, detected from confounds files.

    Run count is NOT constant across ds-asd subjects (most have 9 runs of
    task-chase, a handful have 6) — unlike DNumRisk, this can't be a fixed list.
    """
    func_dir = FMRIPREP / subject / SESSION / 'func'
    pattern  = re.compile(rf'{subject}_{SESSION}_task-{TASK}_run-(\d+)_desc-confounds_timeseries\.tsv')
    runs = sorted(
        int(m.group(1)) for f in func_dir.glob(f'*_task-{TASK}_run-*_desc-confounds_timeseries.tsv')
        if (m := pattern.match(f.name))
    )
    return runs


# --- Denoising parameters (Gordon 2017) ---
FD_THRESHOLD = 0.2
BP_LOW       = 0.009
BP_HIGH      = 0.08
COV_SD_THRESH = 0.5

# --- Smoothing ---
SMOOTH_SIGMA = 2.55

# --- Subcortical structures: aparc+aseg label → (cifti_int, CIFTI structure name)
# Bare names (THALAMUS_LEFT) — wb_command silently drops voxels under any other
# name, e.g. with the CIFTI_STRUCTURE_ prefix. Only used with PFM_VARIANT=subcort.
SUBCORTICAL_LABELS = {
    10: (1,  'THALAMUS_LEFT'),
    11: (2,  'CAUDATE_LEFT'),
    12: (3,  'PUTAMEN_LEFT'),
    13: (4,  'PALLIDUM_LEFT'),
    17: (5,  'HIPPOCAMPUS_LEFT'),
    18: (6,  'AMYGDALA_LEFT'),
    26: (7,  'ACCUMBENS_LEFT'),
     8: (8,  'CEREBELLUM_LEFT'),
    49: (9,  'THALAMUS_RIGHT'),
    50: (10, 'CAUDATE_RIGHT'),
    51: (11, 'PUTAMEN_RIGHT'),
    52: (12, 'PALLIDUM_RIGHT'),
    53: (13, 'HIPPOCAMPUS_RIGHT'),
    54: (14, 'AMYGDALA_RIGHT'),
    58: (15, 'ACCUMBENS_RIGHT'),
    47: (16, 'CEREBELLUM_RIGHT'),
    16: (17, 'BRAIN_STEM'),
}


# --- Group membership ---
GROUP_FILE  = BIDS_ROOT / 'derivatives' / 'participants_with_groups.csv'
GROUP_NAMES = {0: 'Control', 1: 'ASD'}


def get_groups() -> dict[str, int]:
    """{'sub-XX': group} from GROUP_FILE (1 = ASD, 0 = Control).

    sub-95..98 are not in the file (no group label) → absent from the dict.
    """
    import csv
    with open(GROUP_FILE) as f:
        return {row['participant_id']: int(row['group']) for row in csv.DictReader(f)}


# --- Excluded subjects (skipped by run_parallel.py and 07_group_avg_cm.py) ---
EXCLUDE_SUBJECTS = {
    59: 'motion: 80-94% frames censored per run, 150/1161 retained (5.8 min); step 03 fails',
}
