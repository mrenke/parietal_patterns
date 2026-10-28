from pathlib import Path

# --- Data paths ---
BIDS_ROOT   = Path('/mnt_03/ds-dnumrisk')
FMRIPREP    = BIDS_ROOT / 'derivatives' / 'fmriprep'
FREESURFER  = BIDS_ROOT / 'derivatives' / 'freesurfer'

# Per-subject overrides for FREESURFER root / fMRIPrep anat surf dir — used
# by config_asd.py for a local single-subject FreeSurfer test. Not needed here.
FREESURFER_OVERRIDE = {}
ANAT_DIR_OVERRIDE   = {}

# Output root — change BIDS_ROOT_OUT if the input volume runs low on disk.
# Folder structure mirrors the input BIDS tree (derivatives/pfm_fslr/sub-XX/...).
BIDS_ROOT_OUT = BIDS_ROOT   # e.g. Path('/mnt_04/ds-dnumrisk') if redirecting
OUTPUT_ROOT   = BIDS_ROOT_OUT / 'derivatives' / 'pfm_fslr'
ATLAS_DIR     = OUTPUT_ROOT / 'atlases'
# 04b / 05 / 08 run for each (names: see config.py; files: config.atlas_path)
REFERENCES    = ['gordon17', 'ca19magjudgeDDnr']

PLOT_DIR = Path('/mnt_AdaBD_largefiles/Data/SMILE_Data/DNumRisk/ds-dnumrisk'
                 '/plots_and_ims/nets_PFM')

# --- Tools ---
WB_COMMAND    = '/home/ubuntu/workbench/bin_linux64/wb_command'
MRIS_CONVERT  = '/home/ubuntu/freesurfer/bin/mris_convert'

# --- Template surfaces ---
HCP_ATLASES  = Path('/home/ubuntu/git/HCPpipelines/global/templates/standard_mesh_atlases')
HCP_RESAMPLE = HCP_ATLASES / 'resample_fsaverage'
NEUROMAPS_FSLR = Path('/home/ubuntu/neuromaps-data/atlases/fsLR')

# fsLR 32k target sphere (expressed in fsaverage coordinates)
FSLR_SPHERE = {
    'L': HCP_RESAMPLE / 'fs_LR-deformed_to-fsaverage.L.sphere.32k_fs_LR.surf.gii',
    'R': HCP_RESAMPLE / 'fs_LR-deformed_to-fsaverage.R.sphere.32k_fs_LR.surf.gii',
}
# fsLR 32k template midthickness (for area correction in resampling)
FSLR_MIDTHICK = {
    'L': NEUROMAPS_FSLR / 'tpl-fsLR_den-32k_hemi-L_midthickness.surf.gii',
    'R': NEUROMAPS_FSLR / 'tpl-fsLR_den-32k_hemi-R_midthickness.surf.gii',
}
# Medial wall ROI masks — used to exclude non-cortical vertices from CIFTI
FSLR_ROI = {
    'L': HCP_ATLASES / 'L.atlasroi.32k_fs_LR.shape.gii',
    'R': HCP_ATLASES / 'R.atlasroi.32k_fs_LR.shape.gii',
}

# --- Vertex-wise CM / Infomap ---
# Density thresholds: fraction of ALL n*(n-1)/2 pairs kept as edges
INFOMAP_DENSITIES = [0.003, 0.005, 0.01, 0.02, 0.03, 0.05]
CM_CHUNK_SIZE     = 200    # nodes per chunk during correlation computation
CM_DIST_CUTOFF_MM = 30.0   # local connections zeroed within this distance

# --- Acquisition parameters ---
SESSION = 'ses-1'
TASK    = 'magjudge'
TR      = 2.298   # seconds

ALL_SUBJECTS = list(range(1, 67))


def get_runs(subject: str) -> list[int]:
    """Run numbers for a subject. Fixed at 6 for all DNumRisk subjects."""
    return list(range(1, 7))


# --- Denoising parameters (Gordon 2017) ---
FD_THRESHOLD = 0.2   # mm — frames above this are censored
BP_LOW       = 0.009  # Hz
BP_HIGH      = 0.08   # Hz
COV_SD_THRESH = 0.5   # SDs above local mean CoV → voxel excluded

# --- Smoothing ---
SMOOTH_SIGMA = 2.55   # mm (geodesic on surface, Euclidean on volume)

# --- Subcortical structures: aparc+aseg label → (cifti_int, CIFTI structure name)
# Bare names (THALAMUS_LEFT) — wb_command silently drops voxels under any other
# name, e.g. with the CIFTI_STRUCTURE_ prefix. Only used with PFM_VARIANT=subcort.
# Cerebellar cortex only (8=L, 47=R). WM (7, 46) excluded for simplicity.
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
GROUP_FILE  = BIDS_ROOT / 'group_assignment.csv'
GROUP_NAMES = {0: 'TD', 1: 'DD'}


def get_groups() -> dict[str, int]:
    """{'sub-XX': group} from GROUP_FILE (1 = DD / dyscalculia, 0 = TD / control)."""
    import csv
    with open(GROUP_FILE) as f:
        return {f"sub-{int(row['subject']):02d}": int(row['group'])
                for row in csv.DictReader(f)}


# --- Excluded subjects (skipped by run_parallel.py and 07_group_avg_cm.py) ---
# sub-05: only run-5 unusable — PFM redone on 5 runs (2026-07-15), kept in.
EXCLUDE_SUBJECTS = {}
