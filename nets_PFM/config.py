from pathlib import Path

# --- Data paths ---
BIDS_ROOT   = Path('/mnt_03/ds-dnumrisk')
FMRIPREP    = BIDS_ROOT / 'derivatives' / 'fmriprep'
FREESURFER  = BIDS_ROOT / 'derivatives' / 'freesurfer'

# Output root — change BIDS_ROOT_OUT if the input volume runs low on disk.
# Folder structure mirrors the input BIDS tree (derivatives/pfm_fslr/sub-XX/...).
BIDS_ROOT_OUT = BIDS_ROOT   # e.g. Path('/mnt_04/ds-dnumrisk') if redirecting
OUTPUT_ROOT   = BIDS_ROOT_OUT / 'derivatives' / 'pfm_fslr'

# --- Tools ---
WB_COMMAND = '/home/ubuntu/workbench/bin_linux64/wb_command'

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
RUNS    = list(range(1, 7))
TR      = 2.298   # seconds

# --- Denoising parameters (Gordon 2017) ---
FD_THRESHOLD = 0.2   # mm — frames above this are censored
BP_LOW       = 0.009  # Hz
BP_HIGH      = 0.08   # Hz
COV_SD_THRESH = 0.5   # SDs above local mean CoV → voxel excluded

# --- Smoothing ---
SMOOTH_SIGMA = 2.55   # mm (geodesic on surface, Euclidean on volume)

# --- Subcortical structures: aparc+aseg label → (cifti_int, CIFTI_structure_name)
# Cerebellar cortex only (8=L, 47=R). WM (7, 46) excluded for simplicity.
SUBCORTICAL_LABELS = {
    10: (1,  'CIFTI_STRUCTURE_THALAMUS_LEFT'),
    11: (2,  'CIFTI_STRUCTURE_CAUDATE_LEFT'),
    12: (3,  'CIFTI_STRUCTURE_PUTAMEN_LEFT'),
    13: (4,  'CIFTI_STRUCTURE_PALLIDUM_LEFT'),
    17: (5,  'CIFTI_STRUCTURE_HIPPOCAMPUS_LEFT'),
    18: (6,  'CIFTI_STRUCTURE_AMYGDALA_LEFT'),
    26: (7,  'CIFTI_STRUCTURE_ACCUMBENS_LEFT'),
     8: (8,  'CIFTI_STRUCTURE_CEREBELLUM_LEFT'),
    49: (9,  'CIFTI_STRUCTURE_THALAMUS_RIGHT'),
    50: (10, 'CIFTI_STRUCTURE_CAUDATE_RIGHT'),
    51: (11, 'CIFTI_STRUCTURE_PUTAMEN_RIGHT'),
    52: (12, 'CIFTI_STRUCTURE_PALLIDUM_RIGHT'),
    53: (13, 'CIFTI_STRUCTURE_HIPPOCAMPUS_RIGHT'),
    54: (14, 'CIFTI_STRUCTURE_AMYGDALA_RIGHT'),
    58: (15, 'CIFTI_STRUCTURE_ACCUMBENS_RIGHT'),
    47: (16, 'CIFTI_STRUCTURE_CEREBELLUM_RIGHT'),
    16: (17, 'CIFTI_STRUCTURE_BRAIN_STEM'),
}
