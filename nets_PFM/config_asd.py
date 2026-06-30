from pathlib import Path
import re

# --- Data paths ---
BIDS_ROOT   = Path('/mnt_asd/ds-asd')
FMRIPREP    = BIDS_ROOT / 'derivatives' / 'fmriprep'
FREESURFER  = BIDS_ROOT / 'derivatives' / 'freesurfer'
# NOTE: as of 2026-06-27 no derivatives/freesurfer/ exists for this dataset.
# Step 02 (fsnative -> fsLR32k resampling) needs sub-XX/surf/lh.sphere.reg
# from FreeSurfer's recon-all output. The original (FreeSurfer 6.0.1, run
# inside fMRIPrep on the cluster) is inaccessible during cluster maintenance.

# --- Local single-subject FreeSurfer test (temporary, 2026-06-27) ---
# recon-all run locally with FreeSurfer 7.3.2 on sub-01's raw T1w, since the
# cluster's FreeSurfer 6.0.1 dir can't be copied over right now. A locally
# regenerated sphere.reg would NOT have matching vertex topology with
# fMRIPrep's already-exported anat/*.surf.gii (different FreeSurfer version,
# and recon-all's topology-fixing isn't guaranteed reproducible run-to-run
# anyway) — so for this test subject ALL native surfaces (smoothwm, pial,
# midthickness, sphere.reg) are derived from the one local recon-all run,
# overriding fMRIPrep's anat outputs, to keep mesh topology self-consistent.
# Remove these overrides once the real cluster FreeSurfer dirs are copied in.
_LOCAL_TEST_ROOT = Path('/mnt_AdaBD_largefiles/Data/DNumRisk_Data/ds-asd'
                         '/derivatives/freesurfer_local_test_fs7.3.2')
FREESURFER_OVERRIDE = {
    'sub-01': _LOCAL_TEST_ROOT,
}
ANAT_DIR_OVERRIDE = {
    'sub-01': _LOCAL_TEST_ROOT / 'sub-01' / 'anat_gii',
}

# /mnt_asd is nearly full (~28 GB free) — redirect pfm_fslr outputs to the
# largefiles mount instead of writing next to the input data.
OUTPUT_ROOT = Path('/mnt_AdaBD_largefiles/Data/DNumRisk_Data/ds-asd'
                    '/derivatives/pfm_fslr')

# Reference atlases (gordon17, caNets_DDnr) are dataset-independent labelling
# references already in fsLR 32k space — reuse DNumRisk's copy rather than
# duplicating them here.
ATLAS_DIR = Path('/mnt_03/ds-dnumrisk/derivatives/pfm_fslr/atlases')

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

# --- Network label names by reference atlas ---
ATLAS_NETWORK_NAMES = {
    'gordon17': {
        0:  'Unassigned',
        1:  'Default',
        2:  'LatVis',
        3:  'FrontPar',
        4:  'MedVis',
        5:  'DorsAttn',
        6:  'Premotor',
        7:  'Language',
        8:  'Salience',
        9:  'CingOperc',
        10: 'HandSM',
        11: 'FaceSM',
        12: 'Auditory',
        13: 'AntMTL',
        14: 'PostMTL',
        15: 'ParMemory',
        16: 'Context',
        17: 'FootSM',
    },
    'caNets_DDnr': {
        0:  'Unassigned',
        1:  'Visual1',
        2:  'Visual2',
        3:  'Somatomotor',
        4:  'Cingulo-Opercular',
        5:  'Dorsal-attention',
        6:  'Language',
        7:  'Frontoparietal',
        8:  'Auditory',
        9:  'Default',
        10: 'Posterior-Multimodal',
        11: 'Ventral-Multimodal',
        12: 'Orbito-Affective',
    },
}

# --- Subcortical structures: aparc+aseg label → (cifti_int, CIFTI_structure_name)
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
