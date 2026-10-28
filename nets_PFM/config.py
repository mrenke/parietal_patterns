"""
Active dataset config — selected via the PFM_DATASET environment variable.

Usage:
  PFM_DATASET=asd python 01_denoise.py sub-01     # ds-asd
  python 01_denoise.py sub-01                     # defaults to dnumrisk

All other scripts do `from config import X` and get whichever dataset's
config module is active; no other code needs to change when switching.
"""
import os

_DATASETS = {
    'dnumrisk': 'config_dnumrisk',
    'asd':      'config_asd',
}

_dataset = os.environ.get('PFM_DATASET', 'dnumrisk')
if _dataset not in _DATASETS:
    raise ValueError(
        f'Unknown PFM_DATASET={_dataset!r}. Choose one of: {list(_DATASETS)}'
    )

from importlib import import_module as _import_module
_module = _import_module(_DATASETS[_dataset])

# Re-export everything from the active dataset config
globals().update({k: v for k, v in vars(_module).items() if not k.startswith('_')})

# --- Pipeline variant — selected via the PFM_VARIANT environment variable ---
#   (unset)  default: cortex-only CIFTIs (59,412 grayordinates; see the Step 4
#            note in CLAUDE.md for why no subcortical voxels end up in them)
#   subcort  subcortical + cerebellar voxels included in the CIFTI (step 02 on).
#            Reuses the default run's step-01 outputs (DENOISED_ROOT); everything
#            from step 02 on goes to <OUTPUT_ROOT>_subcort (and <PLOT_DIR>_subcort).
#   PFM_DATASET=asd PFM_VARIANT=subcort python 02_surface_cifti.py sub-01
_VARIANTS = {'', 'subcort'}
VARIANT = os.environ.get('PFM_VARIANT', '')
if VARIANT not in _VARIANTS:
    raise ValueError(f'Unknown PFM_VARIANT={VARIANT!r}. Choose one of: {sorted(_VARIANTS)}')

BASE_OUTPUT_ROOT    = OUTPUT_ROOT   # default (cortex-only) run, whatever VARIANT is
DENOISED_ROOT       = BASE_OUTPUT_ROOT
INCLUDE_SUBCORTICAL = VARIANT == 'subcort'
if VARIANT:
    OUTPUT_ROOT = OUTPUT_ROOT.with_name(f'{OUTPUT_ROOT.name}_{VARIANT}')
    PLOT_DIR    = PLOT_DIR.with_name(f'{PLOT_DIR.name}_{VARIANT}')


# --- Reference atlases (labelling in 04b) -----------------------------------
# Ref names: <label set><task><data source>, letters/digits only (BIDS label value):
#   gordon17            Gordon et al. 2017 group networks
#   ca19                Cole-Anticevic network partition (Ji et al. 2019, CAB-NP)
#   ca19magjudgeDDnr    DNumRisk magjudge group average (fsav5), ca19-labelled
#   gordon17chaseASDC   ds-asd chase group average (ASD + Control), gordon17-labelled
# File: atlas_path(ref). The label set is the name prefix → network names / colormap.
LABEL_SETS = ('gordon17', 'ca19')


def label_set(ref: str) -> str:
    """'ca19magjudgeDDnr' → 'ca19'."""
    for base in LABEL_SETS:
        if ref.startswith(base):
            return base
    raise ValueError(f'Reference {ref!r} does not start with a label set {LABEL_SETS}')


def atlas_path(ref: str):
    return ATLAS_DIR / f'{ref}_space-fsLR_den-32k_cortex.npz'


class _NamesByLabelSet(dict):
    """ATLAS_NETWORK_NAMES[ref] for any ref name, via its label-set prefix."""
    def __missing__(self, ref):
        return self[label_set(ref)]


# --- Network label names by label set (used in 08 and the analysis notebooks) ---
ATLAS_NETWORK_NAMES = _NamesByLabelSet({
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
    'ca19': {
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
})
