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

DENOISED_ROOT       = OUTPUT_ROOT
INCLUDE_SUBCORTICAL = VARIANT == 'subcort'
if VARIANT:
    OUTPUT_ROOT = OUTPUT_ROOT.with_name(f'{OUTPUT_ROOT.name}_{VARIANT}')
    PLOT_DIR    = PLOT_DIR.with_name(f'{PLOT_DIR.name}_{VARIANT}')
