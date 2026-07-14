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
