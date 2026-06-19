#!/usr/bin/env python3
"""
Step 6 — Compute individual vertex areas in fsLR 32k space.

Resamples each subject's native midthickness surface to fsLR 32k using
wb_command -surface-resample (ADAP_BARY_AREA), then derives per-vertex areas.

Output (OUTPUT_ROOT/sub-XX/anat/):
  sub-XX_ses-1_hemi-L_vertex_areas_fsLR32k.shape.gii  — (32492,) float32, mm²
  sub-XX_ses-1_hemi-R_vertex_areas_fsLR32k.shape.gii

Usage:
  python 06_comp_indVertexArea.py sub-01
  python 06_comp_indVertexArea.py sub-01 --force
"""
import sys
import argparse
import subprocess
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import (OUTPUT_ROOT, SESSION, FMRIPREP, FREESURFER,
                    HCP_RESAMPLE, FSLR_SPHERE, FSLR_MIDTHICK, WB_COMMAND)


def compute_subject(subject: str, force: bool = False) -> None:
    anat_dir = OUTPUT_ROOT / subject / 'anat'
    anat_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        for hemi in ('L', 'R'):
            h = hemi.lower()
            out_areas = anat_dir / f'{subject}_{SESSION}_hemi-{hemi}_vertex_areas_fsLR32k.shape.gii'

            if out_areas.exists() and not force:
                print(f'  [skip] {out_areas.name} already exists')
                continue

            native_mid  = FMRIPREP / subject / SESSION / 'anat' / f'{subject}_{SESSION}_hemi-{hemi}_midthickness.surf.gii'
            sphere_reg  = FREESURFER / subject / 'surf' / f'{h}h.sphere.reg.surf.gii'
            target_sph  = FSLR_SPHERE[hemi]
            tpl_mid     = FSLR_MIDTHICK[hemi]

            if not native_mid.exists():
                print(f'  [skip] {subject} hemi-{hemi}: native midthickness not found')
                continue
            if not sphere_reg.exists():
                print(f'  [skip] {subject} hemi-{hemi}: sphere.reg not found '
                      f'(run 02_surface_cifti.py first)')
                continue

            out_mid = Path(tmp) / f'{subject}_hemi-{hemi}_midthickness_fsLR32k.surf.gii'

            subprocess.run(
                [WB_COMMAND, '-surface-resample',
                 str(native_mid), str(sphere_reg), str(target_sph),
                 'ADAP_BARY_AREA', str(out_mid),
                 '-area-surfs', str(native_mid), str(tpl_mid)],
                check=True, capture_output=True)

            subprocess.run(
                [WB_COMMAND, '-surface-vertex-areas', str(out_mid), str(out_areas)],
                check=True, capture_output=True)

            print(f'  Saved → {out_areas.name}')


def parse_subject(s: str) -> str:
    return f'sub-{int(s.removeprefix("sub-")):02d}'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', help='subject id: 1, 01, or sub-01')
    parser.add_argument('--force', action='store_true',
                        help='Recompute even if output already exists')
    args = parser.parse_args()
    subject = parse_subject(args.subject)
    print(f'[{subject}] Computing individual vertex areas ...')
    compute_subject(subject, force=args.force)