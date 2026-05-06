#!/usr/bin/env python3
"""
run_pipeline.py — Full PFM pipeline runner.

Runs steps 01–04b for each subject, logs output per subject/step,
tracks failures, and prints a summary table with retained-frame counts.

Steps:
  01  denoise
  02  surface + CIFTI assembly
  03  vertex-wise correlation matrix
  04  Infomap (Gordon17 reference)
  04b relabel with caNets_DDnr reference

Usage:
  python run_pipeline.py                    # all subjects 1-66
  python run_pipeline.py --subjects 1 2 3
  python run_pipeline.py --start 10         # resume from subject 10
  python run_pipeline.py --steps 03 04 04b  # run only specific steps
"""
import argparse
import subprocess
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR  = Path(__file__).parent
LOG_DIR     = SCRIPT_DIR / 'logs'
ATLAS_DIR   = Path('/mnt_03/ds-dnumrisk/derivatives/pfm_fslr/atlases')
OUTPUT_ROOT = Path('/mnt_03/ds-dnumrisk/derivatives/pfm_fslr')

REF_GORDON = ATLAS_DIR / 'gordon17_space-fsLR_den-32k_cortex.npz'
REF_CANETS = ATLAS_DIR / 'caNets_DDnr-magjudge-task-average-from-fsav5_space-fsLR_den-32k_cortex.npz'

PLOT_DIR    = Path('/mnt_AdaBD_largefiles/Data/SMILE_Data/DNumRisk/ds-dnumrisk'
                   '/plots_and_ims/nets_PFM')

CONDA_ENV   = 'numrefields'
PYTHON      = ['conda', 'run', '--no-capture-output', '-n', CONDA_ENV, 'python', '-u']

ALL_SUBJECTS = list(range(1, 67))
ALL_STEPS    = ['01', '02', '03', '04', '04b', '05']

SESSION = 'ses-1'
TASK    = 'magjudge'
RUNS    = list(range(1, 7))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fmt_sub(n: int) -> str:
    return f'sub-{n:02d}'


def elapsed(t0: float) -> str:
    s = time.perf_counter() - t0
    return f'{s/60:.1f} min' if s >= 60 else f'{s:.1f} s'


def run_step(cmd: list, log_path: Path) -> bool:
    """Run a command, tee output to log_path. Returns True on success."""
    with open(log_path, 'a') as fh:
        fh.write(f'\n{"="*60}\n')
        fh.write(f'CMD: {" ".join(str(c) for c in cmd)}\n')
        fh.write(f'START: {datetime.now():%Y-%m-%d %H:%M:%S}\n')
        fh.write(f'{"="*60}\n')
        fh.flush()
        result = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT)
        fh.write(f'\n{"="*60}\n')
        fh.write(f'END: {datetime.now():%Y-%m-%d %H:%M:%S}  '
                 f'returncode={result.returncode}\n')
    return result.returncode == 0


def get_frame_counts(subject: str) -> dict:
    """
    Load per-run scrub masks and return retained-frame statistics.
    Returns dict with keys: total, retained, fraction. None if masks missing.
    """
    den_dir = OUTPUT_ROOT / subject / 'denoised'
    total = retained = 0
    found = 0
    for run in RUNS:
        stem = f'{subject}_{SESSION}_task-{TASK}_run-{run}'
        mask_path = den_dir / f'{stem}_desc-scrubmask.npy'
        if mask_path.exists():
            mask = np.load(mask_path)
            total    += len(mask)
            retained += mask.sum()
            found    += 1
    if found == 0:
        return {'total': None, 'retained': None, 'fraction': None}
    return {
        'total':    int(total),
        'retained': int(retained),
        'fraction': round(retained / total, 3) if total > 0 else None,
    }


# ---------------------------------------------------------------------------
# Completion checks — key output file(s) per step
# ---------------------------------------------------------------------------

def is_done(subject: str, step: str) -> bool:
    """Return True if the key output for this step already exists."""
    sub_dir = OUTPUT_ROOT / subject
    stem    = f'{subject}_{SESSION}_space-fsLR32k'
    checks  = {
        '01':  all(
            (sub_dir / 'denoised' /
             f'{subject}_{SESSION}_task-{TASK}_run-{r}_desc-scrubmask.npy').exists()
            for r in RUNS
        ) or (sub_dir / 'cifti' / f'{stem}_bold_concat.dtseries.nii').exists(),
        '02':  (sub_dir / 'cifti' /
                f'{stem}_bold_concat.dtseries.nii').exists(),
        '03':  (sub_dir / 'cm' / f'{stem}_cm_meta.npz').exists() and all(
            (sub_dir / 'cm' /
             f'{stem}_density-{f"{d:.3f}".replace("0.", "")}_cm.npz').exists()
            for d in [0.003, 0.005, 0.01, 0.02, 0.03, 0.05]
        ),
        '04':  (sub_dir / 'networks' /
                f'{stem}_consensus_ref-gordon17_communities.npz').exists(),
        '04b': (sub_dir / 'networks' /
                f'{stem}_consensus_ref-caNets_DDnr_communities.npz').exists(),
        '05':  all(
            (PLOT_DIR / f'{subject}_ref-{r}_networks.png').exists()
            for r in ('gordon17', 'caNets_DDnr')
        ),
    }
    return checks[step]


# ---------------------------------------------------------------------------
# Per-step commands
# ---------------------------------------------------------------------------

def step_cmds(subject: str) -> dict:
    n = subject  # already 'sub-XX'
    return {
        '01': PYTHON + [str(SCRIPT_DIR / '01_denoise.py'), n],
        '02': PYTHON + [str(SCRIPT_DIR / '02_surface_cifti.py'), n],
        '03': PYTHON + [str(SCRIPT_DIR / '03_vertex_cm.py'), n],
        '04': PYTHON + [str(SCRIPT_DIR / '04_infomap.py'), n,
                        '--ref', str(REF_GORDON), '--ref-name', 'gordon17'],
        '04b': PYTHON + [str(SCRIPT_DIR / '04b_relabel.py'), n,
                         '--ref', str(REF_CANETS), '--ref-name', 'caNets_DDnr'],
        '05':  PYTHON + [str(SCRIPT_DIR / '05_plot.py'), n],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(subjects: list[int], steps: list[str]) -> None:
    LOG_DIR.mkdir(exist_ok=True)
    t_pipeline = time.perf_counter()

    # Master log
    master_log = LOG_DIR / f'pipeline_{datetime.now():%Y%m%d_%H%M%S}.log'
    print(f'Pipeline log: {master_log}')
    print(f'Subjects: {subjects[0]}–{subjects[-1]} ({len(subjects)} total)')
    print(f'Steps: {steps}\n')

    # Track results: {sub_n: {step: 'ok'|'FAIL'|'skip'}}
    results  = {}
    frames   = {}

    for sub_n in subjects:
        subject = fmt_sub(sub_n)
        log_path = LOG_DIR / f'{subject}_pipeline.log'
        results[sub_n] = {s: '-' for s in ALL_STEPS}
        t_sub = time.perf_counter()

        print(f'[{subject}] starting ...')
        with open(master_log, 'a') as f:
            f.write(f'\n### {subject}  {datetime.now():%Y-%m-%d %H:%M:%S}\n')

        failed = False
        for step in steps:
            if failed and step not in ('04b',):
                # skip downstream steps after a failure, but allow 04b
                # to run independently if 04 succeeded on a prior run
                results[sub_n][step] = 'skip'
                continue

            # For 04b, only skip if 03 failed (it needs the CM outputs)
            if step == '04b' and results[sub_n].get('03') == 'FAIL':
                results[sub_n][step] = 'skip'
                continue

            if is_done(subject, step):
                results[sub_n][step] = 'done'
                print(f'  [{subject}] step {step} ... done (skipping)')
                continue

            cmd = step_cmds(subject)[step]
            print(f'  [{subject}] step {step} ...', end='', flush=True)
            t0 = time.perf_counter()

            ok = run_step(cmd, log_path)
            tag = elapsed(t0)

            if ok:
                results[sub_n][step] = 'ok'
                print(f' ok  [{tag}]')
            else:
                results[sub_n][step] = 'FAIL'
                print(f' FAIL  [{tag}]  → see {log_path.name}')
                if step in ('01', '02', '03'):
                    failed = True

        # Collect frame counts (available after step 01)
        frames[sub_n] = get_frame_counts(subject)

        print(f'[{subject}] done in {elapsed(t_sub)}\n')

    # -----------------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------------
    rows = []
    for sub_n in subjects:
        fc = frames[sub_n]
        row = {'subject': fmt_sub(sub_n)}
        row.update({f'step_{s}': results[sub_n][s] for s in ALL_STEPS})
        row['frames_total']    = fc['total']
        row['frames_retained'] = fc['retained']
        row['frames_fraction'] = fc['fraction']
        rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = LOG_DIR / f'pipeline_summary_{datetime.now():%Y%m%d_%H%M%S}.csv'
    df.to_csv(csv_path, index=False)

    print('\n' + '='*70)
    print('PIPELINE SUMMARY')
    print('='*70)
    print(df.to_string(index=False))
    print(f'\nSaved: {csv_path}')

    n_fail = sum(
        1 for sub_n in subjects
        if any(v == 'FAIL' for v in results[sub_n].values())
    )
    print(f'\nTotal time: {elapsed(t_pipeline)}')
    print(f'Subjects with ≥1 failure: {n_fail}/{len(subjects)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--subjects', nargs='+', type=int, default=None,
                        help='Subject numbers to run (default: 1-66)')
    parser.add_argument('--start', type=int, default=None,
                        help='Resume from this subject number')
    parser.add_argument('--steps', nargs='+', default=ALL_STEPS,
                        choices=ALL_STEPS,
                        help='Steps to run (default: all)')
    args = parser.parse_args()

    if args.subjects:
        subjects = args.subjects
    elif args.start:
        subjects = [n for n in ALL_SUBJECTS if n >= args.start]
    else:
        subjects = ALL_SUBJECTS

    main(subjects, args.steps)
