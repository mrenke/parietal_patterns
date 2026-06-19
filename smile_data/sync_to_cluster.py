#!/usr/bin/env python3
"""
Sync a batch of newly converted BIDS sessions to the science cluster.

Reads bids_conversion_log.csv to determine which sessions belong to the
target batch, then rsyncs each sub-X/ses-X directory individually.
Root-level BIDS metadata files are also synced.

Usage:
    python sync_to_cluster.py              # sync the most recent batch
    python sync_to_cluster.py 2026-06-03   # sync a specific date
    python sync_to_cluster.py --dry-run    # preview without transferring
    python sync_to_cluster.py --metadata   # also sync root-level BIDS files
    python sync_to_cluster.py --batch      # write fmriprep_batch_<date>.txt for SLURM array
"""

import csv
import os
import subprocess
import sys
from pathlib import Path

# ── config ────────────────────────────────────────────────────────────────────

REMOTE     = 'sciencecluster:/shares/zne.uzh/mrenke/ds-smile1'
LOCAL_BIDS = '/mnt_AdaBD_largefiles/Data/DNumRisk_Data/ds-smile'
CSV_LOG    = Path(__file__).parent / 'bids_conversion_log.csv'

BIDS_ROOT_FILES = [
    'dataset_description.json',
    'task-magjudge_bold.json',
    'task-placevalue_bold.json',
    'task-rest_bold.json',
    '.bidsignore',
    'participants.tsv',
    'participants.json',
]

# ── helpers ───────────────────────────────────────────────────────────────────

def read_log():
    batches = {}
    with open(CSV_LOG) as f:
        for row in csv.DictReader(f):
            batches.setdefault(row['date_converted'], []).append(
                (row['subject'], row['session'])
            )
    return batches

def ensure_remote_dir(dst, dry_run=False):
    """Create the remote directory via ssh before rsyncing into it."""
    if ':' not in dst:
        return  # local path, no ssh needed
    host, path = dst.split(':', 1)
    path = path.rstrip('/')
    cmd = ['ssh', host, f'mkdir -p "{path}"']
    if not dry_run:
        subprocess.run(cmd, check=True)

def rsync(src, dst, dry_run=False):
    ensure_remote_dir(dst, dry_run)
    cmd = ['rsync', '-rcvz', src, dst]
    if dry_run:
        cmd.insert(1, '--dry-run')
    print('  ' + ' '.join(cmd))
    subprocess.run(cmd, check=True)

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    dry_run   = '--dry-run'  in sys.argv
    metadata  = '--metadata' in sys.argv
    write_batch = '--batch'  in sys.argv
    date_arg  = next((a for a in sys.argv[1:] if not a.startswith('--')), None)

    batches = read_log()
    dated   = sorted(d for d in batches if d != 'unknown')
    if not dated:
        sys.exit('No dated batches found in log.')

    target = date_arg or dated[-1]
    if target not in batches:
        sys.exit(f'No sessions found for date "{target}". Available: {", ".join(dated)}')

    sessions = sorted(batches[target])
    print(f'\nBatch {target}: {len(sessions)} session(s) → {REMOTE}')
    if dry_run:
        print('[DRY RUN — no data will be transferred]')

    for sub, ses in sessions:
        print(f'\n  {sub} {ses}')
        rsync(f'{LOCAL_BIDS}/{sub}/{ses}/', f'{REMOTE}/{sub}/{ses}/', dry_run)

    if metadata:
        print('\n  [root-level BIDS files]')
        for fname in BIDS_ROOT_FILES:
            fpath = f'{LOCAL_BIDS}/{fname}'
            if os.path.exists(fpath):
                rsync(fpath, f'{REMOTE}/', dry_run)

    if write_batch:
        batch_file = Path(__file__).parent / f'fmriprep_batch_{target}.txt'
        with open(batch_file, 'w') as f:
            for sub, ses in sessions:
                # strip "sub-" and "ses-" prefixes — fmriprep wants bare IDs
                f.write(f'{sub[4:]} {ses[4:]}\n')
        print(f'\nBatch file written: {batch_file}  ({len(sessions)} lines)')
        print('Submit with:  sbatch --array=1-'
              f'{len(sessions)} fmriprep_sessions.sh {batch_file.name}')

    print('\nDone.')

if __name__ == '__main__':
    main()
