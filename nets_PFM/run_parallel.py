#!/usr/bin/env python3
"""
run_parallel.py — run run_pipeline.py for many subjects concurrently.

run_pipeline.py processes subjects one after another; that is fine for steps
01–03 but makes step 04 (Infomap, ~75 min/subject, single-threaded) take days.
This launcher starts one run_pipeline.py process per subject, with at most
--jobs running at once. Each process keeps run_pipeline's completion checks,
so finished steps are skipped and the launcher can simply be re-run to resume.

Memory, not CPU, is the limit: Infomap at 5% density (~88M edges) peaked at
~21 GB before the fix in 04_infomap.py (OOM-killed on the 32 GB VM). The
default --jobs is therefore derived from total RAM / --gb-per-job. Starts are
staggered so the 5%-density peaks (last density, reached ~40 min in) don't
coincide.

Usage:
  python run_parallel.py --dataset asd                       # all subjects, steps 04 04b 05
  python run_parallel.py --dataset asd --jobs 12 --start 10
  python run_parallel.py --dataset asd --subjects 1 2 3 --steps 04
"""
import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from threading import Lock

SCRIPT_DIR = Path(__file__).parent


def mem_total_gb() -> float:
    with open('/proc/meminfo') as f:
        for line in f:
            if line.startswith('MemTotal:'):
                return int(line.split()[1]) / 1024**2
    raise RuntimeError('MemTotal not found in /proc/meminfo')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default=os.environ.get('PFM_DATASET', 'dnumrisk'))
    parser.add_argument('--subjects', nargs='+', type=int, default=None)
    parser.add_argument('--start', type=int, default=None,
                        help='Resume from this subject number')
    parser.add_argument('--steps', nargs='+', default=['04', '04b', '05'])
    parser.add_argument('--gb-per-job', type=float, default=20.0,
                        help='Peak RAM to budget per subject (default 20 GB, Infomap at 5%%)')
    parser.add_argument('--jobs', type=int, default=None,
                        help='Max concurrent subjects (default: from RAM and CPU count)')
    parser.add_argument('--stagger', type=float, default=300,
                        help='Seconds between the first --jobs launches (default 300)')
    args = parser.parse_args()

    os.environ['PFM_DATASET'] = args.dataset
    sys.path.insert(0, str(SCRIPT_DIR))
    from config import ALL_SUBJECTS, EXCLUDE_SUBJECTS

    if args.subjects:
        subjects = args.subjects
    elif args.start:
        subjects = [n for n in ALL_SUBJECTS if n >= args.start]
    else:
        subjects = ALL_SUBJECTS
    for n in sorted(set(subjects) & set(EXCLUDE_SUBJECTS)):
        print(f'Excluding sub-{n:02d} (config): {EXCLUDE_SUBJECTS[n]}')
    subjects = [n for n in subjects if n not in EXCLUDE_SUBJECTS]

    ram = mem_total_gb()
    n_cpu = os.cpu_count() or 1
    jobs = args.jobs or max(1, min(n_cpu - 1, int(0.85 * ram / args.gb_per_job)))

    log_dir = SCRIPT_DIR / 'logs' / args.dataset / 'parallel'
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f'{datetime.now():%Y-%m-%d %H:%M:%S}  dataset={args.dataset}  '
          f'subjects={len(subjects)}  steps={args.steps}')
    print(f'RAM {ram:.0f} GB, {n_cpu} CPUs → {jobs} concurrent jobs '
          f'({args.gb_per_job:g} GB/job budget)\n', flush=True)

    # Stagger only the initial wave; later jobs start as earlier ones finish,
    # which naturally keeps them out of phase.
    launch_lock = Lock()
    launched = [0]

    def run_one(sub_n: int) -> tuple[int, int, float]:
        with launch_lock:
            wave_pos = launched[0]
            launched[0] += 1
        if wave_pos < jobs:
            time.sleep(wave_pos * args.stagger)
        out = log_dir / f'sub-{sub_n:02d}_runner.out'
        cmd = [sys.executable, '-u', str(SCRIPT_DIR / 'run_pipeline.py'),
               '--dataset', args.dataset, '--subjects', str(sub_n),
               '--steps', *args.steps]
        t0 = time.perf_counter()
        print(f'{datetime.now():%H:%M:%S}  start sub-{sub_n:02d}', flush=True)
        with open(out, 'w') as fh:
            rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT).returncode
        failed = rc != 0 or ' FAIL ' in out.read_text()
        return sub_n, int(failed), (time.perf_counter() - t0) / 60

    failures = []
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = [pool.submit(run_one, n) for n in subjects]
        for i, fut in enumerate(as_completed(futures), 1):
            sub_n, failed, minutes = fut.result()
            tag = 'FAIL' if failed else 'ok'
            if failed:
                failures.append(sub_n)
            print(f'{datetime.now():%H:%M:%S}  [{i}/{len(subjects)}] sub-{sub_n:02d} '
                  f'{tag}  ({minutes:.0f} min)', flush=True)

    print(f'\nDone. Failures: {sorted(failures) or "none"}')
    if failures:
        print(f'See {log_dir}/sub-XX_runner.out and logs/{args.dataset}/sub-XX_pipeline.log')


if __name__ == '__main__':
    main()
