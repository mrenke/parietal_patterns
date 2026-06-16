"""
SMILE raw data → BIDS conversion

Prerequisites — unzip raw files once before running:
    cd /mnt_04/ds-smile/sourcedata/measurements/mri_files
    target_folder="/mnt_AdaBD_largefiles/Data/DNumRisk_Data/ds-smile/sourcedata"
    for zipfile in *.zip; do
        foldername="${target_folder}/${zipfile%.zip}"
        if [ -d "$foldername" ]; then
            echo "skipping ${zipfile%.zip} — already exists"
            continue
        fi
        mkdir -p "$foldername"
        unzip "$zipfile" -d "$foldername"
    done

    for zipfile in *.zip; do
        foldername="${target_folder}/${zipfile%.zip}"
        if [ -d "$foldername" ]; then
            echo "skipping ${zipfile%.zip} — already exists"
            continue
        fi
        mkdir -p "$foldername"
        unzip "$zipfile" -d "$foldername"
    done

Steps performed:
    1. copy & rename source files into BIDS structure (skips existing sessions)
    2. concatenate 3D volumes → 4D NIfTI where needed
    3. fix TR in NIfTI headers (REPETITION_TIME_MISMATCH)

Validate afterwards:
    deno run -ERN jsr:@bids/validator /mnt_AdaBD_largefiles/Data/DNumRisk_Data/ds-smile
    /mnt_04/ds-smile

Known scanner TRs (confirmed from sub-101):
    task-magjudge  : 2.6 s
    task-rest      : 2.035 s

Known edge case — duplicate source files (sub-202 style):
    Some sessions contain two copies of the same scan differing only by a
    trailing 'a' suffix (_GR.nii vs _GRa.nii).  They are confirmed exact
    duplicates.  The conversion keeps the alphabetically-first file per
    (scan_order, extension) pair.
"""

import csv
import os
import re
import shutil
from datetime import date

import nibabel as nib
import numpy as np

# ── config ────────────────────────────────────────────────────────────────────

SOURCE_PATH = '/mnt_AdaBD_largefiles/Data/DNumRisk_Data/ds-smile/sourcedata' #'/mnt_04/ds-smile/sourcedata/measurements/mri_files'
BIDS_PATH   = '/mnt_AdaBD_largefiles/Data/DNumRisk_Data/ds-smile'

# CSV log lives next to this script; one row per converted session
CSV_LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bids_conversion_log.csv')

TR = {
    'magjudge':   2.6,
    'rest':       2.035,
    'placevalue': 2.7,   # confirmed from task-placevalue_bold.json (different protocol from magjudge)
}

TASK_MAPPING = {
    'NumRisk_R1': 'task-magjudge_run-1',
    'NumRisk_R2': 'task-magjudge_run-2',
    'NumRisk_R3': 'task-magjudge_run-3',
    'PlaceValue': 'task-placevalue_run-1',
    'rsfMRI':     'task-rest_run-1',
    'T1':         'T1w',
}

# ── CSV batch log ─────────────────────────────────────────────────────────────

def _log_conversion(sub_id, ses_id):
    """Append one row to the conversion log CSV (creates header on first write)."""
    write_header = not os.path.exists(CSV_LOG)
    with open(CSV_LOG, 'a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(['subject', 'session', 'date_converted'])
        writer.writerow([f'sub-{sub_id}', f'ses-{ses_id}', date.today().isoformat()])

# ── helpers ───────────────────────────────────────────────────────────────────

def _func_dir(sub_id, ses_id):
    return os.path.join(BIDS_PATH, f'sub-{sub_id}', f'ses-{ses_id}', 'func')

def _make_dir(sub_id, ses_id, modality):
    p = os.path.join(BIDS_PATH, f'sub-{sub_id}', f'ses-{ses_id}', modality)
    os.makedirs(p, exist_ok=True)
    return p

def _already_converted(sub_id, ses_id):
    d = _func_dir(sub_id, ses_id)
    return os.path.isdir(d) and any(f.endswith('.nii') for f in os.listdir(d))

def _smile_entries():
    for entry in sorted(os.scandir(SOURCE_PATH), key=lambda x: x.name):
        if not entry.is_dir():
            continue
        m = re.match(r'SMILE(\d{3})_d(\d+)', entry.name)
        if m:
            yield entry, m.group(1), m.group(2)

# ── step 1: BIDS conversion ───────────────────────────────────────────────────

def step1_convert():
    print('\n=== Step 1: BIDS conversion ===')
    for entry, sub_id, ses_id in _smile_entries():
        if _already_converted(sub_id, ses_id):
            print(f'  sub-{sub_id} ses-{ses_id}  already in BIDS — skipping')
            continue

        for mod_folder in os.scandir(entry.path):
            if not mod_folder.is_dir():
                continue
            bids_label = TASK_MAPPING.get(mod_folder.name)
            if bids_label is None:
                continue

            modality = 'func' if 'task' in bids_label else 'anat'
            dest_dir  = _make_dir(sub_id, ses_id, modality)

            # keep only the first file per (scan_order, extension) to handle
            # duplicate scans (e.g. _GR.nii + _GRa.nii — confirmed exact copies)
            seen = {}
            for f in sorted(os.scandir(mod_folder.path), key=lambda x: x.name):
                if not f.is_file():
                    continue
                m = re.match(r'(\d{3})_.*\.(nii|json)$', f.name)
                if not m:
                    continue
                key = (m.group(1), m.group(2))  # (scan_order, extension)
                if key not in seen:
                    seen[key] = f.path

            for (order, ext), src in seen.items():
                if bids_label == 'T1w':
                    bids_name = f'sub-{sub_id}_ses-{ses_id}_T1w.{ext}'
                else:
                    bids_name = f'sub-{sub_id}_ses-{ses_id}_{bids_label}_bold.{ext}'
                shutil.copy(src, os.path.join(dest_dir, bids_name))

        _log_conversion(sub_id, ses_id)
        print(f'  sub-{sub_id} ses-{ses_id}  converted')

# ── step 2: 3D → 4D concatenation ────────────────────────────────────────────

def step2_concat_3d():
    print('\n=== Step 2: 3D → 4D concatenation ===')
    for entry, sub_id, ses_id in _smile_entries():
        for mod_folder in os.scandir(entry.path):
            if not mod_folder.is_dir():
                continue
            bids_label = TASK_MAPPING.get(mod_folder.name)
            if bids_label is None or 'task' not in bids_label:
                continue

            bids_file = os.path.join(_func_dir(sub_id, ses_id),
                                     f'sub-{sub_id}_ses-{ses_id}_{bids_label}_bold.nii')
            if not os.path.exists(bids_file):
                continue

            img = nib.load(bids_file)
            if img.ndim == 4:
                continue  # already 4D

            src_niis = sorted(f for f in os.listdir(mod_folder.path) if f.endswith('.nii'))
            if len(src_niis) < 2:
                continue

            ref     = nib.load(os.path.join(mod_folder.path, src_niis[0]))
            volumes = [nib.load(os.path.join(mod_folder.path, f)).get_fdata() for f in src_niis]
            nib.save(nib.Nifti1Image(np.stack(volumes, axis=-1), ref.affine, ref.header),
                     bids_file)
            print(f'  sub-{sub_id} ses-{ses_id} {bids_label}  '
                  f'concatenated {len(src_niis)} volumes → 4D')

# ── step 3: fix TR ────────────────────────────────────────────────────────────

TASK_RUNS = [('magjudge', 3), ('placevalue', 1), ('rest', 1)]

def step3_fix_tr():
    print('\n=== Step 3: Fix TR in NIfTI headers and JSON sidecars ===')
    for sub_dir in sorted(os.scandir(BIDS_PATH), key=lambda x: x.name):
        if not sub_dir.name.startswith('sub-'):
            continue
        sub_id = sub_dir.name[4:]

        for ses_dir in sorted(os.scandir(sub_dir.path), key=lambda x: x.name):
            if not ses_dir.name.startswith('ses-'):
                continue
            ses_id   = ses_dir.name[4:]
            func_dir = os.path.join(ses_dir.path, 'func')
            if not os.path.isdir(func_dir):
                continue

            for task, n_runs in TASK_RUNS:
                tr_correct = TR[task]
                for run in range(1, n_runs + 1):
                    stem = f'sub-{sub_id}_ses-{ses_id}_task-{task}_run-{run}_bold'
                    nii  = os.path.join(func_dir, f'{stem}.nii')
                    jsn  = os.path.join(func_dir, f'{stem}.json')
                    if not os.path.exists(nii):
                        continue

                    img       = nib.load(nii, mmap=False)  # mmap=True causes bus error on in-place save
                    tr_actual = float(img.header.get_zooms()[3])
                    if not np.isclose(tr_actual, tr_correct, rtol=1e-3):
                        img.header['pixdim'][4] = tr_correct
                        nib.save(img, nii)
                        print(f'  sub-{sub_id} ses-{ses_id} {task} run-{run}  '
                              f'NIfTI TR fixed: {tr_actual:.4f} → {tr_correct}')

                    if os.path.exists(jsn):
                        import json as _json
                        with open(jsn) as f:
                            meta = _json.load(f)
                        tr_json = meta.get('RepetitionTime')
                        if tr_json is not None and not np.isclose(tr_json, tr_correct, rtol=1e-3):
                            meta['RepetitionTime'] = tr_correct
                            with open(jsn, 'w') as f:
                                _json.dump(meta, f, indent='\t')
                                f.write('\n')
                            print(f'  sub-{sub_id} ses-{ses_id} {task} run-{run}  '
                                  f'JSON TR fixed: {tr_json:.4f} → {tr_correct}')

# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    step1_convert()
    step2_concat_3d()
    step3_fix_tr()
    print('\nDone. Run BIDS validator to check:')
    print('  deno run -ERN jsr:@bids/validator /mnt_04/ds-smile')
