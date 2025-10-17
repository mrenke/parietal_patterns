import argparse
import os
import os.path as op
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn import signal
from nilearn.connectome import ConnectivityMeasure

runs_per_sub_thresh = 2  # Mindestzahl gültiger Runs

def cleanTS(sub, fmriprep_confounds_include, bids_folder,
            TR=2.3, ses="1", task="magjudge",
            runs=range(1, 7), space="fsaverage5",
            scrubbing=True, scrub_thresh=0.3,
            run_FD_filter=True, frames_per_run_thresh=104,
            bp_filtering=True, lower_bpf=0.01, upper_bpf=0.08):

    fmriprep_folder = op.join(bids_folder, "derivatives", "fmriprep",
                              f"sub-{sub}", f"ses-{ses}", "func")

    # Anzahl Vertices je nach Space (Links+Rechts total)
    if space == "fsaverage5":
        number_of_vertices = 20484
    elif space == "fsaverage":
        number_of_vertices = 327684
    else:
        raise ValueError(f"Unbekannter space: {space}")

    clean_per_run = []  # Liste, später entlang Zeit (axis=1) konkatenieren
    N_valid_runs = 0

    for run in runs:
        try:
            # 1) fMRI-Daten je Hemisphäre laden → (T, V_hemi)
            ts_hemi = []
            for hemi in ["L", "R"]:
                filename = op.join(
                    fmriprep_folder,
                    f"sub-{sub}_ses-{ses}_task-{task}_run-{run}_space-{space}_hemi-{hemi}_bold.func.gii"
                )
                print("Loading:", filename)
                ts = nib.load(filename).agg_data()  # (T, V_hemi)
                ts_hemi.append(ts)

            # Hemis zusammen: (T, V_total)
            timeseries = np.hstack(ts_hemi)

            # 2) Confounds robust laden
            confound_file = op.join(
                fmriprep_folder,
                f"sub-{sub}_ses-{ses}_task-{task}_run-{run}_desc-confounds_timeseries.tsv"
            )
            # Verfügbare Spalten prüfen
            with pd.read_table(confound_file, nrows=1) as _tmp:
                pass  # nur Vorab-Check
            all_cols = pd.read_table(confound_file, nrows=1).columns
            use_cols = [c for c in fmriprep_confounds_include if c in all_cols]
            if len(use_cols) == 0:
                print(f"Warnung: keine der gewünschten Confounds gefunden in {confound_file}")
                confounds = None
            else:
                confounds = pd.read_table(confound_file, usecols=use_cols).bfill().ffill()

            # 3) Scrubbing-Maske
            sample_mask = None
            if scrubbing:
                fd = pd.read_table(confound_file, usecols=["framewise_displacement"]).fillna(0.0)["framewise_displacement"].to_numpy()
                sample_mask = (fd < scrub_thresh)
                usable_frames = int(sample_mask.sum())
                if run_FD_filter and usable_frames < frames_per_run_thresh:
                    print(f"Run {run}: nur {usable_frames} brauchbare Frames → Skip.")
                    continue

            # 4) Cleaning: nilearn.signal.clean erwartet (T, V)
            clean_ts = signal.clean(
                timeseries,
                confounds=confounds,
                sample_mask=sample_mask,
                t_r=TR,
                standardize="zscore_sample",
                low_pass=upper_bpf if bp_filtering else None,
                high_pass=lower_bpf if bp_filtering else None
            ).T  # → (V, T)

            clean_per_run.append(clean_ts)
            N_valid_runs += 1

        except Exception as e:
            print(f"Error in sub-{sub}, run-{run}: {e}")

    if N_valid_runs == 0:
        return np.empty((number_of_vertices, 0)), 0

    # 5) Runs aneinander hängen (Zeitachse)
    clean_ts_runs = np.concatenate(clean_per_run, axis=1)  # (V, T_total)
    return clean_ts_runs, N_valid_runs


def main(sub, bids_folder_in, bids_folder_out=None,
         confspec="32P", ses=1, task="magjudge", space="fsaverage5",
         scrubbing=True, scrub_thresh=0.3,
         run_FD_filter=True, frames_per_run_thresh=104,
         bp_filtering=True, lower_bpf=0.01, upper_bpf=0.08):

    # Sub-ID zweistellig
    sub = f"{int(sub):02d}"
    bids_folder_out = bids_folder_out or bids_folder_in

    # 1) Confounds-Spezifikation
    mov_params = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']
    if confspec == "36P":
        general_params = ['csf', 'white_matter', 'global_signal']
    elif confspec == "32P":
        general_params = ['csf', 'white_matter']
    else:
        raise ValueError(f"Unbekannte confspec: {confspec}")
    base_params = mov_params + general_params

    fmriprep_confounds_include = []
    for p in base_params:
        fmriprep_confounds_include += [
            p, f"{p}_derivative1", f"{p}_power2", f"{p}_derivative1_power2"
        ]

    # 2) Cleaning
    clean_ts, N_valid_runs = cleanTS(
        sub, fmriprep_confounds_include, bids_folder_in,
        task=task, ses=ses, space=space,
        scrubbing=scrubbing, scrub_thresh=scrub_thresh,
        run_FD_filter=run_FD_filter, frames_per_run_thresh=frames_per_run_thresh,
        bp_filtering=bp_filtering, lower_bpf=lower_bpf, upper_bpf=upper_bpf
    )

    if N_valid_runs < runs_per_sub_thresh:
        print(f"Sub-{sub} ausgeschlossen: nur {N_valid_runs} gültige Runs.")
        return

    # 3) Korrelationsmatrix (V x V)
    correlation_measure = ConnectivityMeasure(kind="correlation")
    cm = correlation_measure.fit_transform([clean_ts.T])[0].astype(np.float32)

    # 4) Speichern
    out_dir = op.join(bids_folder_out, "derivatives", "correlation_matrices")
    os.makedirs(out_dir, exist_ok=True)
    fn = op.join(
        out_dir,
        f"sub-{sub}_ses-{ses}_task-{task}_space-{space}_confspec-{confspec}_CM.npy"
    )
    np.save(fn, cm)
    print(f"Saved correlation matrix for sub-{sub} → {fn}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', help="Sub-ID (z.B. 1, 2, 53)")
    parser.add_argument('--bids_folder', default='/mnt_01/ds-ASD')
    parser.add_argument('--bids_folder_out', default=None)
    parser.add_argument('--confspec', default='32P', choices=['32P', '36P'])
    parser.add_argument('--task', default='magjudge')
    parser.add_argument('--ses', default=1, type=int)
    parser.add_argument('--space', default='fsaverage5', choices=['fsaverage5', 'fsaverage'])
    args = parser.parse_args()

    main(args.subject, args.bids_folder, args.bids_folder_out,
         confspec=args.confspec, ses=args.ses, task=args.task, space=args.space)
