from glmsingle.glmsingle import GLM_single
import argparse
import os
import os.path as op
from nilearn import image
from numrisk.utils.data import Subject
from nilearn.glm.first_level import make_first_level_design_matrix
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def main(subject,  bids_folder, smoothed=False,  retroicor=False, split_data = None): # 'both', 'run_123', 'run_456'
    
    session = 1
    derivatives = op.join(bids_folder, 'derivatives')
    sub = Subject(subject, bids_folder=bids_folder)
    subject = f'{int(subject):02d}'

    n_stim = 2
    base_dir = f'glm_stim{n_stim}.denoise'
    runs = range(1, 7)
    
    ims = sub.get_preprocessed_bold(session=session, runs=runs)

    if retroicor:
            base_dir += '.retroicor'
            confounds = sub.get_retroicor_confounds(session)
    if smoothed:
        base_dir += '.smoothed'
        ims = [image.smooth_img(im, fwhm=5.0) for im in ims]
 
    data = [image.load_img(im).get_fdata() for im in ims]

    base_dir = op.join(derivatives, base_dir, f'sub-{subject}',
                       f'ses-{session}', 'func')

    if not op.exists(base_dir):
        os.makedirs(base_dir)

    onsets = sub.get_fmri_events_stim2(session=session, runs = runs) # np.shape(onsets)[0] = 2* N_trials !
    tr = 2.3
    n = np.shape(image.load_img(ims[0]).get_fdata())[3] # number of volumes
    frametimes = np.linspace(tr/2., (n - .5)*tr, n)
    onsets['onset'] = ((onsets['onset']+tr/2.) // 2.3) * 2.3

    print(onsets)

    dm = [make_first_level_design_matrix(frametimes, onsets.loc[run], hrf_model='fir', oversampling=100.,
                                         drift_order=0,
                                         drift_model=None).drop('constant', axis=1) for run in runs]

    dm = pd.concat(dm, keys=runs, names=['run']).fillna(0) # keys = range(1, 7)
    dm.columns = [c.replace('_delay_0', '') for c in dm.columns]
    dm /= dm.max()
    print(dm)
    dm[dm < 1.0] = 0.0
    print(dm.shape)

    X = [dm.loc[run].values for run in runs]

    print(len(X))

    # create a directory for saving GLMsingle outputs

    opt = dict()

    # set important fields for completeness (but these would be enabled by default)
    opt['wantlibrary'] = 1
    opt['wantglmdenoise'] = 1
    opt['wantfracridge'] = 1

    # for the purpose of this example we will keep the relevant outputs in memory
    # and also save them to the disk
    opt['wantfileoutputs'] = [0, 0, 0, 1]

    if retroicor:
        opt['extra_regressors'] = [cf.values for cf in confounds]
        print(opt)

    # running python GLMsingle involves creating a GLM_single object
    # and then running the procedure using the .fit() routine
    glmsingle_obj = GLM_single(opt)

    tmp_figuredir = op.join(base_dir, 'GLMestimatesingletrialfigures') # would be written to cwd otherwise and could crash when multiple nodes use it a the same time 
    results_glmsingle = glmsingle_obj.fit(
        X,
        data,
        0.6,
        2.3,
        outputdir=base_dir,
        figuredir = tmp_figuredir)

    betas = results_glmsingle['typed']['betasmd']
    betas = image.new_img_like(ims[0], betas)

    #GLM_single stil gives a single image for each event (even if they are put in as one regressor), chronological order
    # n1s
    betas_n1 = image.index_img(betas, slice(0, None, 2) ) # slice(0, None, 2) where to start, where to end, step size
    betas_n1.to_filename(op.join(base_dir, f'sub-{subject}_ses-{session}_task-magjudge_space-T1w_desc-stims1_est-with-{n_stim}-interest_pe.nii.gz'))
    
    # n2s
    betas_n2 = image.index_img(betas, slice(1, None, 2) ) # slice(0, None, 2) where to start, where to end, step size
    betas_n2.to_filename(op.join(base_dir, f'sub-{subject}_ses-{session}_task-magjudge_space-T1w_desc-stims{n_stim}_pe.nii.gz'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', default=None)
    parser.add_argument('--bids_folder', default='/data')
    parser.add_argument('--smoothed', action='store_true')
    parser.add_argument('--retroicor', action='store_true')


    args = parser.parse_args()

    main(args.subject,bids_folder=args.bids_folder, smoothed=args.smoothed, retroicor=args.retroicor)