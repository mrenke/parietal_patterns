from nilearn import image
import numpy as np
import os.path as op
import nibabel as nib
import os
from nilearn import signal
import pandas as pd
from nipype.interfaces.freesurfer import SurfaceTransform # needs the fsaverage & fsaverage5 in ..derivatives/freesurfer folder!
from nilearn import datasets

# for plotting surface map
from brainspace.utils.parcellation import map_to_labels
from  nilearn.datasets import fetch_surf_fsaverage
import nilearn.plotting as nplt
import matplotlib.pyplot as plt


def fsavTofsav5(sub,ses = 1,bids_folder='/Volumes/mrenkeED/data/ds-dnumrisk',task = 'magjudge'):
    # requires fsaverage and fsaverage5 directory in bids_folder/derivatives/freesurfer !
    runs = range(1,7)
    
    for run in runs:
        for hemi in ['L', 'R']:
            sxfm = SurfaceTransform(subjects_dir=op.join(bids_folder,'derivatives','freesurfer'))
            in_file = f'sub-{sub}_ses-{ses}_task-{task}_run-{run}_space-fsaverage_hemi-{hemi}_bold.func.gii'
            in_file_path = op.join(bids_folder, 'derivatives', 'fmriprep', f'sub-{sub}',f'ses-{ses}','func',in_file)
            out_file = f'sub-{sub}_ses-{ses}_task-{task}_run-{run}_space-fsaverage5_hemi-{hemi}_bold.func.gii'
            out_file_path = op.join(bids_folder, 'derivatives', 'fmriprep', f'sub-{sub}',f'ses-{ses}','func',out_file)

            sxfm.inputs.source_file = in_file_path
            sxfm.inputs.out_file = out_file_path

            sxfm.inputs.source_subject = 'fsaverage'
            sxfm.inputs.target_subject = 'fsaverage5'

            if hemi == 'L':
                sxfm.inputs.hemi = 'lh'
            elif hemi == 'R':
                sxfm.inputs.hemi = 'rh'

            r = sxfm.run()




def surfTosurf(sub,source_space, target_space,     # requires both space directories in bids_folder/derivatives/freesurfer !
                ses = 1, runs=range(1,7), bids_folder='/Volumes/mrenkeED/data/ds-dnumrisk',task = 'magjudge'):

    for run in runs:
        for hemi in ['L', 'R']:
            sxfm = SurfaceTransform(subjects_dir=op.join(bids_folder,'derivatives','freesurfer'))
            in_file = f'sub-{sub}_ses-{ses}_task-{task}_run-{run}_space-{source_space}_hemi-{hemi}_bold.func.gii'
            in_file_path = op.join(bids_folder, 'derivatives', 'fmriprep', f'sub-{sub}',f'ses-{ses}','func',in_file)
            out_file = f'sub-{sub}_ses-{ses}_task-{task}_run-{run}_space-{target_space}_hemi-{hemi}_bold.func.gii'
            out_file_path = op.join(bids_folder, 'derivatives', 'fmriprep', f'sub-{sub}',f'ses-{ses}','func',out_file)

            sxfm.inputs.source_file = in_file_path
            sxfm.inputs.out_file = out_file_path

            sxfm.inputs.source_subject = source_space if source_space != 'fsnative' else f'sub-{int(sub):02d}'
            sxfm.inputs.target_subject = target_space if target_space != 'fsnative' else f'sub-{int(sub):02d}'

            if hemi == 'L':
                sxfm.inputs.hemi = 'lh'
            elif hemi == 'R':
                sxfm.inputs.hemi = 'rh'

            r = sxfm.run()



def saveGradToNPFile(grad, sub, specification='',bids_folder='/Users/mrenke/data/ds-dnumrisk',
                     space='fsaverage5'):
    target_dir = op.join(bids_folder, 'derivatives', 'gradients', f'sub-{sub}') # , f'ses-{ses}')

    if not op.exists(target_dir):
        os.makedirs(target_dir)

    for g, n_grad  in enumerate(range(1,1+np.shape(grad)[0])):
        np.save(op.join(target_dir,f'grad{n_grad}_space-{space}{specification}.npy'), grad[g])

def npFileTofs5Gii(sub, specification='',bids_folder='/Users/mrenke/data/ds-dnumrisk', gradient_Ns = [1,2,3], task = 'magjudge',space='fsaverage5' ): # ses=1
    target_dir = op.join(bids_folder, 'derivatives', 'gradients', f'sub-{sub}') # , f'ses-{ses}'

    for n_grad in gradient_Ns:
        grad = np.load(op.join(target_dir, f'grad{n_grad}_space-{space}{specification}.npy'))
        grad = np.split(grad,2) # for i, hemi in enumerate(['L', 'R']): --> left first

        for h, hemi in enumerate(['L', 'R']):    

            gii_im_datar = nib.gifti.gifti.GiftiDataArray(data=grad[h].astype(np.float32)) #
            gii_im = nib.gifti.gifti.GiftiImage(darrays= [gii_im_datar])

            out_file = op.join(target_dir, f'sub-{sub}_task-{task}_space-{space}_hemi-{hemi}_grad{n_grad}{specification}.surf.gii') # _ses-{ses}
            gii_im.to_filename(out_file) # https://nipy.org/nibabel/reference/nibabel.spatialimages.html
            print(f'saved to {out_file}')



def get_events_confounds(sub, ses, run, bids_folder='/Users/mrenke/data/ds-dnumrisk',task='magjudge' ):
    tr = 2.3 # repetition Time
    n = 188 # number of slices # adjust this important!!

    df_events = pd.read_csv(op.join(bids_folder, f'sub-{sub}', f'ses-{ses}', 'func', f'sub-{sub}_ses-{ses}_task-{task}_run-{run}_events.tsv'.format(sub=sub, ses=ses)), sep='\t') # before run was ot interated over (run-1)
    
    stimulus1 = df_events.loc[df_events['trial_type'] == 'stimulus 1', ['onset', 'trial_nr', 'trial_type', 'n1']]
    stimulus1['duration'] = 0.6 + 0.8
    stimulus1['onset'] = stimulus1['onset'] - 0.8 # cause we want to take the onset of the piechart 
    stimulus1['stim_order'] = int(1)
    stimulus1_int = stimulus1.copy()
    stimulus1_int['trial_type'] = 'stimulus1_int'
    stimulus1_int['modulation'] = 1
    stimulus1_mod= stimulus1.copy()
    stimulus1_mod['trial_type'] = 'stimulus1_mod'
    stimulus1_mod['modulation'] = stimulus1['n1']

    #choices = df_events.xs('choice', 0, 'trial_type', drop_level=False).reset_index('trial_type')[['onset', 'trial_nr', 'trial_type', 'n2']]
    choices = df_events.loc[df_events['trial_type'] == 'choice']

    #stimulus2 = df_events.xs('stimulus 2', 0, 'trial_type', drop_level=False).reset_index('trial_type')[['onset', 'trial_nr', 'trial_type', 'n2']]
    stimulus2 = df_events.loc[df_events['trial_type'] == 'stimulus 2', ['onset', 'trial_nr', 'trial_type', 'n2']]
    stimulus2 = stimulus2.set_index('trial_nr')
    stimulus2['duration'] = choices.set_index('trial_nr')['onset']- stimulus2['onset'] + 0.6 # 0.6 + 0.6 ## looked at the data, is is different for stim 1 and 2... ?!!
    stimulus2['onset'] = stimulus2['onset'] - 0.6
    stimulus2['stim_order'] = int(2)
    stimulus2_int = stimulus2.copy()
    stimulus2_int['trial_type'] = 'stimulus2_int'
    stimulus2_int['modulation'] = 1
    stimulus2_mod= stimulus2.copy()
    stimulus2_mod['trial_type'] = 'stimulus2_mod'
    stimulus2_mod['modulation'] = stimulus2['n2']
    stimulus2_int.reset_index(inplace=True)
    stimulus2_mod.reset_index(inplace=True)

    events = pd.concat((stimulus1_int,stimulus1_mod, stimulus2_int, stimulus2_mod)).set_index(['trial_nr','stim_order'],append=True).sort_index()

    onsets = events[['onset', 'duration', 'trial_type', 'modulation']].copy()
    onsets['onset'] = ((onsets['onset']+tr/2.) // 2.3) * 2.3

    frametimes = np.linspace(tr/2., (n - .5)*tr, n)

    from nilearn.glm.first_level import make_first_level_design_matrix
    dm = make_first_level_design_matrix(frametimes, onsets, 
                                        hrf_model='spm + derivative + dispersion', 
                                        oversampling=100.,drift_order=1, 
                                        drift_model=None).drop('constant', axis=1)
    dm /= dm.max()
    print('Design matrix created to remove task effects. shape:')
    print(dm.shape)
    return dm
