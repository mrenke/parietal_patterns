from nilearn import image
import numpy as np
import os.path as op
import nibabel as nib
import os
from nilearn import signal
import pandas as pd
from nipype.interfaces.freesurfer import SurfaceTransform # needs the fsaverage & fsaverage5 in ..derivatives/freesurfer folder!
from nilearn import datasets
import sys
import glob

def get_basic_mask():
    atlas = datasets.fetch_atlas_surf_destrieux()
    regions = atlas['labels'].copy()
    masked_regions = [b'Medial_wall', b'Unknown']
    masked_labels = [regions.index(r) for r in masked_regions]
    for r in masked_regions:
        regions.remove(r)
    labeling = np.concatenate([atlas['map_left'], atlas['map_right']])
    labeling_noParcel = np.arange(0,len(labeling),1,dtype = int)     # Map gradients to original parcels
    mask = ~np.isin(labeling, masked_labels)
    return mask, labeling_noParcel



def cleanTS(sub, ses =1, task ='magjudge',runs = range(1, 7),space = 'fsaverage5', bids_folder='/Users/mrenke/data/ds-dnumrisk'): #  'magjudge'
    # load in data as timeseries and regress out confounds (for each run sepeprately)
    if bids_folder.endswith('ds-smile1'):
        study = 'smile1'
        if task == 'magjudge':
            runs = range(1, 4)
        elif task =='rest':
            runs = [1]
    elif bids_folder.endswith('ds-numrisk'):
        study = 'miguel'
    elif bids_folder.endswith('ds-dnumrisk'):
        study = 'dyscalc'

    fmriprep_confounds_include = ['global_signal', 'dvars', 'framewise_displacement', 'trans_x',
                                    'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z',
                                    'a_comp_cor_00', 'a_comp_cor_01', 'a_comp_cor_02', 'a_comp_cor_03', 'cosine00', 'cosine01', 'cosine02'
                                    ] # 
    number_of_vertices = 20484 if space == 'fsaverage5' else sys.exit("currently only space='fsaverage5'implemented ")
    clean_ts_runs = np.empty([number_of_vertices,0])
    for run in runs: # loop over runs and concatenate timeseries
        #try:
            timeseries = [None] * 2
            for i, hemi in enumerate(['L', 'R']):
                if study == 'smile1' or study == 'dyscalc':
                    fmriprep_folder = op.join(bids_folder,'derivatives', 'fmriprep', f'sub-{sub}', f'ses-{ses}', 'func') # f'ses-{ses}', 
                    filename =  op.join(fmriprep_folder, f'sub-{sub}_ses-{ses}_task-{task}_run-{run}_space-{space}_hemi-{hemi}_bold.func.gii')   #_ses-{ses}
                    timeseries[i] = nib.load(filename).agg_data()
                elif study == 'miguel':
                    fmriprep_folder = op.join(bids_folder,'derivatives', 'fmriprep', f'sub-{sub}', 'func') # f'ses-{ses}', 
                    filename_pattern = op.join(fmriprep_folder, f"sub-{sub}_task-{task}_acq-*_run-{run}_space-{space}_hemi-{hemi}.func.gii")
                    timeseries[i] = nib.load(glob.glob(filename_pattern)[0]).agg_data()
            timeseries = np.vstack(timeseries) # (20484, 135)

            # confounds
            if study == 'smile1' or study == 'dyscalc':
                fmriprep_confounds_file = op.join(fmriprep_folder,f'sub-{sub}_ses-{ses}_task-{task}_run-{run}_desc-confounds_timeseries.tsv') # _ses-{ses} timeseries
                fmriprep_confounds = pd.read_table(fmriprep_confounds_file)[fmriprep_confounds_include] 
            elif study == 'miguel':
                fmriprep_confounds_filename_pattern = op.join(fmriprep_folder, f"sub-{sub}_task-{task}_acq-*_run-{run}_desc-confounds_regressors.tsv")
                fmriprep_confounds = pd.read_table(glob.glob(fmriprep_confounds_filename_pattern)[0])[fmriprep_confounds_include] 
            fmriprep_confounds= fmriprep_confounds.bfill()

            regressors_to_remove = fmriprep_confounds # remove_task_effects not implemented here (check dnumrisk)
            clean_ts = signal.clean(timeseries.T, confounds=regressors_to_remove).T
            clean_ts_runs = np.append(clean_ts_runs, clean_ts, axis=1)
        #except:
            #print(f'sub-{sub}, run-{run} makes problems') # (prob. confounds ts not there){fmriprep_confounds_file} \n skipping that run') # for sub 5,47,53,62

    return clean_ts_runs


# plotting gradients

import nilearn.plotting as nplt
import matplotlib.pyplot as plt
from  nilearn.datasets import fetch_surf_fsaverage

def plot_grads(grad, sub, spec, confspec):
    fsaverage = fetch_surf_fsaverage() # default 5
    side_view = 'medial'
    cmap = 'jet'
    n_comp = 5

    figure, axes = plt.subplots(nrows=1, ncols=n_comp,figsize = (15,4), subplot_kw=dict(projection='3d'))
    for i in range(0,n_comp):
        gm = np.split(grad[i],2) # for i, hemi in enumerate([‘L’, ‘R’]): --> left first
        gm_r = gm[1]
        nplt.plot_surf(surf_mesh= fsaverage.infl_right, surf_map= gm_r, # infl_right # pial_right
                    view= side_view,cmap=cmap, colorbar=False,  # sub-{sub}, title=f’grad {i+1}‘,
                    bg_map=fsaverage.sulc_right, bg_on_data=True,darkness=0.7 ,axes=axes[i]) #
        axes[i].set(title=f'grad {i+1}')
    figure.suptitle(f'sub-{sub} {spec} {confspec}', y=0.9)


# transform to fsLR for xcp_d postprocessing 
from neuromaps import transforms

def npFileTofsLRGii(sub, specification='',bids_folder='/Users/mrenke/data/ds-dnumrisk', gradient_Ns = [1,2,3], task = 'magjudge',
                    source_space='fsaverage5', target_space='fsLR_den-32k' ): # ses=1
    target_dir = op.join(bids_folder, 'derivatives', 'gradients', f'sub-{sub}') # , f'ses-{ses}'

    for n_grad in gradient_Ns:
        grad = np.load(op.join(target_dir, f'grad{n_grad}_space-{source_space}{specification}.npy'))
        grad = np.split(grad,2) # for i, hemi in enumerate(['L', 'R']): --> left first

        for h, hemi in enumerate(['L', 'R']):    
            gii_im_datar = nib.gifti.gifti.GiftiDataArray(data=grad[h].astype(np.float32)) #
            gii_im_fsav = nib.gifti.gifti.GiftiImage(darrays= [gii_im_datar])

            gii_im_fslr = transforms.fsaverage_to_fslr(gii_im_fsav, '32k',hemi=hemi)
            out_file = op.join(target_dir, f'sub-{sub}_task-{task}_space-{target_space}_hemi-{hemi}_grad{n_grad}{specification}.surf.gii') # _ses-{ses}
            nib.save(gii_im_fslr[0],out_file)
            print(f'saved to {out_file}')