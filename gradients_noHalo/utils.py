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

def get_NPC_mask(bids_folder_orig = '/mnt_03/ds-dnumrisk', space = 'fsaverage5'):
    surf_mask_L = op.join(bids_folder_orig, 'derivatives/surface_masks', f'desc-NPC_L_space-{space}_hemi-lh.label.gii')
    surf_mask_L = nib.load(surf_mask_L).agg_data()
    surf_mask_R = op.join(bids_folder_orig, 'derivatives/surface_masks', f'desc-NPC_R_space-{space}_hemi-rh.label.gii')
    surf_mask_R = nib.load(surf_mask_R).agg_data()
    nprf_r2 = np.concatenate((surf_mask_L, surf_mask_R))

    nprf_r2 = np.bool_(nprf_r2)
    return nprf_r2

def get_glasser_parcels(base_folder='/mnt_03/diverse_neuralData/atlases_parcellations', space='fsaverage'):
    atlas_left = nib.load(op.join(base_folder,f'lh_space-{space}.HCPMMP1.gii')).agg_data()
    atlas_right =  nib.load(op.join(base_folder,f'rh_space-{space}.HCPMMP1.gii')).agg_data()

    labeling = np.concatenate([(atlas_left+1000), (atlas_right+2000)]) # unique labels for left and right!
    mask = ~np.isin(labeling, [1000,2000]) # non-cortex region (unknow and medial wall) have label 0, hence 1000 & 2000 in my variation labels L/R
    # mask.sum() == len(labeling[(labeling != 1000) & (labeling != 2000)]) 
    return mask, labeling
def get_glasser_CAatlas_mapping(datadir = '/mnt_03/diverse_neuralData/atlases_parcellations/ColeAnticevicNetPartition'):
    glasser_CAatlas_mapping = pd.read_csv(op.join(datadir,'cortex_parcel_network_assignments.txt'),header=None)
    glasser_CAatlas_mapping.index.name = 'glasser_parcel'
    glasser_CAatlas_mapping = glasser_CAatlas_mapping.rename({0:'ca_network'},axis=1)

    CAatlas_names = pd.read_csv(op.join(datadir,'network_label-names.csv'),index_col=0)
    CAatlas_names = CAatlas_names.set_index('Label Number')
    CAatlas_names = CAatlas_names.sort_index(level='Label Number')
    
    return glasser_CAatlas_mapping, CAatlas_names

def get_fsav5_CAatlas_mapping():
    mask_glasser, labeling_glasser = get_glasser_parcels(space = 'fsaverage5' )
    glasser_CAatlas_mapping, CAatlas_names = get_glasser_CAatlas_mapping()
    from brainspace.utils.parcellation import map_to_labels
    CAatlas_fsav5 = map_to_labels(glasser_CAatlas_mapping['ca_network'].values , labeling_glasser, mask=mask_glasser) #, fill=np.nan) #grad_sub[n_grad-1]
    return CAatlas_fsav5 #, glasser_CAatlas_mapping, CAatlas_names


# defined and used in getCM_specConf.py
#def cleanTS(sub, ses =1, task ='magjudge',runs = range(1, 7),space = 'fsaverage5', bids_folder='/Users/mrenke/data/ds-dnumrisk'): #  'magjudge'

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

