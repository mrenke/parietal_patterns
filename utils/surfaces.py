import nibabel as nib
import numpy as np
import os.path as op
import nibabel as nib

def get_NPC_mask(bids_folder_orig = '/mnt_03/ds-dnumrisk', space = 'fsaverage5', hemi='both'):
    surf_mask_L = op.join(bids_folder_orig, 'derivatives/surface_masks', f'desc-NPC_L_space-{space}_hemi-lh.label.gii')
    surf_mask_L = nib.load(surf_mask_L).agg_data()
    surf_mask_R = op.join(bids_folder_orig, 'derivatives/surface_masks', f'desc-NPC_R_space-{space}_hemi-rh.label.gii')
    surf_mask_R = nib.load(surf_mask_R).agg_data()
    if hemi == 'both':
        nprf_r2 = np.concatenate((surf_mask_L, surf_mask_R))
    if hemi == 'L':
        nprf_r2 = np.concatenate((surf_mask_L, np.zeros_like(surf_mask_R)))
    if hemi == 'R':
        nprf_r2 = np.concatenate((np.zeros_like(surf_mask_L), surf_mask_R)) 

    nprf_r2 = np.bool_(nprf_r2)
    return nprf_r2