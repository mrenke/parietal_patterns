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

# plotting gradients

import nilearn.plotting as nplt
import matplotlib.pyplot as plt
from  nilearn.datasets import fetch_surf_fsaverage

def plot_grads(grad, title=''):
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
    figure.suptitle(title, y=0.9)


from matplotlib.colors import ListedColormap
import hcp_utils as hcp
 
def plot_nets_CAcolors(modules_fsav5,  hemi_to_plot = 'R', title =None):
    rgb = np.array(list(hcp.ca_network['rgba'].values())[1:])

    grey = np.array([[0.5, 0.5, 0.5, 1.0]])  # RGBA format: grey with full opacity
    cmap_ca = ListedColormap( np.vstack([grey, rgb]))

    fsaverage = fetch_surf_fsaverage('fsaverage5') 
    views = ['medial','lateral','dorsal','posterior']
    cmap = cmap_ca #'Paired'#''viridis' # 

    i_hemi_to_plot = 0 if hemi_to_plot == 'L' else 1  
    modules_fsav5_hemi = np.split(modules_fsav5,2)[i_hemi_to_plot]

    surf_mesh = fsaverage.infl_right if hemi_to_plot =='R' else fsaverage.infl_left
    bg_map = fsaverage.sulc_right if hemi_to_plot =='R' else fsaverage.sulc_left

    figure, axes = plt.subplots(nrows=1, ncols=len(views),figsize = (15,8), subplot_kw=dict(projection='3d'))
    for i,view in enumerate(views):
        colbar = True if view == 'posterior' else False
        nplt.plot_surf(surf_mesh=surf_mesh , surf_map= modules_fsav5_hemi, avg_method = 'median',# infl_right # pial_right
                view= view,cmap=cmap, colorbar=colbar, #title=f'sub-{sub}, grad {n_grad+1}',
                vmin = 0, vmax=12,
                bg_map=bg_map, bg_on_data=True,darkness=0.7, axes=axes[i]) 
    figure.subplots_adjust(wspace=0.01)
    figure.suptitle(title, y=0.92)

    return figure


def get_gradients_tasks(subList,  bids_folder = '/mnt_AdaBD_largefiles/Data/DNumRisk_Data/ds-smile',
    n_gradients = 3,tasks = ['rest', 'magjudge', 'placevalue'], sessions = '1-2', derivative_obj_name = 'g-aligned'):
    
    dfs = []
    mask, _ = get_basic_mask()
    for task in tasks:
        gms = {f'g{i+1}': [] for i in range(n_gradients)}
        
        sub_array = []
        for sub in subList:
            sub_folder = op.join(bids_folder, 'derivatives', 'gradients', f'sub-{sub:03d}')
            fn = op.join(sub_folder, f'sub-{sub}_ses-{sessions}_task-{task}_{derivative_obj_name}.npy')
            if not op.exists(fn):
                print(f'File not found: {fn}')
                continue
            
            gm = np.load(fn)
            for i in range(n_gradients):
                gms[f'g{i+1}'].append(gm[i, mask])
            sub_array.append(int(sub))

        for i in range(n_gradients):
            gradient_df = pd.DataFrame(gms[f'g{i+1}'], index=sub_array)
            gradient_df.index.name = 'subject'
            gradient_df['n_gradient'] = i + 1
            gradient_df = gradient_df.set_index('n_gradient', append=True)
            gradient_df['task'] = task
            gradient_df = gradient_df.set_index('task', append=True)
            dfs.append(gradient_df)

    df_gms = pd.concat(dfs).sort_index()    # final multi-index dataframe

    return df_gms

def get_NPC_mask(bids_folder_orig = '/mnt_03/ds-dnumrisk', space = 'fsaverage5'):
    surf_mask_L = op.join(bids_folder_orig, 'derivatives/surface_masks', f'desc-NPC_L_space-{space}_hemi-lh.label.gii')
    surf_mask_L = nib.load(surf_mask_L).agg_data()
    surf_mask_R = op.join(bids_folder_orig, 'derivatives/surface_masks', f'desc-NPC_R_space-{space}_hemi-rh.label.gii')
    surf_mask_R = nib.load(surf_mask_R).agg_data()
    nprf_r2 = np.concatenate((surf_mask_L, surf_mask_R))

    nprf_r2 = np.bool_(nprf_r2)
    return nprf_r2


from scipy.stats import normaltest, ttest_ind, mannwhitneyu, ttest_rel

def between_group_comparison(df_tmp, y_var, alpha=0.05, group_names = ['Control','Dyscalculic']):
    pval_normal = normaltest(df_tmp[y_var]).pvalue
    if 'group' not in df_tmp.columns:
           df_tmp = df_tmp.reset_index('group')

    group1 = df_tmp[df_tmp['group'] == group_names[0]][y_var].dropna()
    group2 = df_tmp[df_tmp['group'] == group_names[1]][y_var].dropna()

    if pval_normal > alpha:
            stats = ttest_ind(group1, group2, axis=0)
            stats_term = f't({len(group1)+len(group2)-2})'
    else: # non parametric test
            stats = mannwhitneyu(group1, group2, axis=0)
            stats_term = f'U({len(group1)}, {len(group2)})'
    return stats, stats_term


