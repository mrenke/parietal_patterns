from  nilearn.datasets import fetch_surf_fsaverage
import nilearn.plotting as nplt 
import numpy as np
import os.path as op

import matplotlib.patches as mpatches
import hcp_utils as hcp
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

def plot_nets_CAcolors(modules_fsav5,  hemi_to_plot = 'R'):
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

    return figure

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

def get_Gordon17_cmap():
    import matplotlib.colors as mcolors
    rgb = np.array(list(hcp.ca_network['rgba'].values())[1:])
    grey = np.array([[0.5, 0.5, 0.5, 1.0]])  # RGBA format: grey with full opacity

    custom_color_map = ListedColormap(np.vstack([grey, # 0: background
                                                rgb[8], # 1: DMN
                                                rgb[0], # 2: visual2
                                                rgb[6], # 3: fronto parietal
                                                mcolors.to_rgba('bisque'), #  4: visual 2
                                                rgb[4],  # 5: dorsal attention
                                                mcolors.to_rgba(mcolors.CSS4_COLORS['violet']),# 6: premotor
                                                rgb[5], # 7 : language
                                                mcolors.to_rgba('black'), # 8: salience
                                                rgb[3], # 9: Cingulo opercular
                                                rgb[2],# 10: motor - hand
                                                mcolors.to_rgba(mcolors.CSS4_COLORS['orange']),  # 11: motor - mouth
                                                rgb[7],# 12: auditory 
                                                mcolors.to_rgba('steelblue'), # 13: anterior MTL ?
                                                mcolors.to_rgba('palegreen'), # 14: posterior MTL
                                                mcolors.to_rgba('blue'), # 15: par memory
                                                mcolors.to_rgba('white'), # 16: context
                                                mcolors.to_rgba(mcolors.CSS4_COLORS['green']) # 17: motor - foot
                                                ])) 
    
    return custom_color_map

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