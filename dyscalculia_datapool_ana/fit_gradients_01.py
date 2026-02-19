import argparse
from nilearn.connectome import ConnectivityMeasure
from brainspace.gradient import GradientMaps
from brainspace.utils.parcellation import map_to_labels
import numpy as np
import nibabel as nib
from nilearn import datasets
import os.path as op
import os
from nilearn import signal
import pandas as pd
from scipy.sparse.csgraph import connected_components
from utils import get_basic_mask


def main(sub, bids_folder,  sessions, tasks):
            
    #sub = f'{int(sub):02d}'
    key = ''
    target_dir = op.join(bids_folder, 'derivatives', f'gradients{key}', f'sub-{sub}')
    os.makedirs(target_dir, exist_ok=True)

    # Load in CM
    source_folder = op.join(bids_folder, 'derivatives', 'correlation_matrices', f'sub-{sub}')
    cm_fn = op.join(source_folder, f'sub-{sub}_ses-{sessions}_task-{tasks}_funcCM.npy')
    cm = np.load(cm_fn)
    # filter with connected components (remove outlier vertices)
    cc_mask_file = op.join(target_dir,f'sub-{sub}_cc-mask.npy')
    if (os.path.exists(cc_mask_file) == False):
        cc = connected_components(cm)
        mask_cc = cc[1] == 0 # all nodes in 0 belong to the largest connected component, check #-components in cc[0]
        np.save(cc_mask_file, mask_cc) # save all together
        print('connected components derived & mask saved')  
    mask_cc = np.load(cc_mask_file)
    mask, labeling_noParcel = get_basic_mask()
    mask[mask == True] = mask_cc # mark nodes not in component 0  as False in mask
    cm_filtered = cm[mask_cc, :][:, mask_cc]
    print('connectivty matrix loaded and filtered with cc_mask')

    # reference gradients (where grad-2 is anchored in visual2 !)
    ref_grad = '/mnt_AdaBD_largefiles/Data/SMILE_Data/DNumRisk/connectivity_references/dataset-dnumrisk_sub-All_gradients_kernel-normalized_angle_ztransf-True_avMethod-tanH.npy'
    grad_ref = np.load(ref_grad)
    grad_ref_fil = grad_ref[:,mask].T  # only use nodes in mask

    # Fit GMs
    n_components = 10
    gm = GradientMaps(n_components=n_components, alignment='procrustes', kernel='normalized_angle', approach='dm',
                      random_state=0) 
    gm.fit(cm_filtered ,reference=grad_ref_fil)

    # save results
    np.save(op.join(target_dir,f'sub-{sub}_ses-{sessions}_task-{tasks}_lambdas.npy'), gm.lambdas_) # save all together
    gm_= gm.gradients_.T 
    grad = [None] * n_components
    for i, g in enumerate(gm_): # gm.gradients_.T
        grad[i] = map_to_labels(g, labeling_noParcel, mask=mask, fill=np.nan)
    np.save(op.join(target_dir,f'sub-{sub}_ses-{sessions}_task-{tasks}_gradients.npy'), grad) # save all together
    gm_ = gm.aligned_.T
    grad = [None] * n_components
    for i, g in enumerate(gm_): # gm.gradients_.T
        grad[i] = map_to_labels(g, labeling_noParcel, mask=mask, fill=np.nan)
    np.save(op.join(target_dir,f'sub-{sub}_ses-{sessions}_task-{tasks}_g-aligned.npy'), grad) # save all together    
    print(f'finished sub-{sub} ses-{sessions} task-{tasks} : gradients saved')
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('subject', default=None)
    parser.add_argument('--bids_folder', default='/mnt_AdaBD_largefiles/Data/DNumRisk_Data/ds-smile')
    parser.add_argument('--sessions', default='1')
    parser.add_argument('--tasks', default='magjudge')

    cmd_args = parser.parse_args()

    main(cmd_args.subject, cmd_args.bids_folder, 
        sessions=cmd_args.sessions, 
        tasks=cmd_args.tasks,)  
