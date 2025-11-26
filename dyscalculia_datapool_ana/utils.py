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
