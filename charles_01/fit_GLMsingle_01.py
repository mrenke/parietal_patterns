from glmsingle.glmsingle import GLM_single
import argparse
import os
import os.path as op
from nilearn import image
#from neural_priors.utils.data import Subject
from nilearn.glm.first_level import make_first_level_design_matrix
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def get_fmri_events(subject, session=1,task='chase', bids_folder=None, runs=range(1,10)):
    behavior = []
    for run in runs:
        data = pd.read_table(op.join(bids_folder, f'sub-{subject}/ses-{session}/func/sub-{subject}_ses-{session}_task-{task}_run-{run}_events.tsv'))
        data['trial_nr'] = data['trial_nr'] + 40 * (run - 1)  # Adjust trial numbers to be unique across runs
        behavior.append(data)

    behavior = pd.concat(behavior, keys=runs, names=['run'])
    behavior = behavior.reset_index().set_index(['run', 'trial_type'])
    fixations = behavior.xs('fixation', 0, 'trial_type')
    fixations['trial_type'] = fixations.trial_nr.map(lambda trial_nr: f'fixation_{trial_nr}')
    choices = behavior.xs('choice', 0, 'trial_type')
    choices['trial_type'] = choices.trial_nr.map(lambda trial_nr: f'choice_{trial_nr}')
    responses = behavior.xs('response', 0, 'trial_type')
    responses['trial_type'] = responses.trial_nr.map(lambda trial_nr: f'response_{trial_nr}')
    feedbacks = behavior.xs('feedback', 0, 'trial_type')
    feedbacks['trial_type'] = feedbacks.trial_nr.map(lambda trial_nr: f'feedback_{trial_nr}')

    #events = pd.concat([fixations, choices, responses, feedbacks], axis=0).sort_values(['run', 'onset'])
    events = pd.concat([choices, feedbacks], axis=0).sort_values(['run', 'onset'])
    events = events[['onset', 'duration', 'trial_type']]  

    return events

def load_fmri_data(subject, session=1, task='chase', bids_folder=None, runs=range(1,10), space='T1w'):
    import nibabel as nib
    base = op.join(bids_folder, 'derivatives', 'fmriprep',f'sub-{subject}', f'ses-{session}', 'func')

    im_data = []
    for run in runs:
        fn = op.join(base, f'sub-{subject}_ses-{session}_task-{task}_run-{run}_space-{space}_desc-preproc_bold.nii.gz')
        im_data.append(nib.load(fn).get_fdata())

    return im_data

def main(subject,  bids_folder, space,  runs = range(1, 7), session = 1, task='chase'): #, smoothed=False,  retroicor=False, split_data = None): # 'both', 'run_123', 'run_456'



