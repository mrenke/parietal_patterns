import argparse
import os
import os.path as op
from numrisk.utils.data import Subject
from nilearn import surface
import nibabel as nb
import numpy as np

#from numrisk.fmri_analysis.encoding_model.fit_nprf import get_key_target_dir
from tqdm import tqdm
from nipype.interfaces.freesurfer import SurfaceTransform
from nilearn.maskers import NiftiMasker

def transform_fsaverage(in_file, fs_hemi, source_subject, bids_folder, target_space = 'fsaverage5'):

        subjects_dir = op.join(bids_folder, 'derivatives', 'freesurfer')

        sxfm = SurfaceTransform(subjects_dir=subjects_dir)
        sxfm.inputs.source_file = in_file
        sxfm.inputs.out_file = in_file.replace('fsnative', target_space)
        sxfm.inputs.source_subject = source_subject
        sxfm.inputs.target_subject = target_space
        sxfm.inputs.hemi = fs_hemi

        r = sxfm.run()
        return r


def main(subject_id, session, stims, bids_folder):
    
    sub = Subject(subject_id, bids_folder=bids_folder)
    subject = f'{int(subject_id):02d}'

    stimsList = stims.split('-') if '-' in stims else [str(stims)]

    for stim in stimsList:

        key = f'glm_stim.denoise'
        target_dir = op.join('/mnt_AdaBD_largefiles/Data/SMILE_Data/DNumRisk/ds-numrisk/', 'derivatives', key , f'sub-{subject}', f'ses-{session}', 'func')

        surfinfo = sub.get_surf_info_fs()

        print(f'Writing to {target_dir}')


        fn = op.join(target_dir, f'sub-{subject}_ses-{session}_task-magjudge_space-T1w_desc-stims{stim}_pe.nii.gz')
        betas_vol = nb.load(fn)

        mask = sub.get_brain_mask()
        masker = NiftiMasker(mask_img=mask)
        betas = masker.fit_transform(betas_vol)
        betas_it= masker.inverse_transform(betas)

        for hemi in ['L', 'R']:
            samples = surface.vol_to_surf(betas_it, surfinfo[hemi]['outer'], inner_mesh=surfinfo[hemi]['inner'])
            fs_hemi = 'lh' if hemi == 'L' else 'rh'

            darrays = [nb.gifti.GiftiDataArray(samples[:, ix].astype('float32')) for ix in range(samples.shape[1])]
            im = nb.gifti.GiftiImage(darrays=darrays)

            for da in im.darrays:
                da.intent = nb.nifti1.intent_codes['NIFTI_INTENT_TIME_SERIES']

            target_fn = op.join(target_dir, f'sub-{subject}_ses-{session}_task-magjudge_space-fsnative_stim-{stim}_hemi-{hemi}.func.gii')
            nb.save(im, target_fn)

            transform_fsaverage(target_fn, fs_hemi, f'sub-{subject}', bids_folder, target_space = 'fsaverage5')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', default=None)
    parser.add_argument('--session', default=1)
    parser.add_argument('--stims', default='1-2')
    parser.add_argument('--bids_folder', default='/mnt_04/ds-numrisk')

    args = parser.parse_args()

    main(args.subject, args.session, args.stims,bids_folder=args.bids_folder) # ,denoise=args.denoise

    