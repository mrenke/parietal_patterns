import argparse
import os.path as op
from numrisk.utils.data import Subject
from nilearn import surface
import nibabel as nb
import numpy as np

#from numrisk.fmri_analysis.encoding_model.fit_nprf import get_key_target_dir
from tqdm import tqdm
from nipype.interfaces.freesurfer import SurfaceTransform
from nilearn.maskers import NiftiMasker

#def transform_fsaverage(in_file, fs_hemi, source_subject, bids_folder, target_space = 'fsaverage5'):
#
#        subjects_dir = op.join(bids_folder, 'derivatives', 'freesurfer')
#
#        sxfm = SurfaceTransform(subjects_dir=subjects_dir)
#        sxfm.inputs.source_file = in_file
#        sxfm.inputs.out_file = in_file.replace('fsnative', target_space)
#        sxfm.inputs.source_subject = source_subject
#        sxfm.inputs.target_subject = target_space
#        sxfm.inputs.hemi = fs_hemi
#
#        r = sxfm.run()
#        return r

# Transform to fsaverage space while preserving the time dimension
# in_file, fs_hemi, source_subject, bids_folder, target_space = 'fsaverage5', time_dim=180

def transform_fsaverage(in_file, fs_hemi, source_subject, bids_folder, target_space = 'fsaverage5', time_dim=180):

        subjects_dir = op.join(bids_folder, 'derivatives', 'freesurfer')


        # Load the original data
        original_data = nb.load(target_fn).agg_data()
        print(f"Original data shape: {original_data.shape}")

        # Prepare an array to store the transformed data
        transformed_data = np.zeros((163842, time_dim))  # 163842 vertices for fsaverage

        # Loop over each time point and transform it
        for t in range(time_dim):
            # Save the current time point as a temporary GIFTI file
            temp_fn = target_fn.replace('.func.gii', f'_temp_t{t}.func.gii')
            im = nb.gifti.GiftiImage(darrays=[nb.gifti.GiftiDataArray(original_data[:, t].astype('float32'))])
            nb.save(im, temp_fn)

            # Transform the temporary file to fsaverage space
            sxfm = SurfaceTransform(subjects_dir=subjects_dir)
            sxfm.inputs.source_file = temp_fn
            sxfm.inputs.out_file = in_file.replace('fsnative', target_space)
            sxfm.inputs.source_subject = source_subject
            sxfm.inputs.target_subject = target_space
            sxfm.inputs.hemi = fs_hemi
            sxfm.run()

            # Load the transformed file and store the data
            transformed_file = sxfm.inputs.out_file
            transformed_data[:, t] = nb.load(transformed_file).agg_data()

        print(f"Transformed data shape: {transformed_data.shape}")
        return transformed_data

def main(subject_id, session,stim, bids_folder):
    
    sub = Subject(subject_id, bids_folder=bids_folder)
    subject = f'{int(subject_id):02d}'

    key = f'glm_stim{stim}.denoise'
    target_dir = op.join(bids_folder, 'derivatives', key , f'sub-{subject}', f'ses-{session}', 'func')

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

        im = nb.gifti.GiftiImage(darrays=[nb.gifti.GiftiDataArray(samples.astype('float32'))]) #added the as.type('float32') to avoid error
        target_fn = op.join(target_dir, f'sub-{subject}_ses-{session}_task-magjudge_space-fsnative_stim-{stim}_hemi-{hemi}.func.gii')
        nb.save(im, target_fn)

        transform_fsaverage(target_fn, fs_hemi, f'sub-{subject}', bids_folder, target_space = 'fsaverage5')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', default=None)
    parser.add_argument('--session', default=1)
    parser.add_argument('--stim', default=1)
    parser.add_argument('--bids_folder', default='/mnt_03/ds-dnumrisk')

    args = parser.parse_args()

    main(args.subject, args.session, args.stim,bids_folder=args.bids_folder) # ,denoise=args.denoise