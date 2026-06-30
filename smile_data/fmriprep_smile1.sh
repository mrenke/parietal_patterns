#!/bin/bash
# Submit:
#   
#   ls /shares/zne.uzh/$USER/ds-smile1/sub-* -d | grep -oP '(?<=sub-)\d+' | tr '\n' ',' | sed 's/,$/\n/'
#   sbatch --array=... fmriprep_smile1.sh 
#
#SBATCH --job-name=fmriprep
#SBATCH --output=/home/mrenke/logs/fmriprep_smile1_%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00

module load apptainer

export APPTAINERENV_FS_LICENSE=$HOME/freesurfer/license.txt
export PARTICIPANT_LABEL=$(printf "%02d" $SLURM_ARRAY_TASK_ID)
export APPTAINERENV_TEMPLATEFLOW_HOME=/opt/templateflow

apptainer run -u \
  -B /shares/zne.uzh/$USER/ds-smile1:/data \
  -B /scratch/$USER/workflow_folders:/workflow \
  -B /scratch/$USER/templateflow:/opt/templateflow \
  --cleanenv \
  /home/$USER/data/containers/fmriprep_25.2.5.sif \
  /data /data/derivatives participant \
  --participant-label $PARTICIPANT_LABEL \
  --output-spaces T1w fsaverage5 \
  --skip_bids_validation \
  -w /workflow \
  --no-submm-recon