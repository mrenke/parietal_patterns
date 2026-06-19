#!/bin/bash
# Run fmriprep for a specific subject + session (one SLURM array task per line
# in the batch file). Reuses existing FreeSurfer output from derivatives so
# recon-all is skipped for subjects that were already preprocessed.
#
# Generate the batch file first:
#   python sync_to_cluster.py --batch
#
# Submit:
#   sbatch --array=1-N fmriprep_sessions.sh fmriprep_batch_2026-06-03.txt
#
#SBATCH --job-name=fmriprep_ses
#SBATCH --output=/home/mrenke/logs/fmriprep_smile_%A-%a.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00

BATCH_FILE="$1"
if [[ -z "$BATCH_FILE" ]]; then
    echo "Usage: sbatch --array=1-N fmriprep_sessions.sh <batch_file.txt>"
    exit 1
fi

# read subject and session for this array task
read -r SUB SES < <(sed -n "${SLURM_ARRAY_TASK_ID}p" "$BATCH_FILE")

echo "Processing sub-${SUB} ses-${SES}"

module load apptainer
export SINGULARITYENV_FS_LICENSE=$HOME/freesurfer/license.txt
export SINGULARITYENV_TEMPLATEFLOW_HOME=/opt/templateflow

apptainer run -u \
    -B /shares/zne.uzh/$USER/ds-smile1:/data \
    -B /scratch/$USER/workflow_folders:/workflow \
    -B /scratch/$USER/templateflow:/opt/templateflow \
    --cleanenv \
    /home/$USER/data/containers/fmriprep_20.2.3 \
    /data /data/derivatives \
    participant \
    --participant-label "$SUB" \
    --session-id    "$SES" \ # 
    --output-spaces T1w fsaverage5 \
    --skip_bids_validation \
    --no-submm-recon \
    -w /workflow
