#!/bin/bash
#SBATCH --account=gts-gchou3-ideas_l40s
#SBATCH --job-name=cp4f_rem_60
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# Generate timestamp (YYYYMMDD_HHMMSS)
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Create output folder
RESULTS_DIR=/storage/scratch1/9/spanse30/PTT/metrics_local_final_cp4f
mkdir -p $RESULTS_DIR




module load anaconda3
eval "$(conda shell.bash hook)"
conda activate /storage/project/r-gchou3-0/spanse30/ptt_env

cd /storage/project/r-gchou3-0/spanse30/MSF
export PYTHONPATH=$PWD:$PYTHONPATH



srun python eval_history_pr_cp4f.py \
    --cfg_file tools/cfgs/waymo_models/msf_4frames.yaml \
    --pred_dir /storage/scratch1/9/spanse30/PTT/preds_final_msf4/msf4_removal_60 \
    --timing_annos /storage/scratch1/9/spanse30/PTT/annos/removal_60 \
    --history_len 4 \
    --dataset /storage/scratch1/9/spanse30/PTT/datasets/removal_60 \
    --log_file $RESULTS_DIR/cp4f_removal_60_local_${TIMESTAMP}.log \
    --metrics_out $RESULTS_DIR/cp4f_removal_60_local.json\
    > $RESULTS_DIR/slurm_cp4f_local_removal_60_${TIMESTAMP}.out \
    2> $RESULTS_DIR/slurm_cp4f_local_removal_60_${TIMESTAMP}.err


#SBATCH --account=gts-gchou3-ideasci23_dgx 
#SBATCH --job-name=ptt_window
#SBATCH --partition=gpu-h300
#SBATCH --gres=gpu:h300:1