#!/bin/bash
#SBATCH --account=gts-gchou3-ideas_l40s
#SBATCH --job-name=cp4f_rem_40_eval
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=03:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# Generate timestamp (YYYYMMDD_HHMMSS)
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Create output folder
RESULTS_DIR=/storage/scratch1/9/spanse30/PTT/metrics_removal
mkdir -p $RESULTS_DIR




module load anaconda3
eval "$(conda shell.bash hook)"
conda activate /storage/project/r-gchou3-0/spanse30/ptt_env

cd /storage/project/r-gchou3-0/spanse30/MSF
export PYTHONPATH=$PWD:$PYTHONPATH




srun python eval_cp4f.py \
    --cfg_file tools/cfgs/waymo_models/msf_4frames.yaml \
    --dataset /storage/scratch1/9/spanse30/PTT/datasets/removal_50 \
    --pred_dir /storage/scratch1/9/spanse30/PTT/preds_final_msf4/msf4_removal_50 \
    --metrics_out $RESULTS_DIR/cp4f_removal_50.json \
    --log_file $RESULTS_DIR/cp4f_removal_50_eval_${TIMESTAMP}.log \
    > $RESULTS_DIR/slurm_cp4f_removal_50_eval_${TIMESTAMP}.out \
    2> $RESULTS_DIR/slurm_cp4f_removal_50_eval_${TIMESTAMP}.err

# srun python eval_history_decay_msf.py \
#     --cfg_file tools/cfgs/waymo_models/msf_4frames.yaml \
#     --timing_annos /storage/scratch1/9/spanse30/PTT/annos/rel_fixed_hard_a \
#     --dataset /storage/scratch1/9/spanse30/PTT/datasets/rel_fixed_hard_d \
#     --pred_dir /storage/scratch1/9/spanse30/PTT/preds_final/msf_rel_fixed_hard \
#     --detector msf4 \
#     --history 4 \
#     --metrics_out $RESULTS_DIR/msf_rel_fixed_hard_decay_${TIMESTAMP}.pkl \
#     --log_file $RESULTS_DIR/msf_rel_fixed_hard_decay_${TIMESTAMP}.log \
#     > $RESULTS_DIR/slurm_msf_rel_fixed_hard_decay_${TIMESTAMP}.out \
#     2> $RESULTS_DIR/slurm_msf_rel_fixed_hard_decay_${TIMESTAMP}.err






#SBATCH --account=gts-gchou3-ideasci23_dgx 
#SBATCH --job-name=msf_sanity
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:h100:1