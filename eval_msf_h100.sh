#!/bin/bash
#SBATCH --account=gts-gchou3-ideas_l40s
#SBATCH --job-name=msf4_decay
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
RESULTS_DIR=/storage/scratch1/9/spanse30/PTT/metrics_decay_msf4
mkdir -p $RESULTS_DIR




module load anaconda3
eval "$(conda shell.bash hook)"
conda activate /storage/project/r-gchou3-0/spanse30/ptt_env

cd /storage/project/r-gchou3-0/spanse30/MSF
export PYTHONPATH=$PWD:$PYTHONPATH






    # --cfg_file tools/cfgs/waymo_models/msf_4frames.yaml \


# srun python eval_history_curve_msf.py \
#     --cfg_file tools/cfgs/waymo_models/msf_4frames.yaml \
#     --timing_annos /storage/scratch1/9/spanse30/PTT/annos/global_hard_a \
#     --dataset /storage/scratch1/9/spanse30/PTT/datasets/global_hard_d \
#     --pred_dir /storage/scratch1/9/spanse30/PTT/preds_final_msf/msf_global_hard \
#     --detector msf4 \
#     --history 4 \
#     --metrics_out $RESULTS_DIR/msf4_global_hard_local.json \
#     --log_file $RESULTS_DIR/msf4_global_hard_local_${TIMESTAMP}.log \
#     > $RESULTS_DIR/slurm_msf4_global_hard_local_${TIMESTAMP}.out \
#     2> $RESULTS_DIR/slurm_msf4_global_hard_local_${TIMESTAMP}.err

# srun python eval_history_decay_msf.py \
#     --cfg_file tools/cfgs/waymo_models/msf_4frames.yaml \
#     --timing_annos /storage/scratch1/9/spanse30/PTT/annos/global_hard_a \
#     --dataset /storage/scratch1/9/spanse30/PTT/datasets/global_hard_d \
#     --pred_dir /storage/scratch1/9/spanse30/PTT/preds_final_msf/msf_global_hard \
#     --detector msf4 \
#     --history 4 \
#     --metrics_out $RESULTS_DIR/msf4_global_hard_decay.json \
#     --log_file $RESULTS_DIR/msf4_global_hard_decay_${TIMESTAMP}.log \
#     > $RESULTS_DIR/slurm_msf4_global_hard_decay_${TIMESTAMP}.out \
#     2> $RESULTS_DIR/slurm_msf4_global_hard_decay_${TIMESTAMP}.err






#SBATCH --account=gts-gchou3-ideasci23_dgx 
#SBATCH --job-name=msf_sanity
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:h100:1