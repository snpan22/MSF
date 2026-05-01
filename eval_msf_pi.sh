#!/bin/bash
#SBATCH --account=gts-gchou3-ideas_l40s
#SBATCH --job-name=msf8_medium
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=15:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# Generate timestamp (YYYYMMDD_HHMMSS)
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Create output folder
RESULTS_DIR=/storage/scratch1/9/spanse30/PTT/pla/metrics_pi
# mkdir -p $RESULTS_DIR




module load anaconda3
eval "$(conda shell.bash hook)"
conda activate /storage/project/r-gchou3-0/spanse30/ptt_env

cd /storage/project/r-gchou3-0/spanse30/MSF
export PYTHONPATH=$PWD:$PYTHONPATH

srun python eval_msf8_pi.py \
    --cfg_file tools/cfgs/waymo_models/msf_8frames.yaml \
    --ckpt output/cfgs/waymo_models/msf_8frames/default/ckpt/checkpoint_epoch_6.pth \
    --dataset /storage/scratch1/9/spanse30/PTT/pla/datasets/global_medium \
    --pred_dir /storage/scratch1/9/spanse30/PTT/pla/preds_msf8/global_medium \
    --log_file $RESULTS_DIR/_msf8_global_medium_eval${TIMESTAMP}.log \
    --metrics_out $RESULTS_DIR/_msf8_global_medium.json \
    --history 8 \
    --asr_path /storage/scratch1/9/spanse30/PTT/pla/asr/_msf8_global_medium.pkl \
    > $RESULTS_DIR/slurm__msf8_global_medium_eval_${TIMESTAMP}.out \
    2> $RESULTS_DIR/slurm__msf8_global_medium_eval_${TIMESTAMP}.err



# srun python eval_msf.py \
#     --cfg_file tools/cfgs/waymo_models/msf_8frames.yaml \
#     --ckpt output/cfgs/waymo_models/msf_4frames/default/ckpt/checkpoint_epoch_6.pth \
#     --dataset /storage/scratch1/9/spanse30/PTT/datasets/removal_50 \
#     --pred_dir /storage/scratch1/9/spanse30/PTT/preds_final_msf4/msf4_removal_50 \
#     --log_file $RESULTS_DIR/msf4_removal_50_${TIMESTAMP}.log \
#     --history 4 \
#     --metrics_out $RESULTS_DIR/msf4_removal_50.json\
#     > $RESULTS_DIR/slurm_msf4_removal_50_${TIMESTAMP}.out \
#     2> $RESULTS_DIR/slurm_msf4_removal_50_${TIMESTAMP}.err



# srun python eval_msf_sanity.py \
#   --cfg_file tools/cfgs/waymo_models/msf_4frames.yaml \
#   --ckpt output/cfgs/waymo_models/msf_4frames/default/ckpt/checkpoint_epoch_6.pth \
#   --pred_dir /storage/scratch1/9/spanse30/PTT/preds/msf_clean \
#   --metrics_out metrics_msf_clean_sanity.json



#SBATCH --account=gts-gchou3-ideas_l40s
#SBATCH --job-name=msf_sanity
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:l40s:1

#SBATCH --account=gts-gchou3-ideasci23_dgx