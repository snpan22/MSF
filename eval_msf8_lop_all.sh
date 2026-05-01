#!/bin/bash
#SBATCH --account=gts-gchou3-ideas_l40s
#SBATCH --job-name=msf8_all_LOP_rel_fixed_hard
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=16:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# Generate timestamp (YYYYMMDD_HHMMSS)
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Create output folder
RESULTS_DIR=/storage/scratch1/9/spanse30/PTT/LOP/metrics_pi
mkdir -p $RESULTS_DIR




module load anaconda3
eval "$(conda shell.bash hook)"
conda activate /storage/project/r-gchou3-0/spanse30/ptt_env


cd /storage/project/r-gchou3-0/spanse30/MSF
export PYTHONPATH=$PWD:$PYTHONPATH

srun python eval_msf8_pi_lop_all.py \
    --cfg_file tools/cfgs/waymo_models/msf_8frames.yaml \
    --ckpt output/cfgs/waymo_models/msf_8frames/default/ckpt/checkpoint_epoch_6.pth \
    --pred_dir /storage/scratch1/9/spanse30/PTT/LOP/preds_msf8/rel_fixed_hard_all_B06_T044 \
    --dataset /storage/scratch1/9/spanse30/PTT/datasets/rel_fixed_hard_d \
    --asr_path /storage/scratch1/9/spanse30/PTT/LOP/asr/msf8_rel_fixed_hard_B06_T044_all.pkl \
    --lop_ckpt /storage/scratch1/9/spanse30/LOP/ckpts/lop_vehicle/lop_best.pt \
    --boundary 0.6 \
    --pillar_threshold 0.44 \
    --history 8 \
    --log_file $RESULTS_DIR/msf8_rel_fixed_hard_B06_T044_all${TIMESTAMP}.log \
    --metrics_out $RESULTS_DIR/msf8_rel_fixed_hard_B06_T044_all.json \
    > $RESULTS_DIR/slurm_msf8_rel_fixed_hard_B06_T044_all${TIMESTAMP}.out \
    2> $RESULTS_DIR/slurm_msf8_rel_fixed_hard_B06_T044_all${TIMESTAMP}.err

#SBATCH --account=gts-gchou3-ideasci23_dgx 
#SBATCH --job-name=ptt_window
#SBATCH --partition=gpu-h300
#SBATCH --gres=gpu:h300:1

#SBATCH --account=gts-gchou3-ideas_l40s
#SBATCH --job-name=ptt_LOP_hard_04
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:l40s:1