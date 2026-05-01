#!/bin/bash
#SBATCH --account=gts-gchou3-ideas_l40s
#SBATCH --job-name=msf_waymo
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:l40s:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=72:00:00
#SBATCH --output=msf8_waymo_l40s_%j.out
#SBATCH --error=msf8_waymo_l40s_%j.err

# module load anaconda3
# source activate /storage/project/r-gchou3-0/spanse30/ptt_env
module load anaconda3
eval "$(conda shell.bash hook)"
cd /storage/project/r-gchou3-0/spanse30/MSF
conda activate /storage/project/r-gchou3-0/spanse30/ptt_env


export PYTHONPATH=/storage/project/r-gchou3-0/spanse30/MSF:$PYTHONPATH
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG=INFO


# Unset any existing CUDA_VISIBLE_DEVICES that might interfere
# unset CUDA_VISIBLE_DEVICES
NUM_GPUS=2

TRAIN_ROI=/storage/project/r-gchou3-0/spanse30/OpenPCDet/output/cfgs/custom_models/centerpoint_multiframe_waymo/default/eval/epoch_36/train/default/result_fixed.pkl
VAL_ROI=/storage/project/r-gchou3-0/spanse30/OpenPCDet/output/cfgs/custom_models/centerpoint_multiframe_waymo/default/eval/epoch_36/val/default/result_fixed.pkl


srun -u python tools/train.py \
    --launcher slurm \
    --cfg_file tools/cfgs/waymo_models/msf_8frames.yaml \
    --workers 8 \
    --ckpt output/cfgs/waymo_models/msf_8frames/default/ckpt/latest_model.pth \
    --set DATA_CONFIG.ROI_BOXES_PATH.train ${TRAIN_ROI} \
         DATA_CONFIG.ROI_BOXES_PATH.test  ${VAL_ROI}

# torchrun --nproc_per_node=${NUM_GPUS} \
#     tools/train.py \
#     --launcher pytorch \
#     --cfg_file /storage/project/r-gchou3-0/spanse30/PTT/tools/cfgs/waymo_models/ptt_32frames.yaml \
#     --batch_size 8 \
#     --workers 8 \
#     --set DATA_CONFIG.ROI_BOXES_PATH.train ${TRAIN_ROI} \
#          DATA_CONFIG.ROI_BOXES_PATH.test  ${VAL_ROI}
