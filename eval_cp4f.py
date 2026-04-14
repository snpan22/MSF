import torch
import torch.nn.functional as F
import argparse
import pickle as pkl
import logging
import json
from pathlib import Path
import traceback
import os
import sys
from tqdm import tqdm
from pcdet.utils import common_utils
import numpy as np

import importlib
import helpers_ptt
importlib.reload(helpers_ptt)

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.models import build_network, load_data_to_gpu
from pcdet.datasets import build_dataloader
from pcdet.utils import common_utils


# ------------------------------------------------------------
# Logging setup
# ------------------------------------------------------------
def setup_logger(log_path):

    logger = logging.getLogger("eval")
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    # File handler
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger


# ------------------------------------------------------------
# Argument parsing
# ------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--cfg_file', required=True)
    parser.add_argument('--pred_dir', required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--log_file', default="waymo_eval.log")
    parser.add_argument('--metrics_out', default="metrics.json")
    return parser.parse_args()


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def run_evaluations(args, logger):

    # args = parse_args()
    # logger = setup_logger(args.log_file)

    logger.info("Loading config...")

    logger.info("Building dataset (GT only, no model)...")
    
    
    CFG_FILE = args.cfg_file

    cfg_from_yaml_file(CFG_FILE, cfg)
    cfg.TAG = Path(CFG_FILE).stem
    cfg.EXP_GROUP_PATH = 'centerpoint_waymo_demo'

    logger.info(f'Loaded cfg from {CFG_FILE}')


    dataset, test_loader, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=1,
        dist=False,
        workers=4,
        logger=logger,
        training=False
    )

    len_test = len(dataset)
    
    
    ordered_segments = list(dict.fromkeys(
        info['point_cloud']['lidar_sequence'] for info in dataset.infos
    ))

    # ----------------------------------------------------------
    # Load all predictions
    # ----------------------------------------------------------
    segment_preds = {}
    for seg in ordered_segments:
        with open(f"{args.pred_dir}/{seg}_p.pkl", "rb") as f:
            segment_preds[seg] = pkl.load(f)


    # ----------------------------------------------------------
    # Reassemble det_annos in dataset order and build
    # frame_id -> (info, anno) lookup for fast filtering
    # ----------------------------------------------------------
    # segment_frame_counters = {}
    # det_annos = []
    # for info in dataset.infos:
    #     seg = info['point_cloud']['lidar_sequence']
    #     if seg not in segment_frame_counters:
    #         segment_frame_counters[seg] = 0
    #     idx = segment_frame_counters[seg]
        
    #     #index into segment dataset with seg and idx too
        
    #     det_annos.append(segment_preds[seg][idx])
    #     segment_frame_counters[seg] += 1
    
    segment_frame_counters = {}
    det_annos = []
    current_segment = None
    seg_data = None

    
    for info in dataset.infos:
        seg = info['point_cloud']['lidar_sequence']
        if seg not in segment_frame_counters:
            segment_frame_counters[seg] = 0
        idx = segment_frame_counters[seg]

        # load new segment dataset when segment changes
        if seg != current_segment:
            del seg_data
            seg_data = None
            with open(f"{args.dataset}/{seg}_d.pkl", "rb") as f:
                seg_data = pkl.load(f)
            current_segment = seg
        frame = seg_data[idx]
        preds_cp4f = segment_preds[seg][idx].copy()
        
        pred_boxes  = frame['pred_boxes']
        pred_scores = frame['pred_scores']
        pred_labels = frame['pred_labels']

        if isinstance(pred_boxes, torch.Tensor):
            pred_boxes  = pred_boxes.detach().cpu().numpy()
        if isinstance(pred_scores, torch.Tensor):
            pred_scores = pred_scores.detach().cpu().numpy()
        if isinstance(pred_labels, torch.Tensor):
            pred_labels = pred_labels.detach().cpu().numpy()

        preds_cp4f['boxes_lidar'] = pred_boxes
        preds_cp4f['score']       = pred_scores
        preds_cp4f['pred_labels'] = pred_labels

        names = []
        for label in pred_labels:
            if label == 1:
                names.append('Vehicle')
            elif label == 2:
                names.append('Pedestrian')
            elif label == 3:
                names.append('Cyclist')
        preds_cp4f['name'] = names
        det_annos.append(preds_cp4f)
        segment_frame_counters[seg] += 1

    for pred, info in zip(det_annos, dataset.infos):
        assert pred['frame_id'] == info['frame_id'], \
            f"Frame ID mismatch: {pred['frame_id']} vs {info['frame_id']}"
    
    # # --------------------------------------------------------
    # Progress indicator during evaluation
    # --------------------------------------------------------
    # dataset.evaluation() is monolithic, so we simulate progress
    # by wrapping the call — still useful for tracking runtime
    # --------------------------------------------------------

    logger.info("\n\nStarting Waymo evaluation")

    for _ in tqdm(range(1), desc="Waymo Metrics"):
        result_str, result_dict = dataset.evaluation(
            det_annos,
            cfg.CLASS_NAMES,
            eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC
        )

    logger.info("Evaluation complete")

    logger.info("\n===== RESULT STRING =====\n")
    logger.info("\n" + result_str)

    logger.info("\n===== RESULT DICT =====")
    logger.info(json.dumps(result_dict, indent=2, default = float))

    # Save metrics JSON
    with open(args.metrics_out, "w") as f:
        json.dump(result_dict, f, indent=2, default = float)

    logger.info(f"Metrics saved to {args.metrics_out}")

def main():

    args = parse_args()
    logger = setup_logger(args.log_file)

    try:
        run_evaluations(args, logger)

    except Exception as e:
        logger.error("===== EVALUATION FAILED =====")
        logger.error(str(e))
        logger.error("\nFull traceback:\n")
        logger.error(traceback.format_exc())

        # Also print to stderr so Slurm captures it
        print(traceback.format_exc(), file=sys.stderr)

        sys.exit(1)


if __name__ == "__main__":
    main()