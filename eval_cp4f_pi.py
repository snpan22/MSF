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
from pcdet.ops.iou3d_nms import iou3d_nms_utils


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
    parser.add_argument('--asr_path', required=True)

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

    total_target_frames = 0
    num_vehicle_fp = 0
    num_ped_fp = 0
    num_cyc_fp = 0
    num_0_preds = 0
    scores_fp_vehicles = []
    scores_fp_ped = []
    scores_fp_cyc = []
    spoof_rc_survivability_vehicles = []
    spoof_rc_survivability_ped = []
    spoof_rc_survivability_cyc = []
    asr_annos_det_mode = {}
    
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
        preds_cp4f['name'] = np.array(names)
        det_annos.append(preds_cp4f)
        segment_frame_counters[seg] += 1
            
        gt_spoof = frame['spoof_gt']
        if gt_spoof is not None:
            total_target_frames += 1
            pred_boxes = preds_cp4f['boxes_lidar']

            if pred_boxes.shape[0] == 0:
                num_0_preds += 1
                # print("0 preds")
                continue
            gt_boxes = frame['gt_boxes']

            scores = preds_cp4f['score']
            labels = preds_cp4f['pred_labels']

            pred = torch.tensor(pred_boxes[:, :7]).cuda().float()

            gt = torch.tensor(gt_spoof[:7]).unsqueeze(0).cuda().float()

            iou = iou3d_nms_utils.boxes_iou3d_gpu(pred, gt)
            max_iou = iou.max().item()
            idx = torch.argmax(iou).item()
            spoof_score = scores[idx]
            spoof_label = labels[idx]

            # n_spoof_r = frame['lag0_n_spoof_r']
            # n_spoof_k = frame['lag0_n_spoof_k']

            if(max_iou >= 0.7 and spoof_label == 1):
                # print(frame_id)
                # print(max_iou, spoof_score)
                num_vehicle_fp += 1
                scores_fp_vehicles.append(spoof_score)
                # spoof_rc_survivability_vehicles.append((n_spoof_r, n_spoof_k, n_spoof_k/n_spoof_r))
            if(spoof_label == 2 and max_iou >= 0.5):
                num_ped_fp += 1
                scores_fp_ped.append(spoof_score)
                # spoof_rc_survivability_ped.append((n_spoof_r, n_spoof_k, n_spoof_k/n_spoof_r))
                # print("spoof misclasified as pedestrian")
            if(spoof_label == 3 and max_iou >= 0.5):
                num_cyc_fp +=1
                scores_fp_cyc.append(spoof_score)
                # spoof_rc_survivability_cyc.append((n_spoof_r, n_spoof_k, n_spoof_k/n_spoof_r))

    for pred, info in zip(det_annos, dataset.infos):
        assert pred['frame_id'] == info['frame_id'], \
            f"Frame ID mismatch: {pred['frame_id']} vs {info['frame_id']}"
    
    
    asr_annos_det_mode['asr_vehicle'] = num_vehicle_fp/total_target_frames
    asr_annos_det_mode['asr_pedestrian'] = num_ped_fp/total_target_frames
    asr_annos_det_mode['asr_cyclist'] = num_cyc_fp/total_target_frames
    asr_annos_det_mode['zero_pred_rate'] = num_0_preds / total_target_frames
    asr_annos_det_mode['scores_vehicle'] = np.array([t.item() for t in scores_fp_vehicles])
    asr_annos_det_mode['scores_pedestrian'] = np.array([t.item() for t in scores_fp_ped])
    asr_annos_det_mode['scores_cyclist'] = np.array([t.item() for t in scores_fp_cyc])
    # asr_annos_det_mode['spoof_surv_vehicle'] = spoof_rc_survivability_vehicles
    # asr_annos_det_mode['spoof_surv_pedestrian'] = spoof_rc_survivability_ped
    # asr_annos_det_mode['spoof_surv_cyclist'] = spoof_rc_survivability_cyc
    asr_annos_det_mode['num_targets'] = total_target_frames
    logger.info(f"asr : {asr_annos_det_mode['asr_vehicle']}")
    with open(args.asr_path, "wb") as f:
        pkl.dump(asr_annos_det_mode, f)

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