import argparse
import pickle as pkl
import logging
import torch
from pathlib import Path
import traceback
import os
import sys
from tqdm import tqdm
import numpy as np

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.utils import common_utils
from pcdet.ops.iou3d_nms import iou3d_nms_utils


# ------------------------------------------------------------
# Logging setup
# ------------------------------------------------------------
def setup_logger(log_path):
    logger = logging.getLogger("eval_history_decay")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger


# ------------------------------------------------------------
# Argument parsing
# ------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate history DECAY curve: clean obs with spoofed history (timing=3)')
    parser.add_argument('--cfg_file',     required=True)
    parser.add_argument('--pred_dir',     required=True,
                        help='Directory containing <segment>_p.pkl prediction files')
    parser.add_argument('--timing_annos', required=True,
                        help='Directory containing <segment>_a.pkl annotation files')
    parser.add_argument('--detector',     required=True, choices=['ptt', 'msf4', 'msf8'],
                        help='Which timing key to read: timing_ptt or timing_msf')
    parser.add_argument('--dataset',      required=True,
                        help='Directory containing <segment>_d.pkl spoofed dataset files')
    parser.add_argument('--metrics_out',  default='history_decay_ptt.pkl')
    parser.add_argument('--log_file',     default='eval_history_decay_ptt.log')
    parser.add_argument('--workers',      type=int, default=4)
    parser.add_argument('--history',         type=int, default=4)
    return parser.parse_args()


# ------------------------------------------------------------
# Helper: find decay block starts from timing array.
# A decay block begins at the first timing==3 frame that
# immediately follows a spoofed frame (timing 1 or 2).
# ------------------------------------------------------------
def find_decay_starts(timing_arr):
    starts = []
    for i in range(len(timing_arr)):
        if timing_arr[i] == 3 and (i == 0 or timing_arr[i - 1] != 3):
            starts.append(i)
    return starts


# ------------------------------------------------------------
# Pose helpers: convert spoof_gt between ego and global coords
# ------------------------------------------------------------
def spoof_gt_ego_to_global(spoof_gt, pose):
    """Convert a spoof_gt box from ego coordinates to global coordinates."""
    gt = spoof_gt.copy()
    box_hom = np.array([gt[0], gt[1], gt[2], 1.0])
    box_world = box_hom @ pose.T
    gt[0:3] = box_world[0:3]
    gt[6] += np.arctan2(pose[1, 0], pose[0, 0])
    return gt


def spoof_gt_global_to_ego(spoof_gt_global, pose):
    """Convert a spoof_gt box from global coordinates to ego coordinates."""
    gt = spoof_gt_global.copy()
    inv_pose = np.linalg.inv(pose)
    box_hom = np.array([gt[0], gt[1], gt[2], 1.0])
    box_ego = box_hom @ inv_pose.T
    gt[0:3] = box_ego[0:3]
    gt[6] -= np.arctan2(pose[1, 0], pose[0, 0])
    return gt


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def run_evaluations(args, logger):

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = 'centerpoint_waymo_demo'
    logger.info(f'Loaded cfg from {args.cfg_file}')

    dataset, _, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=1,
        dist=False,
        workers=args.workers,
        logger=logger,
        training=False
    )
    logger.info(f'Test set length: {len(dataset)}')

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
    # Load timing annotations
    # ----------------------------------------------------------
    segment_annos = {}
    for seg in ordered_segments:
        with open(f"{args.timing_annos}/{seg}_a.pkl", "rb") as f:
            segment_annos[seg] = pkl.load(f)

    timing_key = f"timing_{args.detector}"

    # ----------------------------------------------------------
    # Reassemble det_annos in dataset order
    # ----------------------------------------------------------
    segment_frame_counters = {}
    det_annos = []
    for info in dataset.infos:
        seg = info['point_cloud']['lidar_sequence']
        if seg not in segment_frame_counters:
            segment_frame_counters[seg] = 0
        idx = segment_frame_counters[seg]
        det_annos.append(segment_preds[seg][idx])
        segment_frame_counters[seg] += 1

    for pred, info in zip(det_annos, dataset.infos):
        assert pred['frame_id'] == info['frame_id'], \
            f"Frame ID mismatch: {pred['frame_id']} vs {info['frame_id']}"

    frame_id_to_info = {info['frame_id']: info for info in dataset.infos}
    frame_id_to_anno = {anno['frame_id']: anno for anno in det_annos}

    # ----------------------------------------------------------
    # Pre-load spoof_gt for spoofed frames (needed to derive
    # the global spoof position for each decay block).
    # ----------------------------------------------------------
    logger.info("Pre-loading spoof_gt from dataset files...")
    frame_id_to_spoof_gt = {}
    for seg in tqdm(ordered_segments, desc="Loading spoof_gt"):
        seg_dataset_file = f"{args.dataset}/{seg}_d.pkl"
        with open(seg_dataset_file, "rb") as f:
            seg_data = pkl.load(f)
        for frame in seg_data:
            gt = frame.get('spoof_gt', None)
            if gt is not None:
                frame_id_to_spoof_gt[frame['frame_id']] = gt
        del seg_data
    logger.info(f"Loaded spoof_gt for {len(frame_id_to_spoof_gt)} frames")

    # ----------------------------------------------------------
    # For each decay block, compute the spoof_gt in GLOBAL
    # coordinates by taking it from the last spoofed frame
    # (timing 1 or 2) and transforming via that frame's ego pose.
    #
    # Then map every decay frame_id to this global spoof_gt so
    # we can convert it to each frame's own ego coords at ASR
    # evaluation time.
    # ----------------------------------------------------------
    logger.info("Computing global spoof_gt for decay blocks...")
    decay_frame_to_spoof_gt_global = {}  #       frame_id -> spoof_gt in global coords

    # PTT: 32-frame window → n_prev = 31 → up to 31 decay frames
    NUM_DECAY_STEPS = args.history-1
    NUM_WINDOWS = NUM_DECAY_STEPS + 1  # k=0 clean baseline, k=1..31 decay

    for seg in ordered_segments:
        timing_arr = np.array(segment_annos[seg][timing_key])
        seg_preds  = segment_preds[seg]
        decay_starts = find_decay_starts(timing_arr)

        for ds in decay_starts:
            # Find source: last spoofed frame just before the decay
            source_idx = ds - 1
            if source_idx < 0:
                logger.warning(f"Decay block at {seg}[{ds}] has no preceding frame, skipping")
                continue

            source_fid = seg_preds[source_idx]['frame_id']
            source_gt  = frame_id_to_spoof_gt.get(source_fid, None)
            if source_gt is None:
                logger.warning(f"No spoof_gt for source frame {source_fid}, skipping decay block")
                continue

            source_info = frame_id_to_info.get(source_fid, None)
            if source_info is None:
                continue
            source_pose = source_info['pose'].reshape(4, 4)

            gt_global = spoof_gt_ego_to_global(source_gt, source_pose)

            # Assign to all timing=3 frames in this decay block
            for offset in range(NUM_DECAY_STEPS):
                idx = ds + offset
                if idx >= len(timing_arr) or timing_arr[idx] != 3:
                    break
                fid = seg_preds[idx]['frame_id']
                decay_frame_to_spoof_gt_global[fid] = gt_global

    logger.info(f"Mapped spoof_gt (global) for {len(decay_frame_to_spoof_gt_global)} decay frames")

    # ----------------------------------------------------------
    # Build per-segment decay window structures
    #
    # k=0  : clean baseline — all timing==0 frames
    #         (clean obs, fully clean history)
    #
    # k=1  : first frame after spoof block ends
    #         → clean obs, 31/31 previous frames spoofed
    # k=2  : second frame after
    #         → clean obs, 30/31 previous frames spoofed
    # ...
    # k=31 : 31st frame after
    #         → clean obs, 1/31 previous frames spoofed
    # ----------------------------------------------------------
    window_frame_ids = {k: [] for k in range(NUM_WINDOWS)}

    for seg in ordered_segments:
        timing_arr = np.array(segment_annos[seg][timing_key])
        seg_preds  = segment_preds[seg]
        decay_starts = find_decay_starts(timing_arr)

        # k=0: all timing==0 frames
        for idx in range(len(timing_arr)):
            if timing_arr[idx] == 0:
                frame_id = seg_preds[idx]['frame_id']
                window_frame_ids[0].append((frame_id, seg, idx))

        # k=1..31: decay frames after each spoof block
        for ds in decay_starts:
            for k in range(1, NUM_WINDOWS):
                offset_idx = ds + (k - 1)
                if offset_idx >= len(timing_arr):
                    break
                if timing_arr[offset_idx] != 3:
                    break  # decay ended early (segment boundary or next spoof block)
                frame_id = seg_preds[offset_idx]['frame_id']
                window_frame_ids[k].append((frame_id, seg, offset_idx))

    for k in range(NUM_WINDOWS):
        logger.info(f"Window k={k:2d}: {len(window_frame_ids[k])} frames")

    # ----------------------------------------------------------
    # Evaluate each window
    # ----------------------------------------------------------
    results_list = []
    original_infos = dataset.infos

    for k in tqdm(range(NUM_WINDOWS), desc="Evaluating windows"):
        entries = window_frame_ids[k]
        if len(entries) == 0:
            logger.warning(f"Window k={k}: no frames found, skipping")
            results_list.append(None)
            continue

        fids          = [e[0] for e in entries]
        segs          = [e[1] for e in entries]
        local_indices = [e[2] for e in entries]

        filtered_infos = [frame_id_to_info[fid] for fid in fids if fid in frame_id_to_info]
        filtered_annos = [frame_id_to_anno[fid] for fid in fids if fid in frame_id_to_anno]

        if len(filtered_infos) == 0:
            logger.warning(f"Window k={k}: no matching infos, skipping")
            results_list.append(None)
            continue

        for pred, info in zip(filtered_annos, filtered_infos):
            assert pred['frame_id'] == info['frame_id']

        # ----------------------------------------------------------
        # ASR: for decay frames (k>=1), check whether the detector
        # STILL produces a vehicle detection at the (now-absent)
        # phantom location. The spoof_gt is derived from the
        # preceding spoof block and transformed to each frame's
        # ego coordinates via the global intermediate.
        # ----------------------------------------------------------
        asr_records = []
        if k >= 1:
            for fid, seg, loc_idx in zip(fids, segs, local_indices):
                gt_global = decay_frame_to_spoof_gt_global.get(fid, None)
                if gt_global is None:
                    continue

                # Transform global spoof_gt → this frame's ego coords
                frame_info = frame_id_to_info[fid]
                frame_pose = frame_info['pose'].reshape(4, 4)
                spoof_gt = spoof_gt_global_to_ego(gt_global, frame_pose)

                frame_pred = frame_id_to_anno[fid]
                pred_boxes = frame_pred.get('boxes_lidar', None)
                if pred_boxes is None or len(pred_boxes) == 0:
                    continue

                pred_labels = frame_pred.get('pred_labels', np.array([]))
                scores      = frame_pred.get('score',       np.array([]))

                veh_mask = (np.array(pred_labels) == 1)
                if not np.any(veh_mask):
                    continue

                veh_boxes  = pred_boxes[veh_mask]
                veh_scores = np.array(scores)[veh_mask]

                gt_tensor   = torch.tensor(spoof_gt[:7], dtype=torch.float32).unsqueeze(0).cuda()
                pred_tensor = torch.tensor(veh_boxes[:, :7], dtype=torch.float32).cuda()

                iou = iou3d_nms_utils.boxes_iou3d_gpu(pred_tensor, gt_tensor)  # (N,1)
                max_iou, best_veh = iou[:, 0].max(0)
                max_iou = max_iou.item()

                if max_iou >= 0.7:
                    asr_records.append({
                        'segment':   seg,
                        'local_idx': loc_idx,
                        'frame_id':  fid,
                        'iou':       max_iou,
                        'score':     float(veh_scores[best_veh.item()]),
                    })

        dataset.infos = filtered_infos

        _, result_dict = dataset.evaluation(
            filtered_annos,
            cfg.CLASS_NAMES,
            eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC
        )

        asr = len(asr_records) / len(entries) if k >= 1 else None

        entry = {
            'k':           k,
            'num_frames':  len(filtered_infos),
            'ap':          result_dict,
            'asr':         asr,
            'asr_records': asr_records,
        }
        results_list.append(entry)

        asr_str = f"ASR={asr:.4f}" if asr is not None else "ASR=N/A (clean baseline)"
        spoofed_in_hist = max(0, NUM_DECAY_STEPS - (k - 1)) if k >= 1 else 'N/A'
        logger.info(f"k={k:2d} | frames={len(filtered_infos):4d} | spoofed_hist={spoofed_in_hist} | {asr_str} | "
                    f"Veh_L1={result_dict.get('OBJECT_TYPE_TYPE_VEHICLE_LEVEL_1/AP', float('nan')):.4f} | "
                    f"Ped_L1={result_dict.get('OBJECT_TYPE_TYPE_PEDESTRIAN_LEVEL_1/AP', float('nan')):.4f} | "
                    f"Cyc_L1={result_dict.get('OBJECT_TYPE_TYPE_CYCLIST_LEVEL_1/AP', float('nan')):.4f}")

    dataset.infos = original_infos

    with open(args.metrics_out, "wb") as f:
        pkl.dump(results_list, f)

    logger.info(f"Decay curve saved to {args.metrics_out}")


def main():
    args = parse_args()
    logger = setup_logger(args.log_file)
    try:
        run_evaluations(args, logger)
    except Exception as e:
        logger.error("===== EVALUATION FAILED =====")
        logger.error(str(e))
        logger.error(traceback.format_exc())
        print(traceback.format_exc(), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()