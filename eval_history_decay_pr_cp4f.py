import argparse
import pickle as pkl
import logging
import torch
from pathlib import Path
import traceback
import sys
from tqdm import tqdm
import numpy as np

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.ops.iou3d_nms import iou3d_nms_utils


# ------------------------------------------------------------
# Logging setup
# ------------------------------------------------------------
def setup_logger(log_path):
    logger = logging.getLogger("eval_history_decay_removal")
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
        description=(
            "Evaluate removal history decay: current frame is clean, "
            "but some number of past history frames actually fired."
        )
    )
    parser.add_argument('--cfg_file', required=True)
    parser.add_argument('--pred_dir', required=True,
                        help='Directory containing <segment>_p.pkl prediction files')
    parser.add_argument('--timing_annos', required=True,
                        help='Directory containing <segment>_a.pkl annotation files')
    parser.add_argument('--dataset', required=True,
                        help='Directory containing <segment>_d.pkl spoofed dataset files')
    parser.add_argument('--detector', required=True, choices=['ptt', 'msf4', 'msf8'],
                        help='Defines past history length: ptt->31, msf4->3, msf8->7')
    parser.add_argument('--metrics_out', default='history_decay_removal.pkl')
    parser.add_argument('--log_file', default='eval_history_decay_removal.log')
    parser.add_argument('--workers', type=int, default=4)

    # visibility / validity gates for the tracked target on the clean current frame
    parser.add_argument('--min_eval_range', type=float, default=0.0)
    parser.add_argument('--max_eval_range', type=float, default=75.0)
    parser.add_argument('--max_abs_y', type=float, default=75.0)

    # GT forward-tracking gate
    parser.add_argument('--max_step', type=float, default=6.0)
    parser.add_argument('--max_volume_ratio', type=float, default=2.0)

    return parser.parse_args()


# ------------------------------------------------------------
# History length helper
# ------------------------------------------------------------
def get_past_len(detector_name: str) -> int:
    if detector_name == 'ptt':
        return 31   # current clean frame + previous 31 frames
    if detector_name == 'msf4':
        return 3
    if detector_name == 'msf8':
        return 7
    raise ValueError(f"Unknown detector setting: {detector_name}")


# ------------------------------------------------------------
# Fired flag helper
# ------------------------------------------------------------
def frame_fired(meta):
    return meta is not None and bool(meta.get('active_this_frame', False))


# ------------------------------------------------------------
# Box helpers
# ------------------------------------------------------------
def box_volume(box):
    return abs(box[3] * box[4] * box[5])


def box_range_xy(box):
    return float(np.linalg.norm(box[:2]))


def is_box_valid_for_eval(box, min_eval_range=0.0, max_eval_range=75.0, max_abs_y=75.0):
    """
    Visibility / relevance gate for the target box on the clean current frame.
    If the target has organically left the effective LiDAR region, drop it.
    """
    x, y = float(box[0]), float(box[1])
    r = np.linalg.norm([x, y])

    if r < min_eval_range:
        return False
    if r > max_eval_range:
        return False
    if abs(y) > max_abs_y:
        return False
    return True


# ------------------------------------------------------------
# Prediction accessor
# ------------------------------------------------------------
def get_pred_boxes_labels_scores(frame_pred):
    pred_boxes = frame_pred.get('boxes_lidar', frame_pred.get('pred_boxes', None))
    pred_labels = frame_pred.get('pred_labels', np.array([]))
    pred_scores = frame_pred.get('score', frame_pred.get('pred_scores', np.array([])))
    return pred_boxes, np.array(pred_labels), np.array(pred_scores)


# ------------------------------------------------------------
# Build clean-current decay windows from ACTUAL fired flags
#
# k = number of past fired frames in the previous `past_len` frames
# current frame itself must be clean
#
# k=0  : clean current frame, fully clean history
# k>0  : clean current frame, but k past fired frames in history
# ------------------------------------------------------------
def build_decay_windows_from_spoof_annos(segment_annos,
                                         segment_preds,
                                         ordered_segments,
                                         past_len):
    num_windows = past_len + 1
    window_frame_ids = {k: [] for k in range(num_windows)}
    decay_frame_to_source_local_idx = {}  # (seg, local_idx_current) -> most recent fired local idx

    for seg in ordered_segments:
        seg_preds_local = segment_preds[seg]
        spoof_annos = segment_annos[seg]['spoof_annos']
        n = len(seg_preds_local)

        fired_flags = np.zeros(n, dtype=np.int32)
        for local_idx in range(n):
            meta = spoof_annos.get(local_idx, None)
            fired_flags[local_idx] = 1 if frame_fired(meta) else 0

        prefix = np.concatenate([[0], np.cumsum(fired_flags)])

        last_fired_before = np.full(n, -1, dtype=np.int32)
        last_seen = -1
        for i in range(n):
            last_fired_before[i] = last_seen
            if fired_flags[i] == 1:
                last_seen = i

        for local_idx in range(n):
            meta = spoof_annos.get(local_idx, None)

            # current frame must be clean
            if frame_fired(meta):
                continue

            left = max(0, local_idx - past_len)
            past_fired_count = int(prefix[local_idx] - prefix[left])  # exclude current frame
            frame_id = seg_preds_local[local_idx]['frame_id']

            window_frame_ids[past_fired_count].append((frame_id, seg, local_idx))

            if past_fired_count > 0:
                src_idx = int(last_fired_before[local_idx])
                if src_idx >= left:
                    decay_frame_to_source_local_idx[(seg, local_idx)] = src_idx

    return window_frame_ids, decay_frame_to_source_local_idx


# ------------------------------------------------------------
# Forward-track the target through GT boxes on future clean frames
#
# This is the key removal-specific fix:
# do NOT freeze the source box in global coordinates.
# Instead, re-associate the real moving target through gt_boxes.
# ------------------------------------------------------------
def match_box_forward(prev_box,
                      curr_boxes,
                      target_class=1,
                      max_step=6.0,
                      max_volume_ratio=2.0,
                      class_col=-1):
    if curr_boxes is None or len(curr_boxes) == 0:
        return None, None, None

    labels = curr_boxes[:, class_col].astype(int)
    valid_idxs = np.where(labels == target_class)[0]
    if len(valid_idxs) == 0:
        return None, None, None

    candidates = curr_boxes[valid_idxs]

    prev_ctr = prev_box[:3]

    # use vx, vy if available
    if len(prev_box) > 8:
        dt = 0.1
        vx, vy = prev_box[7], prev_box[8]
        pred_ctr = prev_ctr + np.array([vx, vy, 0.0]) * dt
    else:
        pred_ctr = prev_ctr

    dists = np.linalg.norm(candidates[:, :3] - pred_ctr[None, :], axis=1)
    order = np.argsort(dists)

    prev_vol = box_volume(prev_box)

    for j in order:
        if dists[j] > max_step:
            break

        cand_vol = box_volume(candidates[j])
        if prev_vol > 0:
            ratio = max(cand_vol, prev_vol) / min(cand_vol, prev_vol)
            if ratio > max_volume_ratio:
                continue

        orig_idx = int(valid_idxs[j])
        return curr_boxes[orig_idx].copy(), orig_idx, float(dists[j])

    return None, None, None


def track_box_forward_through_gt(seg_frames,
                                 source_box,
                                 source_idx,
                                 cur_idx,
                                 target_class=1,
                                 max_step=6.0,
                                 max_volume_ratio=2.0):
    prev_box = source_box.copy()

    for idx in range(source_idx + 1, cur_idx + 1):
        curr_boxes = seg_frames[idx]['gt_boxes']
        new_box, _, _ = match_box_forward(
            prev_box=prev_box,
            curr_boxes=curr_boxes,
            target_class=target_class,
            max_step=max_step,
            max_volume_ratio=max_volume_ratio
        )
        if new_box is None:
            return None
        prev_box = new_box

    return prev_box


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
    # Load predictions
    # ----------------------------------------------------------
    segment_preds = {}
    for seg in ordered_segments:
        with open(f"{args.pred_dir}/{seg}_p.pkl", "rb") as f:
            segment_preds[seg] = pkl.load(f)

    # ----------------------------------------------------------
    # Load annotations
    # ----------------------------------------------------------
    segment_annos = {}
    for seg in ordered_segments:
        with open(f"{args.timing_annos}/{seg}_a.pkl", "rb") as f:
            segment_annos[seg] = pkl.load(f)

    # ----------------------------------------------------------
    # Reassemble det_annos in dataset order
    # ----------------------------------------------------------
    # segment_frame_counters = {}
    # det_annos = []
    # for info in dataset.infos:
    #     seg = info['point_cloud']['lidar_sequence']
    #     if seg not in segment_frame_counters:
    #         segment_frame_counters[seg] = 0
    #     idx = segment_frame_counters[seg]
    #     det_annos.append(segment_preds[seg][idx])
    #     segment_frame_counters[seg] += 1

    # for pred, info in zip(det_annos, dataset.infos):
    #     assert pred['frame_id'] == info['frame_id'], \
    #         f"Frame ID mismatch: {pred['frame_id']} vs {info['frame_id']}"

    # frame_id_to_info = {info['frame_id']: info for info in dataset.infos}
    # frame_id_to_anno = {anno['frame_id']: anno for anno in det_annos}
    
    segment_frame_counters = {}
    det_annos = []
    current_segment = None
    seg_data = None

    logger.info("Pre-loading spoof_gt from dataset files...")
    frame_id_to_spoof_gt = {}
    
    for info in dataset.infos:
        seg = info['point_cloud']['lidar_sequence']
        if seg not in segment_frame_counters:
            segment_frame_counters[seg] = 0
        idx = segment_frame_counters[seg]

        # load new segment dataset when segment changes
        if seg != current_segment:
            if seg_data is not None:
                del seg_data
            seg_data = None
            with open(f"{args.dataset}/{seg}_d.pkl", "rb") as f:
                seg_data = pkl.load(f)
            current_segment = seg
        frame = seg_data[idx]
        preds_cp4f = segment_preds[seg][idx].copy()
        
        gt = frame.get('spoof_gt', None)
        if gt is not None:
            frame_id_to_spoof_gt[frame['frame_id']] = gt
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

    # Map frame_id -> (info, anno) for O(1) lookup
    frame_id_to_info = {info['frame_id']: info for info in dataset.infos}
    frame_id_to_anno = {anno['frame_id']: anno for anno in det_annos}
    # ----------------------------------------------------------
    # Pre-load compact per-segment frame data from generated dataset
    # We need gt_boxes, pose, spoof_gt at each local frame index.
    # ----------------------------------------------------------
    logger.info("Pre-loading compact segment frame data from dataset files...")
    segment_frame_data = {}
    for seg in tqdm(ordered_segments, desc="Loading segment frame data"):
        seg_dataset_file = f"{args.dataset}/{seg}_d.pkl"
        with open(seg_dataset_file, "rb") as f:
            seg_data = pkl.load(f)

        compact = []
        for frame in seg_data:
            compact.append({
                'frame_id': frame['frame_id'],
                'gt_boxes': frame['gt_boxes'],
                'pose': frame['pose'],
                'spoof_gt': frame.get('spoof_gt', None),
            })
        segment_frame_data[seg] = compact
        del seg_data

    # ----------------------------------------------------------
    # Build clean-current decay windows from actual fired flags
    # ----------------------------------------------------------
    past_len = get_past_len(args.detector)
    num_windows = past_len + 1

    logger.info("Building decay windows from spoof_annos.active_this_frame ...")
    window_frame_ids, decay_frame_to_source_local_idx = build_decay_windows_from_spoof_annos(
        segment_annos=segment_annos,
        segment_preds=segment_preds,
        ordered_segments=ordered_segments,
        past_len=past_len
    )

    for k in range(num_windows):
        logger.info(f"Window k={k:2d}: {len(window_frame_ids[k])} frames")

    # ----------------------------------------------------------
    # Evaluate each window
    # ----------------------------------------------------------
    results_list = []
    original_infos = dataset.infos

    for k in tqdm(range(num_windows), desc="Evaluating windows"):
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
        # Removal-decay ASR:
        # current frame is clean, history had k fired frames.
        #
        # Denominator:
        #   clean current frames for which the target can be tracked
        #   to the current frame and still passes the visibility gate
        #
        # Numerator:
        #   those frames where the detector misses the target
        # ----------------------------------------------------------
        asr_records = []
        total_decay_valid = 0
        skipped_no_source = 0
        skipped_track_lost = 0
        skipped_visibility = 0

        if k >= 1:
            for fid, seg, loc_idx in zip(fids, segs, local_indices):
                source_idx = decay_frame_to_source_local_idx.get((seg, loc_idx), None)
                if source_idx is None:
                    skipped_no_source += 1
                    continue

                seg_frames = segment_frame_data[seg]
                source_box = seg_frames[source_idx]['spoof_gt']
                if source_box is None:
                    skipped_no_source += 1
                    continue

                target_class = int(source_box[-1])

                # Try direct spoof_gt first (within-block non-fired frames)
                # direct_gt = seg_frames[loc_idx].get('spoof_gt', None)
                # if direct_gt is not None:
                #     current_gt = direct_gt
                # else:
                    # Post-block: forward-track through GT
                current_gt = track_box_forward_through_gt(
                    seg_frames=seg_frames,
                    source_box=source_box,
                    source_idx=source_idx,
                    cur_idx=loc_idx,
                    target_class=target_class,
                    max_step=args.max_step,
                    max_volume_ratio=args.max_volume_ratio
                )
                if current_gt is None:
                    skipped_track_lost += 1
                    continue

                if not is_box_valid_for_eval(
                    current_gt,
                    min_eval_range=args.min_eval_range,
                    max_eval_range=args.max_eval_range,
                    max_abs_y=args.max_abs_y
                ):
                    skipped_visibility += 1
                    continue

                total_decay_valid += 1

                frame_pred = frame_id_to_anno[fid]
                pred_boxes, pred_labels, pred_scores = get_pred_boxes_labels_scores(frame_pred)

                if pred_boxes is None or len(pred_boxes) == 0:
                    continue

                iou_thresh = 0.7 if target_class == 1 else 0.5

                target_mask = (pred_labels == target_class)
                if not np.any(target_mask):
                    continue

                target_boxes = pred_boxes[target_mask]
                target_scores = pred_scores[target_mask]

                gt_tensor = torch.tensor(current_gt[:7], dtype=torch.float32).unsqueeze(0).cuda()
                pred_tensor = torch.tensor(target_boxes[:, :7], dtype=torch.float32).cuda()

                iou = iou3d_nms_utils.boxes_iou3d_gpu(pred_tensor, gt_tensor)
                max_iou, best_obj = iou[:, 0].max(0)
                max_iou = max_iou.item()
                
                #attack failure... see how good the score is after surviving detection 
                if  max_iou >= iou_thresh:
                    asr_records.append({
                        'segment': seg,
                        'local_idx': loc_idx,
                        'frame_id': fid,
                        'iou': max_iou,
                        'score': float(target_scores[best_obj.item()]),
                        'source_idx': source_idx,
                        'target_range_xy': box_range_xy(current_gt),
                    })

        dataset.infos = filtered_infos

        _, result_dict = dataset.evaluation(
            filtered_annos,
            cfg.CLASS_NAMES,
            eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC
        )

        asr = 1 - (len(asr_records) / total_decay_valid) if k >= 1 and total_decay_valid > 0 else None

        entry = {
            'k': k,  # number of past fired frames
            'num_frames': len(filtered_infos),
            'ap': result_dict,
            'asr': asr,
            'num_decay_valid': total_decay_valid,
            'skipped_no_source': skipped_no_source,
            'skipped_track_lost': skipped_track_lost,
            'skipped_visibility': skipped_visibility,
            'asr_records': asr_records,
        }
        results_list.append(entry)

        asr_str = f"ASR={asr:.4f}" if asr is not None else "ASR=N/A (clean baseline)"
        logger.info(
            f"k={k:2d} | frames={len(filtered_infos):4d} | past_fired={k if k >= 1 else 'N/A'} | "
            f"valid_decay={total_decay_valid:4d} | {asr_str} | "
            f"skip_no_source={skipped_no_source:4d} | "
            f"skip_track_lost={skipped_track_lost:4d} | "
            f"skip_visibility={skipped_visibility:4d} | "
            f"Veh_L1={result_dict.get('OBJECT_TYPE_TYPE_VEHICLE_LEVEL_1/AP', float('nan')):.4f} | "
            f"Ped_L1={result_dict.get('OBJECT_TYPE_TYPE_PEDESTRIAN_LEVEL_1/AP', float('nan')):.4f} | "
            f"Cyc_L1={result_dict.get('OBJECT_TYPE_TYPE_CYCLIST_LEVEL_1/AP', float('nan')):.4f}"
        )

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



# import argparse
# import pickle as pkl
# import logging
# import torch
# from pathlib import Path
# import traceback
# import sys
# from tqdm import tqdm
# import numpy as np

# from pcdet.config import cfg, cfg_from_yaml_file
# from pcdet.datasets import build_dataloader
# from pcdet.ops.iou3d_nms import iou3d_nms_utils


# # ------------------------------------------------------------
# # Logging setup
# # ------------------------------------------------------------
# def setup_logger(log_path):
#     logger = logging.getLogger("eval_history_decay_removal")
#     logger.setLevel(logging.INFO)
#     fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
#     fh = logging.FileHandler(log_path)
#     fh.setFormatter(fmt)
#     logger.addHandler(fh)
#     ch = logging.StreamHandler()
#     ch.setFormatter(fmt)
#     logger.addHandler(ch)
#     return logger


# # ------------------------------------------------------------
# # Argument parsing
# # ------------------------------------------------------------
# def parse_args():
#     parser = argparse.ArgumentParser(
#         description="Evaluate history decay for removal: clean current frame, but past fired frames in history."
#     )
#     parser.add_argument('--cfg_file', required=True)
#     parser.add_argument('--pred_dir', required=True,
#                         help='Directory containing <segment>_p.pkl prediction files')
#     parser.add_argument('--timing_annos', required=True,
#                         help='Directory containing <segment>_a.pkl annotation files')
#     parser.add_argument('--dataset', required=True,
#                         help='Directory containing <segment>_d.pkl spoofed dataset files')
#     parser.add_argument('--detector', required=True, choices=['ptt', 'msf4', 'msf8'],
#                         help='Defines history length: ptt->31 past frames, msf4->3, msf8->7')
#     parser.add_argument('--metrics_out', default='history_decay_removal.pkl')
#     parser.add_argument('--log_file', default='eval_history_decay_removal.log')
#     parser.add_argument('--workers', type=int, default=4)
#     return parser.parse_args()


# # ------------------------------------------------------------
# # Detector history length helpers
# # ------------------------------------------------------------
# def get_past_len(detector_name: str) -> int:
#     if detector_name == 'ptt':
#         return 31  # current clean frame + 31 prior frames in the 32-frame window
#     if detector_name == 'msf4':
#         return 3   # current clean frame + 3 prior frames
#     if detector_name == 'msf8':
#         return 7   # current clean frame + 7 prior frames
#     raise ValueError(f"Unknown detector setting: {detector_name}")


# def frame_fired(meta):
#     return meta is not None and bool(meta.get('active_this_frame', False))


# # ------------------------------------------------------------
# # Pose helpers: convert spoof_gt between ego and global coords
# # ------------------------------------------------------------
# def spoof_gt_ego_to_global(spoof_gt, pose):
#     gt = spoof_gt.copy()
#     box_hom = np.array([gt[0], gt[1], gt[2], 1.0], dtype=np.float32)
#     box_world = box_hom @ pose.T
#     gt[0:3] = box_world[0:3]
#     gt[6] += np.arctan2(pose[1, 0], pose[0, 0])
#     return gt


# def spoof_gt_global_to_ego(spoof_gt_global, pose):
#     gt = spoof_gt_global.copy()
#     inv_pose = np.linalg.inv(pose)
#     box_hom = np.array([gt[0], gt[1], gt[2], 1.0], dtype=np.float32)
#     box_ego = box_hom @ inv_pose.T
#     gt[0:3] = box_ego[0:3]
#     gt[6] -= np.arctan2(pose[1, 0], pose[0, 0])
#     return gt


# # ------------------------------------------------------------
# # Build decay windows from ACTUAL FIRED flags
# #
# # k = number of past fired frames in the previous `past_len` frames
# # current frame must be clean (active_this_frame == False)
# #
# # k=0  : clean current frame, fully clean history
# # k>0  : clean current frame, but k past fired frames in history
# # ------------------------------------------------------------
# def build_decay_windows_from_spoof_annos(segment_annos,
#                                          segment_preds,
#                                          ordered_segments,
#                                          past_len):
#     num_windows = past_len + 1
#     window_frame_ids = {k: [] for k in range(num_windows)}
#     decay_frame_to_source_local_idx = {}  # (seg, local_idx_current) -> source_local_idx

#     for seg in ordered_segments:
#         seg_preds_local = segment_preds[seg]
#         spoof_annos = segment_annos[seg]['spoof_annos']
#         n = len(seg_preds_local)

#         fired_flags = np.zeros(n, dtype=np.int32)
#         for local_idx in range(n):
#             meta = spoof_annos.get(local_idx, None)
#             fired_flags[local_idx] = 1 if frame_fired(meta) else 0

#         # prefix sums for counting fired frames in a history window
#         prefix = np.concatenate([[0], np.cumsum(fired_flags)])

#         # last_fired_before[i] = most recent fired frame strictly before i, else -1
#         last_fired_before = np.full(n, -1, dtype=np.int32)
#         last_seen = -1
#         for i in range(n):
#             last_fired_before[i] = last_seen
#             if fired_flags[i] == 1:
#                 last_seen = i

#         for local_idx in range(n):
#             # current frame must be clean
#             meta = spoof_annos.get(local_idx, None)

#             # skip if current frame actually fired
#             if frame_fired(meta):
#                 continue

#             left = max(0, local_idx - past_len)
#             # count only PREVIOUS fired frames, exclude current
#             past_fired_count = int(prefix[local_idx] - prefix[left])

#             frame_id = seg_preds_local[local_idx]['frame_id']
#             window_frame_ids[past_fired_count].append((frame_id, seg, local_idx))

#             if past_fired_count > 0:
#                 src_idx = int(last_fired_before[local_idx])
#                 if src_idx >= left:
#                     decay_frame_to_source_local_idx[(seg, local_idx)] = src_idx

#     return window_frame_ids, decay_frame_to_source_local_idx


# # ------------------------------------------------------------
# # Main
# # ------------------------------------------------------------
# def run_evaluations(args, logger):
#     cfg_from_yaml_file(args.cfg_file, cfg)
#     cfg.TAG = Path(args.cfg_file).stem
#     cfg.EXP_GROUP_PATH = 'centerpoint_waymo_demo'
#     logger.info(f'Loaded cfg from {args.cfg_file}')

#     dataset, _, _ = build_dataloader(
#         dataset_cfg=cfg.DATA_CONFIG,
#         class_names=cfg.CLASS_NAMES,
#         batch_size=1,
#         dist=False,
#         workers=args.workers,
#         logger=logger,
#         training=False
#     )
#     logger.info(f'Test set length: {len(dataset)}')

#     ordered_segments = list(dict.fromkeys(
#         info['point_cloud']['lidar_sequence'] for info in dataset.infos
#     ))

#     # ----------------------------------------------------------
#     # Load predictions
#     # ----------------------------------------------------------
#     segment_preds = {}
#     for seg in ordered_segments:
#         with open(f"{args.pred_dir}/{seg}_p.pkl", "rb") as f:
#             segment_preds[seg] = pkl.load(f)

#     # ----------------------------------------------------------
#     # Load annotations
#     # ----------------------------------------------------------
#     segment_annos = {}
#     for seg in ordered_segments:
#         with open(f"{args.timing_annos}/{seg}_a.pkl", "rb") as f:
#             segment_annos[seg] = pkl.load(f)

#     # ----------------------------------------------------------
#     # Reassemble det_annos in dataset order
#     # ----------------------------------------------------------
#     segment_frame_counters = {}
#     det_annos = []
#     for info in dataset.infos:
#         seg = info['point_cloud']['lidar_sequence']
#         if seg not in segment_frame_counters:
#             segment_frame_counters[seg] = 0
#         idx = segment_frame_counters[seg]
#         det_annos.append(segment_preds[seg][idx])
#         segment_frame_counters[seg] += 1

#     for pred, info in zip(det_annos, dataset.infos):
#         assert pred['frame_id'] == info['frame_id'], \
#             f"Frame ID mismatch: {pred['frame_id']} vs {info['frame_id']}"

#     frame_id_to_info = {info['frame_id']: info for info in dataset.infos}
#     frame_id_to_anno = {anno['frame_id']: anno for anno in det_annos}

#     # ----------------------------------------------------------
#     # Pre-load spoof_gt for source (fired) frames
#     # ----------------------------------------------------------
#     logger.info("Pre-loading spoof_gt from dataset files...")
#     frame_id_to_spoof_gt = {}
#     for seg in tqdm(ordered_segments, desc="Loading spoof_gt"):
#         seg_dataset_file = f"{args.dataset}/{seg}_d.pkl"
#         with open(seg_dataset_file, "rb") as f:
#             seg_data = pkl.load(f)
#         for frame in seg_data:
#             gt = frame.get('spoof_gt', None)
#             if gt is not None:
#                 frame_id_to_spoof_gt[frame['frame_id']] = gt
#         del seg_data
#     logger.info(f"Loaded spoof_gt for {len(frame_id_to_spoof_gt)} frames")

#     # ----------------------------------------------------------
#     # Build clean-current decay windows from actual fired flags
#     # ----------------------------------------------------------
#     past_len = get_past_len(args.detector)
#     num_windows = past_len + 1

#     logger.info("Building decay windows from spoof_annos.active_this_frame ...")
#     window_frame_ids, decay_frame_to_source_local_idx = build_decay_windows_from_spoof_annos(
#         segment_annos=segment_annos,
#         segment_preds=segment_preds,
#         ordered_segments=ordered_segments,
#         past_len=past_len
#     )

#     for k in range(num_windows):
#         logger.info(f"Window k={k:2d}: {len(window_frame_ids[k])} frames")

#     # ----------------------------------------------------------
#     # Precompute source spoof_gt in global coordinates
#     #
#     # For each clean current frame with k>0 past fired frames,
#     # use the MOST RECENT fired frame in history as the source.
#     # ----------------------------------------------------------
#     logger.info("Computing source spoof_gt in global coordinates for decay frames ...")
#     decay_frame_to_spoof_gt_global = {}  # current frame_id -> source spoof_gt in global coords

#     for seg in ordered_segments:
#         seg_preds_local = segment_preds[seg]

#         for (seg_name, local_idx_cur), source_idx in decay_frame_to_source_local_idx.items():
#             if seg_name != seg:
#                 continue

#             cur_fid = seg_preds_local[local_idx_cur]['frame_id']
#             src_fid = seg_preds_local[source_idx]['frame_id']

#             source_gt = frame_id_to_spoof_gt.get(src_fid, None)
#             if source_gt is None:
#                 continue

#             source_info = frame_id_to_info.get(src_fid, None)
#             if source_info is None:
#                 continue

#             source_pose = source_info['pose'].reshape(4, 4)
#             gt_global = spoof_gt_ego_to_global(source_gt, source_pose)
#             decay_frame_to_spoof_gt_global[cur_fid] = gt_global

#     logger.info(f"Mapped spoof_gt (global) for {len(decay_frame_to_spoof_gt_global)} clean-current decay frames")

#     # ----------------------------------------------------------
#     # Evaluate each window
#     # ----------------------------------------------------------
#     results_list = []
#     original_infos = dataset.infos

#     for k in tqdm(range(num_windows), desc="Evaluating windows"):
#         entries = window_frame_ids[k]
#         if len(entries) == 0:
#             logger.warning(f"Window k={k}: no frames found, skipping")
#             results_list.append(None)
#             continue

#         fids          = [e[0] for e in entries]
#         segs          = [e[1] for e in entries]
#         local_indices = [e[2] for e in entries]

#         filtered_infos = [frame_id_to_info[fid] for fid in fids if fid in frame_id_to_info]
#         filtered_annos = [frame_id_to_anno[fid] for fid in fids if fid in frame_id_to_anno]

#         if len(filtered_infos) == 0:
#             logger.warning(f"Window k={k}: no matching infos, skipping")
#             results_list.append(None)
#             continue

#         for pred, info in zip(filtered_annos, filtered_infos):
#             assert pred['frame_id'] == info['frame_id']

#         # ----------------------------------------------------------
#         # ASR for decay:
#         # current frame is CLEAN, but history had k fired frames.
#         # Success = target is STILL hidden on this clean current frame.
#         # ----------------------------------------------------------
#         asr_records = []
#         total_decay = 0

#         if k >= 1:
#             for fid, seg, loc_idx in zip(fids, segs, local_indices):
#                 gt_global = decay_frame_to_spoof_gt_global.get(fid, None)
#                 if gt_global is None:
#                     continue

#                 total_decay += 1

#                 # source target box projected into current frame ego coords
#                 frame_info = frame_id_to_info[fid]
#                 frame_pose = frame_info['pose'].reshape(4, 4)
#                 spoof_gt = spoof_gt_global_to_ego(gt_global, frame_pose)

#                 frame_pred = frame_id_to_anno[fid]
#                 pred_boxes = frame_pred.get('boxes_lidar', frame_pred.get('pred_boxes', None))
#                 if pred_boxes is None or len(pred_boxes) == 0:
#                     asr_records.append({
#                         'segment': seg,
#                         'local_idx': loc_idx,
#                         'frame_id': fid,
#                         'iou': 0.0,
#                         'score': 0.0,
#                     })
#                     continue

#                 pred_labels = frame_pred.get('pred_labels', np.array([]))
#                 scores = frame_pred.get('score', frame_pred.get('pred_scores', np.array([])))

#                 target_class = int(spoof_gt[-1])
#                 iou_thresh = 0.7 if target_class == 1 else 0.5

#                 target_mask = (np.array(pred_labels) == target_class)
#                 if not np.any(target_mask):
#                     asr_records.append({
#                         'segment': seg,
#                         'local_idx': loc_idx,
#                         'frame_id': fid,
#                         'iou': 0.0,
#                         'score': 0.0,
#                     })
#                     continue

#                 target_boxes = pred_boxes[target_mask]
#                 target_scores = np.array(scores)[target_mask]

#                 gt_tensor = torch.tensor(spoof_gt[:7], dtype=torch.float32).unsqueeze(0).cuda()
#                 pred_tensor = torch.tensor(target_boxes[:, :7], dtype=torch.float32).cuda()

#                 iou = iou3d_nms_utils.boxes_iou3d_gpu(pred_tensor, gt_tensor)  # (N,1)
#                 max_iou, best_obj = iou[:, 0].max(0)
#                 max_iou = max_iou.item()

#                 # success = still hidden on clean current frame
#                 if not (max_iou >= iou_thresh):
#                     asr_records.append({
#                         'segment': seg,
#                         'local_idx': loc_idx,
#                         'frame_id': fid,
#                         'iou': max_iou,
#                         'score': float(target_scores[best_obj.item()]),
#                     })

#         dataset.infos = filtered_infos

#         _, result_dict = dataset.evaluation(
#             filtered_annos,
#             cfg.CLASS_NAMES,
#             eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC
#         )

#         asr = len(asr_records) / total_decay if k >= 1 and total_decay > 0 else None

#         entry = {
#             'k': k,  # number of past fired frames
#             'num_frames': len(filtered_infos),
#             'ap': result_dict,
#             'asr': asr,
#             'num_decay_frames': total_decay,
#             'asr_records': asr_records,
#         }
#         results_list.append(entry)

#         asr_str = f"ASR={asr:.4f}" if asr is not None else "ASR=N/A (clean baseline)"
#         logger.info(
#             f"k={k:2d} | frames={len(filtered_infos):4d} | past_fired={k if k >= 1 else 'N/A'} | "
#             f"decay_frames={total_decay:4d} | {asr_str} | "
#             f"Veh_L1={result_dict.get('OBJECT_TYPE_TYPE_VEHICLE_LEVEL_1/AP', float('nan')):.4f} | "
#             f"Ped_L1={result_dict.get('OBJECT_TYPE_TYPE_PEDESTRIAN_LEVEL_1/AP', float('nan')):.4f} | "
#             f"Cyc_L1={result_dict.get('OBJECT_TYPE_TYPE_CYCLIST_LEVEL_1/AP', float('nan')):.4f}"
#         )

#     dataset.infos = original_infos

#     with open(args.metrics_out, "wb") as f:
#         pkl.dump(results_list, f)

#     logger.info(f"Decay curve saved to {args.metrics_out}")


# def main():
#     args = parse_args()
#     logger = setup_logger(args.log_file)
#     try:
#         run_evaluations(args, logger)
#     except Exception as e:
#         logger.error("===== EVALUATION FAILED =====")
#         logger.error(str(e))
#         logger.error(traceback.format_exc())
#         print(traceback.format_exc(), file=sys.stderr)
#         sys.exit(1)


# if __name__ == "__main__":
#     main()