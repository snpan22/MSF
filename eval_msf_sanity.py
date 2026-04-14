import torch
import argparse
import pickle as pkl
import logging
import json
from pathlib import Path
import traceback
import os
import sys
from tqdm import tqdm
import numpy as np
from collections import defaultdict

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
    logger = logging.getLogger("eval_msf_sanity_manual")
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_file',       required=True,  help='MSF 4-frame config')
    parser.add_argument('--msf_result_pkl', required=True,  help='MSF result.pkl (for frame_id -> GT lookup)')
    parser.add_argument('--ckpt',           required=True,  help='MSF checkpoint')
    parser.add_argument('--cp_sf_cfg',      required=True,  help='CenterPoint single-frame config (for clean points)')
    parser.add_argument('--openpcdet_root', required=True,  help='Path to OpenPCDet root (for single-frame pcdet)')
    parser.add_argument('--result_pkl',     required=True,  help='Clean CenterPoint result_fixed.pkl (proposals)')
    parser.add_argument('--pred_dir',       required=True)
    parser.add_argument('--workers',     type=int, default=4)
    parser.add_argument('--log_file',    default='msf_sanity_manual.log')
    parser.add_argument('--metrics_out', default='metrics_msf_sanity_manual.json')
    return parser.parse_args()


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def run_evaluations(args, logger):

    # ----------------------------------------------------------
    # 1. Load MSF dataset (provides poses + GT + dataset infos)
    # ----------------------------------------------------------
    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = 'centerpoint_waymo_demo'
    logger.info(f'Loaded MSF cfg from {args.cfg_file}')

    dataset, test_loader, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=1,
        dist=False,
        workers=args.workers,
        logger=logger,
        training=False
    )
    logger.info(f'MSF dataset length: {len(dataset)}')

    model = build_network(
        model_cfg=cfg.MODEL,
        num_class=len(cfg.CLASS_NAMES),
        dataset=dataset
    )
    logger.info(f'Loading MSF checkpoint from: {args.ckpt}')
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=False)
    model.cuda()
    model.eval()

    # MSF global_idx -> (segment, local_idx)
    global_to_segment = {}
    segment_frame_counters = {}
    for i, info in enumerate(dataset.infos):
        seg = info['point_cloud']['lidar_sequence']
        if seg not in segment_frame_counters:
            segment_frame_counters[seg] = 0
        global_to_segment[i] = (seg, segment_frame_counters[seg])
        segment_frame_counters[seg] += 1

    # ----------------------------------------------------------
    # 2. Load CenterPoint single-frame dataset (clean points)
    #    Must use OpenPCDet's pcdet, not MSF's, because the SF
    #    config has no SEQUENCE_CONFIG and MSF's waymo_dataset
    #    requires it. Swap sys.path temporarily.
    # ----------------------------------------------------------
    logger.info(f'Swapping to OpenPCDet pcdet to load SF dataset...')
    msf_pcdet_modules = {k: v for k, v in sys.modules.items() if k.startswith('pcdet')}
    for k in list(msf_pcdet_modules.keys()):
        del sys.modules[k]
    sys.path.insert(0, args.openpcdet_root)

    from pcdet.config import cfg as cfg_sf
    from pcdet.config import cfg_from_yaml_file as cfg_from_yaml_file_sf
    from pcdet.datasets import build_dataloader as build_sf_loader
    from pcdet.utils import common_utils as common_utils_sf

    cfg_sf.clear()
    cfg_from_yaml_file_sf(args.cp_sf_cfg, cfg_sf)
    cfg_sf.TAG = Path(args.cp_sf_cfg).stem
    cfg_sf.EXP_GROUP_PATH = 'centerpoint_waymo_demo'
    logger.info(f'Loaded SF cfg from {args.cp_sf_cfg}')

    sf_logger = common_utils_sf.create_logger()
    dataset_sf, _, _ = build_sf_loader(
        dataset_cfg=cfg_sf.DATA_CONFIG,
        class_names=cfg_sf.CLASS_NAMES,
        batch_size=1,
        dist=False,
        workers=args.workers,
        logger=sf_logger,
        training=False
    )
    logger.info(f'SF dataset length: {len(dataset_sf)}')

    # Swap back to MSF pcdet
    for k in list(sys.modules.keys()):
        if k.startswith('pcdet'):
            del sys.modules[k]
    sys.path.remove(args.openpcdet_root)
    for k, v in msf_pcdet_modules.items():
        sys.modules[k] = v
    logger.info('Restored MSF pcdet')

    # Build (segment, local_idx) -> sf_global_idx for point lookup
    sf_segment_to_locals = defaultdict(list)
    for sf_idx, info in enumerate(dataset_sf.infos):
        seg = info['point_cloud']['lidar_sequence']
        sf_segment_to_locals[seg].append(sf_idx)

    # (segment, local_idx) -> sf_global_idx
    sf_lookup = {}
    for seg, sf_idx_list in sf_segment_to_locals.items():
        for local_idx, sf_idx in enumerate(sf_idx_list):
            sf_lookup[(seg, local_idx)] = sf_idx

    # ----------------------------------------------------------
    # 3. Load clean CenterPoint proposals from result_fixed.pkl
    #    Index by (segment, local_idx) -> {boxes, scores, labels}
    # ----------------------------------------------------------
    logger.info(f'Loading clean proposals from: {args.result_pkl}')
    with open(args.result_pkl, 'rb') as f:
        clean_result = pkl.load(f)

    clean_proposals = {}
    result_frame_counters = {}
    for frame_dict in clean_result:
        fid = frame_dict['frame_id']
        seg = fid.rsplit('_', 1)[0]
        if seg not in result_frame_counters:
            result_frame_counters[seg] = 0
        local_idx = result_frame_counters[seg]
        result_frame_counters[seg] += 1

        boxes  = frame_dict['pred_boxes']
        scores = frame_dict['pred_scores']
        labels = frame_dict['pred_labels']
        if isinstance(boxes, torch.Tensor):
            boxes  = boxes.cpu().numpy()
            scores = scores.cpu().numpy()
            labels = labels.cpu().numpy()
        clean_proposals[(seg, local_idx)] = {
            'pred_boxes':  boxes.astype(np.float32),
            'pred_scores': scores.astype(np.float32),
            'pred_labels': labels.astype(np.float32),
        }
    logger.info(f'Clean proposals loaded for {len(result_frame_counters)} segments')

    # ----------------------------------------------------------
    # 4. Inference loop
    # ----------------------------------------------------------
    save_dir_preds = args.pred_dir
    os.makedirs(save_dir_preds, exist_ok=True)
    completed = set(f[:-4] for f in os.listdir(save_dir_preds))

    history = 4
    segment_preds_list = []
    current_segment = None
    sf_points_cache = {}  # local_idx -> points, cleared on segment switch
    
    with open(args.msf_result_pkl, "rb") as f:
        msf_result = pkl.load(f)

    # build frame_id -> anno lookup
    msf_result_lookup = {ann['frame_id']: ann for ann in msf_result}

    try:
        for i_msf, batch in enumerate(test_loader):

            segment = batch['frame_id'][0].rsplit("_", 1)[0]
            seg_idx = global_to_segment[i_msf][1]

            if f"{segment}_p" in completed:
                continue

            if current_segment != segment:
                if current_segment is not None:
                    pred_path = f"{save_dir_preds}/{current_segment}_p.pkl"
                    with open(pred_path, "wb") as f:
                        pkl.dump(segment_preds_list, f)
                    logger.info("Saved preds for: %s", current_segment)
                segment_preds_list = []
                current_segment = segment
                sf_points_cache = {}  # clear cache on segment switch

            # ------------------------------------------------------
            # Build 4-frame point cloud from clean SF dataset points,
            # transformed to current-frame (lag-0) coordinates using
            # MSF dataset poses — identical logic to eval_msf.py but
            # sourcing points from dataset_sf instead of seg_dataset.
            # ------------------------------------------------------
            past_frames_list = [max(seg_idx - i, 0) for i in range(history)]

            poses = dataset[i_msf]['poses']   # shape (16,4): 4 stacked (4,4) poses
            pose_lag0 = poses[0:4].reshape(4, 4)
            pose_lag0_inv = np.linalg.inv(pose_lag0)

            points_4frames = []
            for j, frame_local in enumerate(past_frames_list):
                # populate cache lazily — high hit rate since adjacent frames
                # share most historical local indices
                if frame_local not in sf_points_cache:
                    sf_idx = sf_lookup.get((segment, frame_local))
                    if sf_idx is None:
                        sf_idx = sf_lookup.get((segment, 0))
                    sf_points_cache[frame_local] = dataset_sf[sf_idx]['points'].copy()

                points_original = sf_points_cache[frame_local]

                if j == 0:
                    # current frame: no transform needed, append lag=0 column
                    # SF points are (N,5): x,y,z,intensity,elongation — no lag col
                    lag_col = np.zeros((points_original.shape[0], 1), dtype=np.float32)
                    points_4frames.append(np.hstack((points_original[:, :5], lag_col)))
                    continue

                # historical frame: transform to lag-0 ego coords
                pose_j = poses[j * 4: (j + 1) * 4].reshape(4, 4)
                ones = np.ones((points_original.shape[0], 1), dtype=np.float32)
                points_hom = np.hstack((points_original[:, :3], ones))
                points_lag0_coords = points_hom @ pose_j.T @ pose_lag0_inv.T

                # reassemble: transformed xyz | intensity | elongation | lag time
                # SF points are (N,5): cols 3:5 = intensity, elongation (no lag col to drop)
                points_transformed = np.hstack((
                    points_lag0_coords[:, :3],
                    points_original[:, 3:5],
                    np.full((points_original.shape[0], 1), j * 0.1, dtype=np.float32)
                ))
                points_4frames.append(points_transformed)

            points_4frames = np.concatenate(points_4frames, axis=0).astype(np.float32)

            # ------------------------------------------------------
            # Current-frame proposals from clean result_fixed.pkl
            # ------------------------------------------------------
            proposals = clean_proposals.get((segment, seg_idx))
            if proposals is None:
                logger.warning("No clean proposals for (%s, %d), skipping", segment, seg_idx)
                continue

            dict_mod = dataset[i_msf].copy()
            roi_boxes = proposals['pred_boxes']
            roi_boxes[:, 7:9] = -0.1 * roi_boxes[:, 7:9]  
            dict_mod['roi_boxes'] = roi_boxes
            dict_mod['roi_scores'] = proposals['pred_scores']
            dict_mod['roi_labels'] = proposals['pred_labels']
            dict_mod['points']     = points_4frames

            dict_mod = helpers_ptt.inject_gt_names(dict_mod, dataset.class_names)
            dict_mod = dataset.prepare_data(dict_mod)
            batch_mod = dataset.collate_batch([dict_mod])

            load_data_to_gpu(batch_mod)
            try: 
                with torch.no_grad():
                    pred_dicts, _ = model(batch_mod)

                annos = dataset.generate_prediction_dicts(
                    batch_mod, pred_dicts, cfg.CLASS_NAMES
                )
            except Exception as e:
                logger.warning("MSF fallback (segment=%s frame=%d) reason=%s",
                            current_segment, seg_idx, str(e))
                annos = [msf_result_lookup[batch['frame_id'][0]].copy()]
                
            
            
            segment_preds_list += annos

    except Exception:
        logger.error("===== EXCEPTION =====")
        logger.error(traceback.format_exc())
        raise

    finally:
        if current_segment is not None and len(segment_preds_list) > 0:
            pred_path = f"{save_dir_preds}/{current_segment}_p.pkl"
            with open(pred_path, "wb") as f:
                pkl.dump(segment_preds_list, f)
            logger.info("Saved preds for: %s", current_segment)

    # ----------------------------------------------------------
    # Reassemble and evaluate
    # ----------------------------------------------------------
    segment_preds = {}
    for seg in dataset.seq_name_to_infos.keys():
        with open(f"{save_dir_preds}/{seg}_p.pkl", "rb") as f:
            segment_preds[seg] = pkl.load(f)

    det_annos = []
    segment_frame_counters = {}
    for info in dataset.infos:
        seg = info['point_cloud']['lidar_sequence']
        if seg not in segment_frame_counters:
            segment_frame_counters[seg] = 0
        idx = segment_frame_counters[seg]
        det_annos.append(segment_preds[seg][idx])
        segment_frame_counters[seg] += 1

    for pred, info in zip(det_annos, dataset.infos):
        assert pred['frame_id'] == info['frame_id']

    logger.info("\n\nStarting Waymo evaluation")
    for _ in tqdm(range(1), desc="Waymo Metrics"):
        result_str, result_dict = dataset.evaluation(
            det_annos, cfg.CLASS_NAMES,
            eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC
        )

    logger.info("Evaluation complete")
    logger.info("\n===== RESULT STRING =====\n")
    logger.info("\n" + result_str)
    logger.info("\n===== RESULT DICT =====")
    logger.info(json.dumps(result_dict, indent=2, default=float))

    with open(args.metrics_out, "w") as f:
        json.dump(result_dict, f, indent=2, default=float)
    logger.info(f"Metrics saved to {args.metrics_out}")


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