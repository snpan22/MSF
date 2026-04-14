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
    parser.add_argument('--cfg_file', required=True)
    parser.add_argument('--ckpt', required=True)
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
    cfg_from_yaml_file(args.cfg_file, cfg)

    logger.info("Building dataset (GT only, no model)...")
    
    
    CFG_FILE = args.cfg_file
    CKPT = args.ckpt

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
    logger.info(f'Test set length: {len_test}')
    model = build_network(
        model_cfg=cfg.MODEL,
        num_class=len(cfg.CLASS_NAMES),
        dataset=dataset
    )

    logger.info(f'Loading checkpoint from: {CKPT}')
    model.load_params_from_file(filename=CKPT, logger=logger, to_cpu=False)
    model.cuda()
    model.eval()

    
    
    global_to_segment = {}
    segment_frame_counters = {}
    for i, info in enumerate(dataset.infos):
        seg = info['point_cloud']['lidar_sequence']
        if seg not in segment_frame_counters:
            segment_frame_counters[seg] = 0
        global_to_segment[i] = (seg, segment_frame_counters[seg])
        segment_frame_counters[seg] += 1


    ordered_segments = [info['point_cloud']['lidar_sequence'] for info in dataset.infos]
    ordered_segments = list(dict.fromkeys(ordered_segments))  # remove duplicates while preserving order


    save_dir_preds = args.pred_dir
    os.makedirs(save_dir_preds, exist_ok=True)

    completed = set(
        f[:-4] for f in os.listdir(save_dir_preds)
    )
    
    result_file = "output/cfgs/waymo_models/msf_4frames/default/eval/eval_with_train/epoch_6/val/result.pkl"
    
    
    with open(result_file, "rb") as f:
        result = pkl.load(f)

    history = args.history
    segment_preds_list = []
    current_segment = None
    used_batch = None

    try: 
        for (i_msf,batch)  in enumerate(test_loader):
            
            segment = batch['frame_id'][0].rsplit("_", 1)[0]
            seg_idx = global_to_segment[i_msf][1]

            # skip completed segments immediately
            if f"{segment}_p" in completed:
                continue

            # initialize or switch segment
            if current_segment != segment:

                # save previous segment if it exists
                if current_segment is not None:
                    pred_path = f"{save_dir_preds}/{current_segment}_p.pkl"
                    with open(pred_path, "wb") as f:
                        pkl.dump(segment_preds_list, f)
                    # print(f"saved {current_segment}")
                    logger.info("Saved preds for : %s", current_segment)


                segment_preds_list = []
                current_segment = segment
                # print(current_segment)

                # load spoof dataset
                seg_dataset_file = f"{args.dataset}/{current_segment}_d.pkl"
                with open(seg_dataset_file, "rb") as f:
                    seg_dataset = pkl.load(f)
            
                    
            assert batch['frame_id'][0] == seg_dataset[seg_idx]['frame_id']

            
            past_frames_list = [max(seg_idx - i, 0) for i in range(history)]
            
            # print(past_frames_list)
            
            
            
            #update points structure TODO
            points_4frames = []
            
                # print(seg_idx, past_frames_list[0])
            poses = dataset[i_msf]['poses']
            pose_lag0 = poses[:4]
            pose_lag0_inv = np.linalg.inv(pose_lag0)
            for j,frame in enumerate(past_frames_list):
                if(j==0):
                    points_4frames.append(seg_dataset[frame]['points'])
                    continue
                # lag = seg_idx - frame
                # print(j, lag)
                pose = poses[0+j*4:4+j*4]

                points_original = seg_dataset[frame]['points']
                ones = np.ones(points_original.shape[0]).reshape(-1, 1)
                points_hom = np.hstack((points_original[:, :3], ones))

                points_lag0 = points_hom @ pose.T @ pose_lag0_inv.T

                # print(lag*0.1*ones)

                points_transformed = np.hstack((points_lag0[:, :-1], points_original[:, 3:-1], j*0.1*ones))

                # print(points_transformed[:, -1])
                points_4frames.append(points_transformed)


            points_4frames = np.concatenate(points_4frames, axis = 0)
            
            
            
            dict_mod = dataset[i_msf].copy()
            roi_boxes = seg_dataset[seg_idx]['pred_boxes'].cpu().numpy().copy()
            roi_boxes[:, 7:9] = -0.1 * roi_boxes[:, 7:9]   
            dict_mod['roi_boxes'] = roi_boxes
            dict_mod['roi_scores'] = seg_dataset[seg_idx]['pred_scores'].cpu().numpy()
            dict_mod['roi_labels'] = seg_dataset[seg_idx]['pred_labels'].cpu().numpy()
            dict_mod['points'] = points_4frames
            
            dict_mod = helpers_ptt.inject_gt_names(dict_mod, dataset.class_names)
            dict_mod = dataset.prepare_data(dict_mod)
            batch_mod = dataset.collate_batch([dict_mod])
            
            load_data_to_gpu(batch_mod)

            try:
                with torch.no_grad():
                    pred_dicts, _ = model(batch_mod)
                annos = dataset.generate_prediction_dicts(
                    batch_mod,
                    pred_dicts,
                    cfg.CLASS_NAMES
                )

            except Exception as e:
                logger.info(
                    "MSF fallback (segment=%s frame=%d) reason=%s",
                    current_segment, seg_idx, str(e)
                )
                
                clean_pred = result[i_msf]
                assert batch['frame_id'][0] == clean_pred['frame_id']
                annos = [clean_pred.copy()]  
            # print(f"================================MAKING PTT PREDICTION {seg_idx} ===============================")
            

            segment_preds_list+=annos
            
            
    except Exception:
        logger.error("===== EXCEPTION =====")
        logger.error(traceback.format_exc())
        raise  
        
    finally:
        if current_segment is not None and len(segment_preds_list) > 0:
            pred_path = f"{save_dir_preds}/{current_segment}_p.pkl"
            with open(pred_path, "wb") as f:
                pkl.dump(segment_preds_list, f)
            logger.info("Saved preds for : %s", current_segment)

            
    
    

    pred_dir = args.pred_dir

    segment_preds = {}
    unique_segment_names = dataset.seq_name_to_infos.keys()
    for segment_name in unique_segment_names:
        with open(f"{pred_dir}/{segment_name}_p.pkl", "rb") as f:
            segment_preds[segment_name] = pkl.load(f)
    infos = dataset.infos

    det_annos = []

    segment_frame_counters = {}

    for info in infos:
        segment = info['point_cloud']['lidar_sequence']
        
        if segment not in segment_frame_counters:
            segment_frame_counters[segment] = 0
        
        idx = segment_frame_counters[segment]
        
        det_annos.append(segment_preds[segment][idx])
        
        segment_frame_counters[segment] += 1
    for pred, info in zip(det_annos, dataset.infos):
        assert pred['frame_id'] == info['frame_id']
    # --------------------------------------------------------
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