# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import pickle
import cv2
import mmcv
import numpy as np
import os
import os.path as osp
import shutil
import torch
# from mmaction.apis import (detection_inference, inference_skeleton,
#                            init_recognizer, pose_inference)

from pathlib import Path
from typing import List, Optional, Tuple, Union

import mmengine
import torch.nn as nn

from mmengine.structures import InstanceData
from mmengine.utils import track_iter_progress


def parse_args():
    parser = argparse.ArgumentParser(description='PoseC3D demo')
    # parser.add_argument('video', help='video file/url')
    parser.add_argument(
        '--config',
        default='configs/posec3d/slowonly_r50_ntu120_xsub/joint.py',
        help='skeleton action recognition config file path')
    parser.add_argument(
        '--det-config',
        default='faster-rcnn_r50-caffe_fpn_ms-1x_coco-person.py',
        help='human detection config file path (from mmdet)')
    parser.add_argument(
        '--det-checkpoint',
        default=('https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco-person/'
                 'faster_rcnn_r50_fpn_1x_coco-person_20201216_175929-d022e227.pth'),
        help='human detection checkpoint file/url')
    parser.add_argument(
        '--det-cat-id',
        type=int,
        default=0,
        help='the category id for human detection')

    parser.add_argument(
        '--pose-config',
        default='td-hm_hrnet-w32_8xb64-210e_coco-256x192_infer.py',
        help='human pose estimation config file path (from mmpose)')
    parser.add_argument(
        '--pose-checkpoint',
        default='https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth',
        help='human pose estimation checkpoint file/url')
    parser.add_argument(
        '--det-score-thr',
        type=float,
        default=0.9,
        help='the threshold of human detection score')
    parser.add_argument(
        '--label-map',
        default='tools/data/label_map/nturgbd_120.txt',
        help='label map file')
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='CPU/CUDA device option')
    parser.add_argument(
        '--short-side',
        type=int,
        default=512,
        help='specify the short-side length of the image')
    args = parser.parse_args()
    return args


def frame_extraction(video_path, short_side):
    """Extract frames given video_path.

    Args:
        video_path (str): The video_path.
    """
    # Load the video, extract frames into ./tmp/video_name
    target_dir = osp.join('./tmp', osp.splitext(video_path)[0])
    os.makedirs(target_dir, exist_ok=True)
    # Should be able to handle videos up to several hours
    frame_tmpl = osp.join(target_dir, 'img_{:06d}.jpg')
    vid = cv2.VideoCapture(video_path)
    frames = []
    frame_paths = []
    flag, frame = vid.read()
    cnt = 0
    new_h, new_w = None, None
    while flag:
        if new_h is None:
            h, w, _ = frame.shape
            new_w, new_h = mmcv.rescale_size((w, h), (short_side, np.Inf))

        frame = mmcv.imresize(frame, (new_w, new_h))

        frames.append(frame)
        frame_path = frame_tmpl.format(cnt + 1)
        frame_paths.append(frame_path)

        cv2.imwrite(frame_path, frame)
        cnt += 1
        flag, frame = vid.read()

    return frame_paths, frames


args = parse_args()
det_config = args.det_config
det_checkpoint = args.det_checkpoint

try:
    from mmdet.apis import inference_detector, init_detector
    from mmdet.structures import DetDataSample
except (ImportError, ModuleNotFoundError):
    raise ImportError('Failed to import `inference_detector` and '
                      '`init_detector` from `mmdet.apis`. These apis are '
                      'required in this inference api! ')
if isinstance(det_config, nn.Module):
    det_model = det_config
else:
    det_model = init_detector(
        config=det_config, checkpoint=det_checkpoint, device="cuda:0")


def detection_inference(
        frame_paths: List[str],
        det_score_thr: float = 0.9,
        det_cat_id: int = 0,
        with_score: bool = False) -> tuple:
    results = []
    data_samples = []
    print('Performing Human Detection for each frame')
    for frame_path in track_iter_progress(frame_paths):
        det_data_sample: DetDataSample = inference_detector(det_model, frame_path)
        pred_instance = det_data_sample.pred_instances.cpu().numpy()
        bboxes = pred_instance.bboxes
        scores = pred_instance.scores
        # We only keep human detection bboxs with score larger
        # than `det_score_thr` and category id equal to `det_cat_id`.
        valid_idx = np.logical_and(pred_instance.labels == det_cat_id,
                                   pred_instance.scores > det_score_thr)
        bboxes = bboxes[valid_idx]
        scores = scores[valid_idx]

        if with_score:
            bboxes = np.concatenate((bboxes, scores[:, None]), axis=-1)
        results.append(bboxes)
        data_samples.append(det_data_sample)

    return results, data_samples


pose_config = args.pose_config
pose_checkpoint = args.pose_checkpoint

try:
    from mmpose.apis import inference_topdown, init_model
    from mmpose.structures import PoseDataSample, merge_data_samples
except (ImportError, ModuleNotFoundError):
    raise ImportError('Failed to import `inference_topdown` and '
                      '`init_model` from `mmpose.apis`. These apis '
                      'are required in this inference api! ')
if isinstance(pose_config, nn.Module):
    pose_model = pose_config
else:
    pose_model = init_model(pose_config, pose_checkpoint, 'cuda:1')


def pose_inference(
        frame_paths: List[str],
        det_results: List[np.ndarray],
) -> tuple:
    results = []
    data_samples = []
    print('Performing Human Pose Estimation for each frame')
    for f, d in track_iter_progress(list(zip(frame_paths, det_results))):
        pose_data_samples: List[PoseDataSample] \
            = inference_topdown(pose_model, f, d[..., :4], bbox_format='xyxy')
        pose_data_sample = merge_data_samples(pose_data_samples)
        pose_data_sample.dataset_meta = pose_model.dataset_meta
        # make fake pred_instances
        if not hasattr(pose_data_sample, 'pred_instances'):
            num_keypoints = pose_model.dataset_meta['num_keypoints']
            pred_instances_data = dict(
                keypoints=np.empty(shape=(0, num_keypoints, 2)),
                keypoints_scores=np.empty(shape=(0, 17), dtype=np.float32),
                bboxes=np.empty(shape=(0, 4), dtype=np.float32),
                bbox_scores=np.empty(shape=(0), dtype=np.float32))
            pose_data_sample.pred_instances = InstanceData(
                **pred_instances_data)

        poses = pose_data_sample.pred_instances.to_dict()
        results.append(poses)
        data_samples.append(pose_data_sample)

    return results, data_samples


def main(video_index):
    video_path = file_paths[video_index]
    # tmp_frame_dir = osp.dirname(frame_paths[0])
    tmp_frame_dir = target_dir = osp.join('./tmp', osp.splitext(video_path)[0])
    print(f"tmp_frame_dir: {tmp_frame_dir}")
    # Create a Path object for the folder
    folder = Path(tmp_frame_dir)

    # Check if the file "fake_anno.pkl" exists within the folder
    file_to_check = folder / "fake_anno.pkl"
    if file_to_check.exists():
        print("The file 'fake_anno.pkl' exists in the folder:" + tmp_frame_dir)
        return
    else:
        print("The file 'fake_anno.pkl' does not exist in the folder.")

    # video_path = args.video
    frame_paths, original_frames = frame_extraction(video_path, args.short_side)
    num_frame = len(frame_paths)
    h, w, _ = original_frames[0].shape



    # Get Human detection results
    det_results, _ = detection_inference(frame_paths, args.det_score_thr, args.det_cat_id)
    torch.cuda.empty_cache()

    # Get Pose estimation results.
    pose_results, pose_data_samples = pose_inference(frame_paths, det_results)
    torch.cuda.empty_cache()

    fake_anno = dict(
        frame_dir='',
        label=-1,
        img_shape=(h, w),
        original_shape=(h, w),
        start_index=0,
        modality='Pose',
        total_frames=num_frame)

    # align the num_person among frames
    num_persons = max([pose['keypoints'].shape[0] for pose in pose_results])
    num_points = pose_results[0]['keypoints'].shape[1]
    num_frames = len(pose_results)
    keypoints = np.zeros((num_persons, num_frames, num_points, 2),
                         dtype=np.float32)
    scores = np.zeros((num_persons, num_frames, num_points), dtype=np.float32)

    for f_idx, frm_pose in enumerate(pose_results):
        frm_num_persons = frm_pose['keypoints'].shape[0]
        for p_idx in range(frm_num_persons):
            keypoints[p_idx, f_idx] = frm_pose['keypoints'][p_idx]
            scores[p_idx, f_idx] = frm_pose['keypoint_scores'][p_idx]

    fake_anno['keypoint'] = keypoints
    fake_anno['keypoint_score'] = scores

    # save fake_anno to pkl
    with open(f'{tmp_frame_dir}/fake_anno.pkl', 'wb') as f:
        pickle.dump(fake_anno, f)


import os

# List of folder names from 1 to 17
folder_names = [str(i) for i in range(16)]

# Initialize an empty list to store the file paths
file_paths = []

# Iterate over each folder
for folder_name in folder_names:
    folder_path = f"SetA1_clip/{folder_name}"
    if os.path.isdir(folder_path):
        # Walk through the folder directory tree
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                file_paths.append(file_path)

# main(0)
from pathlib import Path
from tqdm import tqdm, auto
from multiprocessing import pool

num_files = len(file_paths)
start_idx = 0
end_idx = num_files
# results = pool.ThreadPool(16).imap(main, list(range(start_idx, end_idx)))
# pbar = auto.tqdm(results, total=end_idx - start_idx)
# for _ in pbar:
#     pass



for i in range(end_idx):
    main(i)
