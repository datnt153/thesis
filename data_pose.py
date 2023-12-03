import albumentations as A
import numpy as np
import cv2
from torch.utils.data import DataLoader, Dataset
from albumentations.pytorch import ToTensorV2
import pandas as pd
import random
import torch
import copy
import os 
import pickle
from mmaction.datasets.transforms import FormatGCNInput, PackActionInputs

from mmaction.datasets.transforms import  GenSkeFeat, PoseDecode, PreNormalize2D, UniformSampleFrames



class MyDataset(Dataset):
    def __init__(self, df, view="Dashboard", data_path=None):
        self.view = view
        self.df = df
        self.frames = df.frame_index.values
        self.folder_name = df.folder_name.values
        self.label = df.class_name.values
        self.data_path = data_path

    def __len__(self):
        return len(self.df)
    
    def transform_pose(self, pose: dict):
        pre_normalize_2d = PreNormalize2D(img_shape=(512, 512))
        inp = pre_normalize_2d(pose)
        gen_ske_feat = GenSkeFeat(dataset='coco', feats=['j'])
        ret2 = gen_ske_feat(inp)
        # sampling = UniformSampleFrames(clip_len=60)
        # results = sampling(ret2)
        # pose_decode = PoseDecode()
        # decode_results = pose_decode(ret2)
        format_shape = FormatGCNInput(num_person=1)
        results = format_shape(ret2)
        transform = PackActionInputs()
        results = transform(results)
        return results['inputs']


    def __getitem__(self, idx):
        frame = self.frames[idx]
        label = self.label[idx]
        fold_name = self.folder_name[idx]
        

        file_pkl = f'{self.data_path}/{label}/{fold_name}/fake_anno.pkl'
        with open(file_pkl, 'rb') as f:
            pose_data = pickle.load(f)

        
        pose_data = dict(
            total_frames=60,
            label=pose_data['label'],
            keypoint=pose_data['keypoint'][:, frame:frame+60, :], 
            keypoint_score = pose_data['keypoint_score'][:, frame:frame+60, :],
            img_shape=(512, 512))
        
        
        pose = self.transform_pose(pose_data)
        label = torch.tensor(label)
        
        return pose[0], label

