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
    def __init__(self, df, view="Dashboard", mode="train", img_size = 512, use_pose = False, data_path=None):
        self.view = view
        self.df = df
        self.frames = df.frame_index.values
        self.folder_name = df.folder_name.values
        self.label = df.class_name.values
        self.aug = A.Compose([
                   A.Resize(img_size, img_size),
                   A.HorizontalFlip(p=0.25),
                   A.Normalize(mean=[0.], std=[1.]),
                   ToTensorV2()
                ])

        if mode != "train": 
            self.aug = A.Compose([
                       A.Resize(img_size, img_size),
                       A.Normalize(mean=[0.], std=[1.]),
                       ToTensorV2()
                    ]) 
        self.mode = mode
        self.use_pose = use_pose
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
        window = 14 * 4

        frame = self.frames[idx]
        label = self.label[idx]
        fold_name = self.folder_name[idx]
        
        imgs = []
        for view in ['Dashboard', 'Rear_view', 'Right_side_window']:
            fold_name.replace("Dashboard", view)
            for frame_index in range(frame, frame + window + 1, 4):
                filename = f'{self.data_path}/{label}/{fold_name}/img_{frame_index:06d}.jpg'
                
                img = cv2.imread(filename, 0)
                if img is None:
                    print(filename)
                imgs.append(img)

        img = np.array(imgs).transpose(1, 2, 0) 
        img = self.aug(image=img)["image"]

        


        if self.use_pose:

            file_pkl = f'{self.data_path}/{label}/{fold_name}/fake_anno.pkl'
            with open(file_pkl, 'rb') as f:
                pose_data = pickle.load(f)

            
            pose_data = dict(
                total_frames=60,
                label=pose_data['label'],
                keypoint=pose_data['keypoint'][:, frame:frame+60, :], 
                keypoint_score = pose_data['keypoint_score'][:, frame:frame+60, :],
                img_shape=(512, 512))
            
            # pose_data['total_frames'] = 60
            # pose_data['keypoint'] = pose_data['keypoint'][:, frame:frame+60, :]
            # pose_data['keypoint_score'] = pose_data['keypoint_score'][:, frame:frame+60, :]
            
            pose = self.transform_pose(pose_data)
            label = torch.tensor(label)
            
            return img, pose[0], label

        else:
            label = torch.tensor(label)
            return img, label 
