import timm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmaction.models import STGCN



def mixup_data(x, y, alpha):
    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam




class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        #self.stgcn = STGCN(graph_cfg=dict(layout='coco', mode='stgcn_spatial'), gcn_adaptive='init', gcn_with_res=True, tcn_type='mstcn')
        self.stgcn = STGCN(graph_cfg=dict(layout='coco', mode='stgcn_spatial'))
        self.pool_pose = nn.AdaptiveAvgPool2d(1)
        self.fc_pose = nn.Linear(256, 32)
        # 32 for pose stgcn and 512 for efficientnet
        self.fc = nn.Linear(32, 16)


    def forward(self, feature_pose,  target=None, mixup_alpha=0.1):
        # Pose feature
        # print(f"feature_pose shape: {feature_pose.shape}")
        # print(f"type of feature_pose: {type(feature_pose)}")
        x = self.stgcn(feature_pose)
        # print(f"feature_pose shape: {x.shape}")
        # N: batch size (48)
        # C: Number of channels (3)
        # T: number of frame per video (60)
        # V: Number of nodes per person per frame (17)
        # M: number of persons considered (1)
        N, M, C, T, V = x.shape
        x = x.view(N * M, C, T, V)
        x = self.pool_pose(x)
        x = x.view(N, M, C)
        pose_feature = x.mean(dim=1) 
        cat_features = self.fc_pose(pose_feature)

        if target is not None:
            cat_features, y_a, y_b, lam = mixup_data(cat_features, target, mixup_alpha)
            y = self.fc(cat_features)
            
            return y, y_a, y_b, torch.tensor(lam).to("cuda")
        else:
            y = self.fc(cat_features)
            return y



# model = Model()


# im = torch.randn((8, 15, 512, 512))
# # Class for pose 
# num_joints = 17
# batch_size, num_person, num_frames = 8, 1, 60

# #model.init_weights()
# feature_pose = torch.randn(batch_size, num_person,
#                      num_frames, num_joints, 3)

# # feature = torch.randn((8, 68))
# y = model(im, feature_pose)
# print(y.shape )
