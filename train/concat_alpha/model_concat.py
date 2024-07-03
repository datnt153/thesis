import timm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmaction.models import STGCN

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1. / p)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(
            self.eps) + ')'


# Replace concat 2 feature img (512) and pose (32) => use alpha mix img feature  (64) and pose feature (64)

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


class Residual3DBlock(nn.Module):
    def __init__(self):
        super(Residual3DBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv3d(512, 512, 3, stride=1, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(512)
        )

        self.block2 = nn.Sequential(
            nn.Conv3d(512, 512, 3, stride=1, padding=1),
            nn.BatchNorm3d(512),
        )

    def forward(self, images):
        short_cut = images
        h = self.block(images)
        h = self.block2(h)

        return F.relu(h + short_cut)


class Model(nn.Module):
    def __init__(self, model_name="tf_efficientnet_b0_ns", h_dim=64):
        super(Model, self).__init__()
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=16, in_chans=3 )
        self.h_dim = h_dim
        self.conv_proj = nn.Sequential(
            nn.Conv2d(1280, 512, 1, stride=1),

            nn.BatchNorm2d(512),
            nn.ReLU(),
        )


        # with 1 view 
        self.neck = nn.Sequential(
            nn.Linear(512*1, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
        )

        self.triple_layer = nn.Sequential(
            Residual3DBlock(),
        )

        self.pool = GeM()
        self.img_fc = nn.Linear(512, h_dim)


        self.stgcn = STGCN(graph_cfg=dict(layout='coco', mode='stgcn_spatial'))
        self.pool_pose = nn.AdaptiveAvgPool2d(1)
        self.fc_pose = nn.Linear(256, h_dim)

        # 32 for pose stgcn and 512 for efficientnet
        self.fc = nn.Linear(h_dim*2, 16)


    def forward(self, images, feature_pose,  target=None, mixup_alpha=0.1):
        # print(f"images shape:{images.shape}")
        # print(f"feature_pose shape:{feature_pose.shape}")
        b, t, h, w = images.shape  # 8, 15, 512, 512
        # print(f"images.shape: {images.shape}")
        #images = images.view(b * t // 3, 3, h, w)  #  40, 3, 512, 512

        images = images.reshape(b * t // 3, 3, h, w)  # 40, 3, 512, 512
        # print(f"images shape:{images.shape}")

        backbone_maps = self.backbone.forward_features(images)  # 10, 1280, 16, 16
        # print(f"backbone_maps shape:{backbone_maps.shape}")

        feature_maps = self.conv_proj(backbone_maps)  # 10, 512,16, 16
        # print(feature_maps.size())

        _, c, h, w = feature_maps.size()
        feature_maps = feature_maps.contiguous().view(b, c, t // 3 , h, w)  
        feature_maps = self.triple_layer(feature_maps)  # 2, 512, 5, 16, 16
        # print(f"feature_map: {feature_maps.shape}")

        middle_maps = feature_maps[:, :, 2, :, :]  # Get feature frame t: 2, 512, 16, 16
        # after pool: 2, 512, 1, 1 => reshape:  8, 1024 => neck: 8, 1024
        # print(f"middle_maps shape: {middle_maps.shape}")
        # print(self.pool(middle_maps).shape)
        # print(self.pool(middle_maps).reshape(b, -1).shape)
        nn_feature = self.neck(self.pool(middle_maps).reshape(b, -1))
        nn_feature = self.img_fc(nn_feature)

        # Pose feature
        # print(f"feature_pose shape: {feature_pose.shape}")
        # print(f"type of feature_pose: {type(feature_pose)}")
        x = self.stgcn(feature_pose)
        # print(f"feature_pose shape: {x.shape}")
        N, M, C, T, V = x.shape
        x = x.view(N * M, C, T, V)
        x = self.pool_pose(x)
        x = x.view(N, M, C)
        pose_feature = x.mean(dim=1)
        pose_feature = self.fc_pose(pose_feature)

        cat_features = torch.cat([nn_feature, pose_feature], dim=1)

        if target is not None:
            cat_features, y_a, y_b, lam = mixup_data(cat_features, target, mixup_alpha)

            y = self.fc(cat_features)
            # print(f"y: {y}")

            return y, y_a, y_b, torch.tensor(lam).to("cuda")
        else:
            y = self.fc(cat_features)
            return y



# model = Model()
#
#
# im = torch.randn((8, 15, 512, 512))
# # Class for pose
# num_joints = 17
# batch_size, num_person, num_frames = 8, 1, 60
#
# #model.init_weights()
# feature_pose = torch.randn(batch_size, num_person,
#                      num_frames, num_joints, 3)
#
# # feature = torch.randn((8, 68))
# y = model(im, feature_pose)
# print(y.shape )
