import timm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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
    def __init__(self, model_name="tf_efficientnet_b0_ns"):
        super(Model, self).__init__()
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=16, in_chans=3)

        self.conv_proj = nn.Sequential(
            nn.Conv2d(1280, 512, 1, stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

        self.neck = nn.Sequential(
            nn.Linear(1536, 1536),
            nn.BatchNorm1d(1536),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
        )

        self.triple_layer = nn.Sequential(
            Residual3DBlock(),
        )

        self.pool = GeM()

        self.fc = nn.Linear(512 * 3, 16)

    def forward(self, images, target=None, mixup_hidden=False, mixup_alpha=0.1, layer_mix=None):
        b, t, h, w = images.shape  # 2, 9, 512, 512
        #images = images.view(b * t // 3, 3, h, w)  # 6, 3, 512, 512
        images = images.reshape(b * t // 3, 3, h, w)  # 6, 3, 512, 512
        backbone_maps = self.backbone.forward_features(images)  # 6, 1280, 16, 16
        feature_maps = self.conv_proj(backbone_maps)  # 6, 512, 16, 16
        _, c, h, w = feature_maps.size()
        feature_maps = feature_maps.contiguous().view(b * 3, c, t // 3 // 3, h, w)  
        feature_maps = self.triple_layer(feature_maps)  # 6, 512, 5, 16, 16
        middle_maps = feature_maps[:, :, 1, :, :]  # Get feature frame t: 6, 512, 16, 16
        # after pool: 16, 512, 1, 1 => reshape:  8, 1024 => neck: 8, 1024
        nn_feature = self.neck(self.pool(middle_maps).reshape(b, -1))
        if target is not None:
            nn_feature, y_a, y_b, lam = mixup_data(nn_feature, target, mixup_alpha)
            y = self.fc(nn_feature)
            return y, y_a, y_b, lam
        else:
            y = self.fc(nn_feature)
            return y

