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
    def __init__(self):
        super(Model, self).__init__()
        self.backbone = timm.create_model("tf_efficientnet_b0_ns", pretrained=True, num_classes=1, in_chans=3)

        self.conv_proj = nn.Sequential(
            nn.Conv2d(1280, 512, 1, stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

        self.neck = nn.Sequential(
            nn.Linear(512*3, 512*3),
            nn.BatchNorm1d(512*3),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
        )

        self.triple_layer = nn.Sequential(
            Residual3DBlock(),
        )

        self.pool = GeM()

        self.fc = nn.Linear(512*3, 1)

    def forward(self, images, feature, target=None, mixup_hidden=False, mixup_alpha=0.1, layer_mix=None):
        b, t, h, w = images.shape  # 8, 45, 512, 512

        images = images.view(b * t // 3, 3, h, w)  # 120, 10, 512, 512
        backbone_maps = self.backbone.forward_features(images)  # 120, 1280, 16, 16

        feature_maps = self.conv_proj(backbone_maps)  # 120, 512, 16, 16
        # print(feature_maps.size())
        _, c, h, w = feature_maps.size()
        feature_maps = feature_maps.contiguous().view(b * 3, c, t // 3 // 3, h, w)  # 24, 512, 5, 16, 16
        feature_maps = self.triple_layer(feature_maps)  # 24, 512, 5, 16, 16
        middle_maps = feature_maps[:, :, 2, :, :]  # Get feature frame t: 24, 512, 16, 16
        # after pool: 24, 512, 1, 1 => reshape:  8, 512 *3 => neck: 8, 1024
        # print(self.pool(middle_maps).shape)
        # print(self.pool(middle_maps).reshape(b, -1).shape)
        nn_feature = self.neck(self.pool(middle_maps).reshape(b, -1))
        # cat_features = torch.cat([nn_feature, feature], dim=1)
        cat_features = nn_feature
        # print(cat_features.shape)
        if target is not None:
            cat_features, y_a, y_b, lam = mixup_data(cat_features, target, mixup_alpha)
            y = self.fc(cat_features)
            return y, y_a, y_b, lam
        else:
            y = self.fc(cat_features)
            print(y.shape)
            return y

if __name__ == "__main__":
    model = Model()
    im = torch.randn((8, 45, 512, 512))
    feature = torch.randn((8, 68))
    model(im, feature)
