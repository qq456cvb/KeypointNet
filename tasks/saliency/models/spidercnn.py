import os, sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from . import pytorch_utils as pt_utils

import os, sys


def _knn_indices(feat, k, centroid=None, dist=False):
    '''
    feat : B x C x N
    centroid : B x C x n
    dist : whether return dist value
    k : int
    return
    knn_indices : B x n x k
    dist : B x n x dist
    '''
    if centroid is None:
        centroid = feat
    pow2_feat = torch.sum(feat.pow(2), dim=1, keepdim=True) # B x 1 x N
    pow2_centroid = torch.sum(centroid.pow(2), dim=1, keepdim=True) # B x 1 x n
    centroid_feat = torch.bmm(centroid.transpose(1,2), feat).mul_(-2) # B x n x N
    pow2_centroid = pow2_centroid.permute(0, 2, 1)
    distances = centroid_feat + pow2_centroid + pow2_feat
    k_dist, indices = torch.topk(distances, k, dim=-1, largest=False, sorted=False)
    if dist:
        return indices, k_dist
    else:
        return indices
    # r_a = torch.sum(feat.pow(2), dim=1, keepdim=True)
    # dis = torch.bmm(feat.transpose(1,2), feat).mul_(-2)
    # dis.add_(r_a.transpose(1,2) + r_a)

    # _, indices = torch.topk(dis, k, dim=-1, largest=False, sorted=False)

    # return indices

def _indices_group(feat, indices):
    '''
    input
    feat : B x C x N
    indices : B x n x k
    output
    group_feat : B x C x N x k
    '''
    B, C, N = feat.size()
    _, n, k = indices.size()

    indices = indices.unsqueeze(1).expand(B, C, n, k)
    group_feat = feat.unsqueeze(-1).expand(B, C, N, k)
    group_feat = torch.gather(group_feat, 2, indices)

    return group_feat # B x C x n x k

def _knn_group(feat, k):
    '''
    input
    feat : B x C x N
    k : int
    output :
    group_feat : B x C x N x k
    '''
    B, C, N = feat.size()
    knn_indices = _knn_indices(feat, k) # B x N x k
    group_feat = _indices_group(feat, knn_indices)


    return group_feat


class _BaseSpiderConv(nn.Module):
    def __init__(self, in_channel, out_channel, taylor_channel, K_knn):
        super().__init__()
        self.K_knn = K_knn
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.taylor_channel = taylor_channel

        self.conv1 = pt_utils.Conv2d(19, self.taylor_channel, bn=False, activation=None)

        self.conv2 = pt_utils.Conv2d(
            in_channel * taylor_channel,
            out_channel,
            kernel_size=[1, K_knn],
            bn=True
        )

    def forward(self, feat, idx, group_pc):
        '''
        feat : B x in_channel x N
        idx(knn_indices) : B x N x k
        group_pc : B x 3 x N x k
        return:
        feat : B x out_channel x N
        '''
        B, in_channel, N = feat.size()
        _, _, k = idx.size()

        assert k == self.K_knn, 'illegal k'

        group_feat = _indices_group(feat, idx)  # B x inchannel x N x k

        X = group_pc[:, 0, :, :].unsqueeze(1)
        Y = group_pc[:, 1, :, :].unsqueeze(1)
        Z = group_pc[:, 2, :, :].unsqueeze(1)

        XX, YY, ZZ = X ** 2, Y ** 2, Z ** 2
        XXX, YYY, ZZZ = XX * X, YY * Y, ZZ * Z
        XY, XZ, YZ = X * Y, X * Z, Y * Z
        XXY, XXZ, YYZ, YYX, ZZX, ZZY, XYZ = X * XY, X * XZ, Y * YZ, Y * XY, Z * XZ, Z * YZ, XY * Z

        group_XYZ = torch.cat([
            X, Y, Z, XX, YY, ZZ, XXX, YYY, ZZZ, \
            XY, XZ, YZ, XXY, XXZ, YYZ, YYX, ZZX, ZZY, XYZ
        ], dim=1)  # B x 20 x N x k

        taylor = self.conv1(group_XYZ)  # B x taylor_channel x N x k

        group_feat = group_feat.unsqueeze(2)  # B x inchannel x 1 x N x k
        taylor = taylor.unsqueeze(1)  # B x 1 x taylor_channel x N x k

        group_feat = torch.mul(group_feat, taylor).view(B, self.in_channel * self.taylor_channel, N, k)

        group_feat = self.conv2(group_feat)  # B x out_channel x N x 1

        group_feat = group_feat.squeeze(-1)

        return group_feat


# if __name__ == '__main__':
#     in_channel = 3
#     out_channel = 6
#     taylor_channel = 9
#     k = 3
#     batch_size = 3
#     num_points = 10
#     model = _BaseSpiderConv(in_channel, out_channel, taylor_channel, k)
#
#     pc = torch.randn(batch_size, 3, num_points)
#     feat = torch.randn(batch_size, in_channel, num_points)
#     idx = _knn_indices(pc, k)
#     group_pc = _indices_group(pc, idx)
#     pc = pc.unsqueeze(-1).expand(batch_size, 3, num_points, k)
#     group_pc = group_pc - pc
#
#     output = model(feat, idx, group_pc)
#     print(output.size())


class Spidercnn_seg_feature(nn.Module):
    def __init__(self, K_knn: int = 16, taylor_channel: int = 3, withnor=False):
        super().__init__()

        self.K_knn = K_knn
        self.taylor_channel = taylor_channel
        self.withnor = withnor
        self.inchannel = 6 if withnor else 3

        self.spiderconv1 = _BaseSpiderConv(self.inchannel, 32, self.taylor_channel, self.K_knn)
        self.spiderconv2 = _BaseSpiderConv(32, 64, self.taylor_channel, self.K_knn)
        self.spiderconv3 = _BaseSpiderConv(64, 128, self.taylor_channel, self.K_knn)
        self.spiderconv4 = _BaseSpiderConv(128, 256, self.taylor_channel, self.K_knn)

    def forward(self, pc):
        '''
        pc_withnor : B x N x 6
        or pc_withoutnor : B x N x3
        '''
        assert pc.size()[2] == self.inchannel, 'illegal input pc size:{}'.format(pc.size())
        B, N, _ = pc.size()
        pc = pc.permute(0, 2, 1)
        pc_xyz = pc[:, 0:3, :]
        idx = _knn_indices(pc_xyz, k=self.K_knn)
        grouped_xyz = _indices_group(pc_xyz, idx)  # B x 3 x N x k

        grouped_pc = pc_xyz.unsqueeze(-1).expand(B, 3, N, self.K_knn)
        grouped_pc = grouped_xyz - grouped_pc

        feat_1 = self.spiderconv1(pc, idx, grouped_pc)  # B x 64 x N
        feat_2 = self.spiderconv2(feat_1, idx, grouped_pc)
        feat_3 = self.spiderconv3(feat_2, idx, grouped_pc)
        feat_4 = self.spiderconv4(feat_3, idx, grouped_pc)

        cat_feat = torch.cat([feat_1, feat_2, feat_3, feat_4], dim=1)  # B x 480 x N
        point_feat = cat_feat
        cat_feat = torch.topk(cat_feat, 2, dim=2)[0]  # B x 480 x 2
        cat_feat = cat_feat.view(B, -1)  # B x 960

        return cat_feat, point_feat


class Spidercnn_seg_classifier(nn.Module):
    def __init__(self, num_parts: int = 50):
        super().__init__()
        self.num_parts = num_parts
        self.drop = nn.Dropout2d(p=0.2)
        self.fc1 = pt_utils.Conv2d(1440, 256, bn=True)
        self.fc2 = pt_utils.Conv2d(256, 256, bn=True)
        self.fc3 = pt_utils.Conv2d(256, 128, bn=True)
        self.fc4 = pt_utils.Conv2d(128, self.num_parts, bn=False, activation=None)

    def forward(self, feat):
        feat = self.drop(self.fc1(feat))
        feat = self.drop(self.fc2(feat))
        feat = self.drop(self.fc3(feat))
        feat = self.fc4(feat)

        return feat


class Spidercnn_seg_fullnet(nn.Module):
    def __init__(self,  num_parts: int = 128, K_knn: int = 16, taylor_channel: int = 3, withnor=False):
        super().__init__()

        self.K_knn = K_knn
        self.taylor_channel = taylor_channel
        self.withnor = withnor
        self.num_parts = num_parts

        self.feature_extractor = Spidercnn_seg_feature(self.K_knn, self.taylor_channel, self.withnor)
        self.classifier = Spidercnn_seg_classifier(self.num_parts)

    def forward(self, x):
        '''
        batch_data : dict contains ['pc'], ['one_hot_labels']
        output : B x num_parts x N
        '''
        pc = x.permute(0, 2, 1)  # B x N x 6/3
        # one_hot_labels = batch_data['one_hot_labels']  # B x num_classes(16)
        # _, num_classes = one_hot_labels.size()
        B, N, _ = pc.size()
        # one_hot_labels = one_hot_labels.unsqueeze(-1).expand(B, num_classes, N)

        global_feat, point_feat = self.feature_extractor(pc)  # B x 960; B x 480 x N
        global_feat = global_feat.unsqueeze(-1).expand(B, 960, N)

        global_point_feat = torch.cat([global_feat, point_feat], dim=1)
        global_point_feat = global_point_feat.unsqueeze(-1)  # B x 1456 x N x 1

        scores = self.classifier(global_point_feat)

        embedding = scores.squeeze(-1).permute(0, 2, 1)
        return embedding# B x num_parts x N


if __name__ == '__main__':
    input = torch.randn((8, 3, 2048)).cuda()
    net = Spidercnn_seg_fullnet(num_parts=64).cuda()
    print(net(input).shape)