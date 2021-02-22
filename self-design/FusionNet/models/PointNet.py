# coding:utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import config

def conv_bn(inp, oup, kernel, stride=1, activation='relu'):
    seq = nn.Sequential(
        nn.Conv2d(inp, oup, kernel, stride),
        nn.BatchNorm2d(oup)
    )
    if activation == 'relu':
        seq.add_module('2', nn.LeakyReLU())
    return seq


def fc_bn(inp, oup):
    return nn.Sequential(
        nn.Linear(inp, oup),
        nn.BatchNorm1d(oup),
        nn.LeakyReLU()
    )


class PointNet_SA_module_basic(nn.Module):
    def __init__(self):
        super(PointNet_SA_module_basic, self).__init__()

    def index_points(self, points, idx):
        """
        Description:
            this function select the specific points from the whole points according to the idx.
        Input:
            points: input points data, [B, N, C]
            idx: sample index data, [B, D1, D2, ..., Dn]
        Return:
            new_points:, indexed points data, [B, D1, D2, ..., Dn, C]
        """
        device = points.device
        B = points.shape[0]
        view_shape = list(idx.shape)
        view_shape[1:] = [1] * (len(view_shape) - 1)
        repeat_shape = list(idx.shape)
        repeat_shape[0] = 1
        batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
        new_points = points[batch_indices, idx, :]
        return new_points

    def square_distance(self, src, dst):
        """
        Description:
            just the simple Euclidean distance fomula，(x-y)^2,
        Input:
            src: source points, [B, N, C]
            dst: target points, [B, M, C]
        Output:
            dist: per-point square distance, [B, N, M]
        """
        B, N, _ = src.shape
        _, M, _ = dst.shape
        dist = -2 * torch.matmul(src, dst.permute(0, 2, 1).contiguous())
        dist += torch.sum(src ** 2, -1).view(B, N, 1)
        dist += torch.sum(dst ** 2, -1).view(B, 1, M)
        return dist

    def farthest_point_sample(self, xyz, npoint):
        """
        Description:
            first we choose a point from the point set randomly, at the same time,
            see it as a centroid.the calculate the distance of the point and any others,
            and choose the farthest as the second centroid.
            repeat until the number of choosed point has arrived npoint.
        Input:
            xyz: pointcloud data, [B, N, C]
            npoint: number of samples
        Return:
            centroids: the index sampled pointcloud data, [B, npoint, C]
        """
        device = xyz.device
        B, N, C = xyz.shape
        Np = npoint
        centroids = torch.zeros(B, Np, dtype=torch.long).to(device)
        distance = torch.ones(B, N).to(device) * 1e10
        farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
        batch_indices = torch.arange(B, dtype=torch.long).to(device)
        for i in range(Np):
            centroids[:, i] = farthest
            centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
            dist = torch.sum((xyz - centroid) ** 2, -1)
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = torch.max(distance, -1)[1]
        return centroids

    def farthest_point_sample_uniform(self, xyz, npoint):
        """
           Description:
                different with the front function.the function choose the next centroid by
                calculate the distance of one point with other centroids, rather than other point.
                finally, get the max distance.
           Input:
               xyz: pointcloud data, [B, N, C]
               npoint: number of samples
           Return:
               centroids: sampled pointcloud data, [B, npoint, C]
        """
        pass

    def knn(self, xyz, npoint):
        """
           Description:
               first we choose a point from the point set randomly, at the same time,
               see it as a centroid.the calculate the distance of the point and any others,
               and choose the farthest as the second centroid.
               repeat until the number of choosed point has arrived npoint.
           Input:
               xyz: pointcloud data, [B, N, C]
               npoint: number of samples
           Return:
               centroids: sampled pointcloud data, [B, npoint, C]
       """
        pass

    def ball_query(self, radius, nsample, xyz, new_xyz):
        """
        Input:
            radius: local region radius
            nsample: max sample number in local region
            xyz: all points, [B, N, C]
            new_xyz: query points, [B, Np, C]
        Return:
            group_idx: grouped points index, [B, Np, Ns]
        """
        device = xyz.device
        B, N, C = xyz.shape
        _, Np, _ = new_xyz.shape
        Ns = nsample
        group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, Np, 1])
        sqrdists = self.square_distance(new_xyz, xyz)
        group_idx[sqrdists > radius ** 2] = N
        group_idx = group_idx.sort(dim=-1)[0][:, :, :Ns]
        group_first = group_idx[:, :, 0].view(B, Np, 1).repeat([1, 1, Ns])
        mask = group_idx == N
        group_idx[mask] = group_first[mask]
        return group_idx

    def sample_and_group(self, npoint, radius, nsample, xyz, points):
        """
        Input:
            npoint: the number of points that make the local region.
            radius: the radius of the local region
            nsample: the number of points in a local region
            xyz: input points position data, [B, N, C]
            points: input points data, [B, N, D]
        Return:
            new_xyz: sampled points position data, [B, 1, C]
            new_points: sampled points data, [B, 1, N, C+D]
        """
        B, N, C = xyz.shape
        Np = npoint
        assert isinstance(Np, int)

        new_xyz = self.index_points(xyz, self.farthest_point_sample(xyz, npoint))
        idx = self.ball_query(radius, nsample, xyz, new_xyz)
        grouped_xyz = self.index_points(xyz, idx)
        grouped_xyz -= new_xyz.view(B, Np, 1, C)  # the points of each group will be normalized with their centroid
        if points is not None:
            grouped_points = self.index_points(points, idx)
            new_points = torch.cat([grouped_xyz, grouped_points], dim=-1)
        else:
            new_points = grouped_xyz
        return new_xyz, new_points

    def sample_and_group_all(self, xyz, points):
        """
        Description:
            Equivalent to sample_and_group with npoint=1, radius=np.inf, and the centroid is (0, 0, 0)
        Input:
            xyz: input points position data, [B, N, C]
            points: input points data, [B, N, D]
        Return:
            new_xyz: sampled points position data, [B, 1, C]
            new_points: sampled points data, [B, 1, N, C+D]
        """
        device = xyz.device
        B, N, C = xyz.shape
        new_xyz = torch.zeros(B, 1, C).to(device)
        grouped_xyz = xyz.view(B, 1, N, C)
        if points is not None:
            new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
        else:
            new_points = grouped_xyz
        return new_xyz, new_points


class Pointnet_SA_module(PointNet_SA_module_basic):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):

        super(Pointnet_SA_module, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all

        self.conv_bns = nn.Sequential()
        in_channel += 3  # +3是因为points 与 xyz concat的原因
        for i, out_channel in enumerate(mlp):
            m = conv_bn(in_channel, out_channel, 1)
            self.conv_bns.add_module(str(i), m)
            in_channel = out_channel

    def forward(self, xyz, points):
        """
        Input:
            xyz: the shape is [B, N, 3]
            points: thes shape is [B, N, D], the data include the feature infomation
        Return:
            new_xyz: the shape is [B, Np, 3]
            new_points: the shape is [B, Np, D']
        """

        if self.group_all:
            new_xyz, new_points = self.sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = self.sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        new_points = new_points.permute(0, 3, 1, 2).contiguous()  # change size to (B, C, Np, Ns), adaptive to conv
        # print("1:", new_points.shape)
        new_points = self.conv_bns(new_points)
        # print("2:", new_points.shape)
        new_points = torch.max(new_points, 3)[0]  # 取一个local region里所有sampled point特征对应位置的最大值。

        new_points = new_points.permute(0, 2, 1).contiguous()
        # print(new_points.shape)
        return new_xyz, new_points


class Pointnet_SA_MSG_module(PointNet_SA_module_basic):
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        super(Pointnet_SA_MSG_module, self).__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list

        assert len(self.radius_list) == len(self.nsample_list)

        self.in_channel = in_channel
        self.sequentials = nn.ModuleList()
        for sid, mlp in enumerate(mlp_list):
            seq = nn.Sequential()
            self.in_channel = in_channel + 3
            for mid, out_channel in enumerate(mlp):
                m = conv_bn(self.in_channel, out_channel, 1)
                seq.add_module(str(mid), m)
                self.in_channel = out_channel
            self.sequentials.append(seq)

    def forward(self, xyz, points):
        """
        Input:
            xyz: the shape is [B, N, 3]
            points: the shape is [B, N, D]
        Return:
            new_xyz: the shape is [B, Np, 3]
            new_ points: the shape is [B, Np, D']
        """
        (B, N, C), Np = xyz.shape, self.npoint
        new_xyz = self.index_points(xyz, self.farthest_point_sample(xyz, Np))  # B, Np, C
        cat_new_points = []
        for i, radius in enumerate(self.radius_list):
            grouped_idx = self.ball_query(radius, self.nsample_list[i], xyz, new_xyz)  # B, Np, Ns
            grouped_xyz = self.index_points(xyz, grouped_idx)
            grouped_xyz -= new_xyz.view(B, Np, 1, C)  # B, Np, Ns , C
            if points is None:
                grouped_points = grouped_xyz  # B, Np, Ns, C
            else:
                grouped_points = self.index_points(points, grouped_idx)  # B, Np, Ns, D
                grouped_points = torch.cat([grouped_xyz, grouped_points], dim=-1)  # B, Np, Ns, C+D

            grouped_points = grouped_points.permute(0, 3, 2, 1).contiguous()
            new_points = self.sequentials[i](grouped_points)
            new_points = torch.max(new_points, 2)[0]  # B, D', Np
            cat_new_points.append(new_points)
        new_points = torch.cat(cat_new_points, dim=1).permute(0, 2, 1).contiguous()  # B, Np, D'
        return new_xyz, new_points


class PointNet_BA_module:

    def __init__(self, block):

        self.conv1 = conv_bn(block[0], block[1])
        self.conv2 = conv_bn(block[1], block[2])
        self.conv3 = conv_bn(block[2], block[3], activation=None)

    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = F.max_pool2d(x, [config.num_point, 1])
        return x

