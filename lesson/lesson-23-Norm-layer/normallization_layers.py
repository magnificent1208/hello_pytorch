# -*- coding: utf-8 -*-
"""
# @brief      : pytorch中常见的 normalization layers
"""

'''
Why Normalizaiton ?
Coz ICS

常见的Norm方法：1. BN 2.LN(适用于) 3.IN(生成对抗比较多) 4.GN
相同之处是，公式都一样
不同之处是，均值与方差求取方式

BatchNorm：batch方向做归一化，算NHW的均值，对小batchsize效果不好；BN主要缺点是对batchsize的大小比较敏感，由于每次计算均值和方差是在一个batch上，所以如果batchsize太小，则计算的均值、方差不足以代表整个数据分布
LayerNorm：channel方向做归一化，算CHW的均值，主要对RNN作用明显；
InstanceNorm：一个channel内做归一化，算H*W的均值，用在风格化迁移；因为在图像风格化中，生成结果主要依赖于某个图像实例，所以对整个batch归一化不适合图像风格化中，因而对HW做归一化。可以加速模型收敛，并且保持每个图像实例之间的独立。
GroupNorm：将channel方向分group，然后每个group内做归一化，算(C//G)HW的均值；这样与batchsize无关，不受其约束

'''
import torch
import numpy as np
import torch.nn as nn
import sys, os
hello_pytorch_DIR = os.path.abspath(os.path.dirname(__file__)+os.path.sep+".."+os.path.sep+"..")
sys.path.append(hello_pytorch_DIR)
from tools.common_tools import set_seed


set_seed(1)  # 设置随机种子

# ======================================== nn.layer norm
# flag = 1
flag = 0
if flag:
    batch_size = 8
    num_features = 6

    features_shape = (3, 4)

    feature_map = torch.ones(features_shape)  # 2D
    feature_maps = torch.stack([feature_map * (i + 1) for i in range(num_features)], dim=0)  # 3D
    feature_maps_bs = torch.stack([feature_maps for i in range(batch_size)], dim=0)  # 4D

    # feature_maps_bs shape is [8, 6, 3, 4],  B * C * H * W
    # ln = nn.LayerNorm(feature_maps_bs.size()[1:], elementwise_affine=True)
    # ln = nn.LayerNorm(feature_maps_bs.size()[1:], elementwise_affine=False)
    # ln = nn.LayerNorm([6, 3, 4])
    ln = nn.LayerNorm([6, 3])

    output = ln(feature_maps_bs)

    print("Layer Normalization")
    print(ln.weight.shape)
    print(feature_maps_bs[0, ...])
    print(output[0, ...])

# ======================================== nn.instance norm 2d
# flag = 1
flag = 0
if flag:

    batch_size = 3
    num_features = 3
    momentum = 0.3

    features_shape = (2, 2)

    feature_map = torch.ones(features_shape)    # 2D
    feature_maps = torch.stack([feature_map * (i + 1) for i in range(num_features)], dim=0)  # 3D
    feature_maps_bs = torch.stack([feature_maps for i in range(batch_size)], dim=0)  # 4D

    print("Instance Normalization")
    print("input data:\n{} shape is {}".format(feature_maps_bs, feature_maps_bs.shape))

    instance_n = nn.InstanceNorm2d(num_features=num_features, momentum=momentum)

    for i in range(1):
        outputs = instance_n(feature_maps_bs)

        print(outputs)
        # print("\niter:{}, running_mean.shape: {}".format(i, bn.running_mean.shape))
        # print("iter:{}, running_var.shape: {}".format(i, bn.running_var.shape))
        # print("iter:{}, weight.shape: {}".format(i, bn.weight.shape))
        # print("iter:{}, bias.shape: {}".format(i, bn.bias.shape))


# ======================================== nn.grop norm
flag = 1
# flag = 0
if flag:

    batch_size = 2
    num_features = 4
    num_groups = 4   # 3 Expected number of channels in input to be divisible by num_groups

    features_shape = (2, 2)

    feature_map = torch.ones(features_shape)    # 2D
    feature_maps = torch.stack([feature_map * (i + 1) for i in range(num_features)], dim=0)  # 3D
    feature_maps_bs = torch.stack([feature_maps * (i + 1) for i in range(batch_size)], dim=0)  # 4D

    gn = nn.GroupNorm(num_groups, num_features)
    outputs = gn(feature_maps_bs)

    print("Group Normalization")
    print(gn.weight.shape)
    print(outputs[0])