# -*- coding: utf-8 -*-
"""
# @brief      : bn与权值初始化

概念 Batch Normalization 批标准化
批 - 一批数据，通常为mini-batch
标准化 - 0 均值 1 方差

    * according to Batch Norm 原论文，BN的优点如下：
        1. 更大的学习率，加速模型收敛
        2. 不用精心设置权值初始化 #数据尺度过大过小 会导致模型无法训练，但是加了BN层就会把数据规范化
        3. 不用dropout或者较小的dropout
        4. 不用L2或者较小的weight decay策略
        5. 不用LRN(local response normalization)

BN层计算方式(流程)：
S1 算均值
S2 算方差
S3 套标准化公式
S4 scale&shift步骤 #affine transform，加入两个尺度变换的参数gamma和beta，调整数据分布，从而增强capacity
    类似于y=scale∗x+shift的形式
--------------回顾--------------------------
Internal Co-variate shift 问题

internal -- 因为是在神经网络层与层之间 输入输出产生的问题
co-variate -- 协变量 #不为实验者所操作的独立变量
shift -- 描述的是这个问题的形态

定义：
    深度神经网络涉及到很多层的叠加，
    而每一层的参数更新会导致上层的输入数据分布发生变化，通过层层叠加，
    高层的输入分布变化会非常剧烈，这就使得高层需要不断去重新适应底层的参数更新。
    为了训好模型，我们需要非常谨慎地去设定学习率、初始化权重、以及尽可能细致的参数更新策略。
    Google 将这一现象总结为 Internal Co-variate Shift，简称 ICS.

违背机器学习的ddl假设 (输入域 输出空间 分布相同）
https://blog.csdn.net/zqnnn/article/details/88106591
"""
import torch
import numpy as np
import torch.nn as nn

import sys, os
hello_pytorch_DIR = os.path.abspath(os.path.dirname(__file__)+os.path.sep+".."+os.path.sep+"..")
sys.path.append(hello_pytorch_DIR)

from tools.common_tools import set_seed

set_seed(1)  # 设置随机种子


class MLP(nn.Module):
    def __init__(self, neural_num, layers=100):
        super(MLP, self).__init__()
        self.linears = nn.ModuleList([nn.Linear(neural_num, neural_num, bias=False) for i in range(layers)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(neural_num) for i in range(layers)])
        self.neural_num = neural_num

    def forward(self, x):

        for (i, linear), bn in zip(enumerate(self.linears), self.bns):
            x = linear(x)
            x = bn(x)
            x = torch.relu(x)

            if torch.isnan(x.std()):
                print("output is nan in {} layers".format(i))
                break

            print("layers:{}, std:{}".format(i, x.std().item()))

        return x

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):

                # method 1
                # nn.init.normal_(m.weight.data, std=1)    # normal: mean=0, std=1

                # method 2 kaiming
                nn.init.kaiming_normal_(m.weight.data)


neural_nums = 256
layer_nums = 100
batch_size = 16

net = MLP(neural_nums, layer_nums)
# net.initialize()

inputs = torch.randn((batch_size, neural_nums))  # normal: mean=0, std=1

output = net(inputs)
print(output)
