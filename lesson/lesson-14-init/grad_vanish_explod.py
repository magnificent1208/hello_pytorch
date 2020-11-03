# -*- coding: utf-8 -*-
"""
# @brief      : 梯度消失与爆炸实验
关于网络权值初始化
前提/标准--
几种权值初始化的讨论与应用
1： 使用正态分布初始化。根据公式推导，每向前传播一层，标准差扩大根号n倍。初始n=神经元个数，之后就是上一层的标准差。【实验1&2】
2: 使用Xavier初始化。(公式推导是从“方差一致性”出发，初始化的分布有均匀分布和正态分布两种)【实验】
            “方差一致性”：保持数据尺度位置在恰当范围，通常方差为1
            “饱和类激活函数”：sigmoid tanh等
            讨论了具有激活函数时应该如何初始化网络。
3. 使用Kaiming初始化。
            同样遵循方差一致性
            "非饱和激活函数"：relu及其变种
---------------------------------------------------------
关于nn.init.calculate_gain()的介绍
计算激活函数的方差变化尺度
Return the recommended gain value for the given nonlinearity function.
are as follows: @https://pytorch.org/docs/stable/nn.init.html


"""
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
import torch
import random
import numpy as np
import torch.nn as nn
path_tools = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "tools", "common_tools.py"))
assert os.path.exists(path_tools), "{}不存在，请将common_tools.py文件放到 {}".format(path_tools, os.path.dirname(path_tools))

import sys
hello_pytorch_DIR = os.path.abspath(os.path.dirname(__file__)+os.path.sep+".."+os.path.sep+"..")
sys.path.append(hello_pytorch_DIR)

from tools.common_tools import set_seed

set_seed(1)  # 设置随机种子


class MLP(nn.Module):
    def __init__(self, neural_num, layers):
        super(MLP, self).__init__()
        self.linears = nn.ModuleList([nn.Linear(neural_num, neural_num, bias=False) for i in range(layers)])
        self.neural_num = neural_num

    def forward(self, x):
        for (i, linear) in enumerate(self.linears):
            x = linear(x)  #每向前传播一层，标准差扩大根号n倍。初始n=神经元格式，之后就是上一层的标准差
            # x = torch.tanh(x) #引出Xavier -- init.normal+直接加饱和激活函数sigmoid或者tanh，会发现梯度慢慢消失
            x = torch.relu(x) #引出Kaiming -- 直接用init.Xavier+非饱和激活函数的话，会发现梯度发生爆炸
            #
            print("layer:{}, std:{}".format(i, x.std()))
            if torch.isnan(x.std()):
                print("output is nan in {} layers".format(i))
                break

        return x

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                #实验1 引发梯度爆炸，出现nan现象 纯linear
                # nn.init.normal_(m.weight.data) #origin权值初始化

                #实验2 保持网络层输出值尺度不变的初始化 采用 mean=0,std=sqrt(1/n) to init weight,观察层的std输出
                # nn.init.normal_(m.weight.data, std=np.sqrt(1/self.neural_num)) #   # normal: mean=0, std=1

                #实验3 手动计算，采用Xavier对权值进行初始化再观察网络层的输出
                # 计算a的数值大小（由于本例中输入和输出神经元个数相同，所以同值）
                # a = np.sqrt(6 / (self.neural_num + self.neural_num))
                # tanh_gain = nn.init.calculate_gain('tanh') #激活函数的增益
                # a *= tanh_gain #利用Pytorch的内置函数calculate_gain计算tanh增益
                # nn.init.uniform_(m.weight.data, -a, a) #利用上、下限对权值进行均匀分布初始化

                #实验4 Pytorch中提供了Xavier，对权值进行初始化，观察与手动计算的输出结果有无区别
                # nn.init.xavier_uniform_(m.weight.data, gain=tanh_gain)

                #实验5 Kaiming初始化 -- 根据公式手动初始化&pytorch内置init.kaiming
                # nn.init.normal_(m.weight.data, std=np.sqrt(2 / self.neural_num))
                nn.init.kaiming_normal_(m.weight.data)

flag = 0
# flag = 1

if flag:
    layer_nums = 100
    neural_nums = 256#256
    batch_size = 16

    net = MLP(neural_nums, layer_nums)
    net.initialize()

    inputs = torch.randn((batch_size, neural_nums))  # normal: mean=0, std=1

    output = net(inputs)
    print(output)

# ======================================= calculate gain =======================================

# flag = 0
flag = 1

if flag:

    x = torch.randn(10000)
    out = torch.tanh(x)

    gain = x.std() / out.std()
    print('gain:{}'.format(gain))

    tanh_gain = nn.init.calculate_gain('tanh')
    print('tanh_gain in PyTorch:', tanh_gain)


