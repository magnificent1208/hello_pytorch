# -*- coding: utf-8 -*-
"""
# @brief      : 梯度消失与爆炸实验
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
            # x = torch.relu(x)
            #
            print("layer:{}, std:{}".format(i, x.std()))
            # if torch.isnan(x.std()):
            #     print("output is nan in {} layers".format(i))
            #     break

        return x

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data)
                # nn.init.normal_(m.weight.data, std=np.sqrt(1/self.neural_num))    # normal: mean=0, std=1

                a = np.sqrt(6 / (self.neural_num + self.neural_num)) #观察均匀分布的上限和下限(-a,a)

                tanh_gain = nn.init.calculate_gain('tanh')
                a *= tanh_gain
                '''
                关于gain 是指数据输入激活函数之后 标准差的变化
                '''

                #
                # nn.init.uniform_(m.weight.data, -a, a)

                nn.init.xavier_uniform_(m.weight.data, gain=tanh_gain)
                '''
                关于Xavier初始化：
                作者Xavier Glorot认为，优秀的初始化应该使得各层的激活值和状态梯度的方差在传播过程中的方差保持一致
                '''

                # nn.init.normal_(m.weight.data, std=np.sqrt(2 / self.neural_num))
                # nn.init.kaiming_normal_(m.weight.data)

# flag = 0
flag = 1

if flag:
    layer_nums = 100
    neural_nums = 400#256
    batch_size = 16

    net = MLP(neural_nums, layer_nums)
    net.initialize()

    inputs = torch.randn((batch_size, neural_nums))  # normal: mean=0, std=1

    output = net(inputs)
    print(output)

# ======================================= calculate gain =======================================

flag = 0
# flag = 1

if flag:

    x = torch.randn(10000)
    out = torch.tanh(x)

    gain = x.std() / out.std()
    print('gain:{}'.format(gain))

    tanh_gain = nn.init.calculate_gain('tanh')
    print('tanh_gain in PyTorch:', tanh_gain)


