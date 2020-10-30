# -*- coding: utf-8 -*-
"""
# @brief      : 其它网络层
pool linear activation

###pooling method
nn.MaxPool2d(
    ***略
    return_indices =  记录池化层索引  @后面可以通过这个索引，做unpooling
    )
nn.AvgPool2d(
    ***
    divisor_override =  设置除法因子，可以把分母设置为自己想要的，而不一定是像素的个数
    ）
"""
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
import random
import numpy as np
import torchvision
import torch.nn as nn
from torchvision import transforms
from matplotlib import pyplot as plt
from PIL import Image

#Tips： 跨层级导入模块~ 记住这个格式
path_tools = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "tools", "common_tools.py"))
assert os.path.exists(path_tools), "{}不存在，请将common_tools.py文件放到 {}".format(path_tools, os.path.dirname(path_tools))

import sys
hello_pytorch_DIR = os.path.abspath(os.path.dirname(__file__)+os.path.sep+".."+os.path.sep+"..")
sys.path.append(hello_pytorch_DIR)

from tools.common_tools import set_seed
from tools.common_tools import transform_invert

set_seed(1)  # 设置随机种子

# ================================= load img ==================================
path_img = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test.jpg")
img = Image.open(path_img).convert('RGB')  # 0~255

# convert to tensor
img_transform = transforms.Compose([transforms.ToTensor()])
img_tensor = img_transform(img)
img_tensor.unsqueeze_(dim=0)    # C*H*W to B*C*H*W

# ================================= create convolution layer ==================================

# ================ maxpool
# flag = 1
flag = 0
if flag:
    maxpool_layer = nn.MaxPool2d((2, 2), stride=(2, 2))   # input:(i, o, size) weights:(o, i , h, w)
    img_pool = maxpool_layer(img_tensor)

# ================ avgpool
# flag = 1
flag = 0
if flag:
    avgpoollayer = nn.AvgPool2d((2, 2), stride=(2, 2))   # input:(i, o, size) weights:(o, i , h, w)
    img_pool = avgpoollayer(img_tensor)

# ================ avgpool divisor_override
# flag = 1
flag = 0
if flag:
    img_tensor = torch.ones((1, 1, 4, 4))
    avgpool_layer = nn.AvgPool2d((2, 2), stride=(2, 2), divisor_override=3)
    img_pool = avgpool_layer(img_tensor)

    print("raw_img:\n{}\npooling_img:\n{}".format(img_tensor, img_pool))


# ================ max unpool  #没有vis
# flag = 1
flag = 0
if flag:
    # pooling
    img_tensor = torch.randint(high=5, size=(1, 1, 4, 4), dtype=torch.float)
    maxpool_layer = nn.MaxPool2d((2, 2), stride=(2, 2), return_indices=True)
    img_pool, indices = maxpool_layer(img_tensor)

    # unpooling
    img_reconstruct = torch.randn_like(img_pool, dtype=torch.float)
    maxunpool_layer = nn.MaxUnpool2d((2, 2), stride=(2, 2))
    img_unpool = maxunpool_layer(img_reconstruct, indices)

    print("raw_img:\n{}\nimg_pool:\n{}".format(img_tensor, img_pool))
    print("img_reconstruct:\n{}\nimg_unpool:\n{}".format(img_reconstruct, img_unpool))


# ================ linear  #没有vis
'''
Linear 又称 full conv 全连接层
'''
flag = 1
# flag = 0
if flag:
    inputs = torch.tensor([[1., 2, 3]])
    linear_layer = nn.Linear(3, 4)
    linear_layer.weight.data = torch.tensor([[1., 1., 1.],
                                             [2., 2., 2.],
                                             [3., 3., 3.],
                                             [4., 4., 4.]])

    linear_layer.bias.data.fill_(0.5)
    output = linear_layer(inputs)
    print(inputs, inputs.shape)
    print(linear_layer.weight.data, linear_layer.weight.data.shape)
    print(output, output.shape)



# ================================= visualization ==================================
# print("池化前尺寸:{}\n池化后尺寸:{}".format(img_tensor.shape, img_pool.shape))
# img_pool = transform_invert(img_pool[0, 0:3, ...], img_transform)
# img_raw = transform_invert(img_tensor.squeeze(), img_transform)
# plt.subplot(122).imshow(img_pool)
# plt.subplot(121).imshow(img_raw)
# plt.show()
