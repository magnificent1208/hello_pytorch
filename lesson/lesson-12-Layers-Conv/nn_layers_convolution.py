# -*- coding: utf-8 -*-
"""
# @brief      : 学习不同类型的卷积层及其在pytorch中的实现
Definition
卷积运算：卷积核在输入信号(图像)上滑动，在相应位置上进行乘加运算
卷积核：又称滤波器，可认为是模式,特征
1. dimension of conv：
In general,In general,In general, 一般情况下卷积核在几个维度滑动，就是几维卷积
2. nn.Conv2d(
        **其他略
        groups =   (设置分组卷积。例如AlexNet，分成两个通道提取特征，到最后再进行特征融合；
                    又如，STN的U支路)
"""
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from matplotlib import pyplot as plt
path_tools = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "tools", "common_tools.py"))
assert os.path.exists(path_tools), "{}不存在，请将common_tools.py文件放到 {}".format(path_tools, os.path.dirname(path_tools))

import sys
hello_pytorch_DIR = os.path.abspath(os.path.dirname(__file__)+os.path.sep+".."+os.path.sep+"..")
sys.path.append(hello_pytorch_DIR)

from tools.common_tools import transform_invert, set_seed

set_seed(3)  # 设置随机种子



# ================================= load img ==================================
path_img = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test.jpg")
img = Image.open(path_img).convert('RGB')  # 0~255

# convert to tensor
img_transform = transforms.Compose([transforms.ToTensor()])
img_tensor = img_transform(img)
img_tensor.unsqueeze_(dim=0)    # C*H*W to B*C*H*W

# ================================= create convolution layer ==================================

# ================ 2d
# flag = 1
flag = 0
if flag:
    conv_layer = nn.Conv2d(3, 1, 3)   # input:(i, o, size) weights:(o, i , h, w)
    nn.init.xavier_normal_(conv_layer.weight.data)

    # calculation
    img_conv = conv_layer(img_tensor)

# ================ transposed
flag = 1
# flag = 0
if flag:
    conv_layer = nn.ConvTranspose2d(3, 1, 3, stride=2)   # input:(i, o, size)
    nn.init.xavier_normal_(conv_layer.weight.data)

    # calculation
    img_conv = conv_layer(img_tensor)


# ================================= visualization ==================================
print("卷积前尺寸:{}\n卷积后尺寸:{}".format(img_tensor.shape, img_conv.shape))
img_conv = transform_invert(img_conv[0, 0:1, ...], img_transform)
img_raw = transform_invert(img_tensor.squeeze(), img_transform)
plt.subplot(122).imshow(img_conv, cmap='gray')
plt.subplot(121).imshow(img_raw)
plt.show()
