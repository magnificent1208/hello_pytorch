# -*- coding: utf-8 -*-
"""
# @brief      : inference demo
"""

'''
模型如何完成图像分类？

3d tensor -> [模型f(X)] -> 向量 -> [人类] -> 实际意义

模型f(x)
nn.Module



1. 类别名与标签的转换 label_name = {"ants":0, "bees":1}
2. 去除向量最大值的标号 _, pred_int = torch.max(outputs.data, 1)
3. 复杂运算 outputs = resnet18(img_tensor)

将数据映射到特征的过程

图像分类过程：
1.获取数据与标签
2.选择模型，损失函数，优化器
3.训练代码
4.推理代码 *本段代码重点关注

##推理代码基本步骤：
1. 获取数据与模型
2. 数据变换,如RGB -> 4D-Tensor
3. 前向传播 4D-tensor映射到向量维度
4. 输出保存预测结果

### 推理阶段注意事项：
1. 确保model处于eval状态而非training ##BN和Dropout在不同状态执行任务是不同的
2. s设置 torch.no_grad(),减少内存消耗
3. 数据预处理需保持一致，RGB or rBGR 通道顺序/均值方差
'''


import os
import time
import torch.nn as nn
import torch
import torchvision.transforms as transforms
from PIL import Image
from matplotlib import pyplot as plt
import torchvision.models as models
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# config *需要与训练的时候valid的transform一致，不然会带来精度大幅下降。
vis = True #是否可视化开关
# vis = False
vis_row = 4

norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]

inference_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std),
])

classes = ["ants", "bees"]


def img_transform(img_rgb, transform=None):
    """
    将数据转换为模型读取的形式
    :param img_rgb: PIL Image
    :param transform: torchvision.transform
    :return: tensor
    """

    if transform is None:
        raise ValueError("找不到transform！必须有transform对img进行处理")

    img_t = transform(img_rgb)
    return img_t


def get_img_name(img_dir, format="jpg"):
    """
    获取文件夹下format格式的文件名
    :param img_dir: str
    :param format: str
    :return: list
    """
    file_names = os.listdir(img_dir)
    img_names = list(filter(lambda x: x.endswith(format), file_names))

    if len(img_names) < 1:
        raise ValueError("{}下找不到{}格式数据".format(img_dir, format))
    return img_names


def get_model(m_path, vis_model=False):

    resnet18 = models.resnet18()
    num_ftrs = resnet18.fc.in_features
    resnet18.fc = nn.Linear(num_ftrs, 2)

    checkpoint = torch.load(m_path)
    resnet18.load_state_dict(checkpoint['model_state_dict'])

    if vis_model:
        from torchsummary import summary
        summary(resnet18, input_size=(3, 224, 224), device="cpu")

    return resnet18


if __name__ == "__main__":

    BASEDIR = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(BASEDIR, "..", "..", "data", "hymenoptera_data"))
    if not os.path.exists(data_dir):
        raise Exception("\n{} 不存在，请下载 07-02-数据-模型finetune.zip  放到\n{}  下，并解压即可".format(
            data_dir, os.path.dirname(data_dir)))

    img_dir = os.path.join(data_dir, "val", "bees")
    model_path = os.path.abspath(os.path.join(BASEDIR, "..", "..", "data", "resnet_checkpoint_14_epoch.pkl"))
    if not os.path.exists(model_path):
        raise Exception("\n{} 不存在，请下载 08-01-数据-20200724.zip  放到\n{}  下，并解压即可".format(
            model_path, os.path.dirname(model_path)))

    time_total = 0
    img_list, img_pred = list(), list()

    # 1. data
    img_names = get_img_name(img_dir)
    num_img = len(img_names)

    # 2. model
    resnet18 = get_model(model_path, True)
    resnet18.to(device)
    resnet18.eval() #必须要设置模型状态 在eval 而不是训练状态

    with torch.no_grad(): #设定不需要反向传播 减少内存消耗
        for idx, img_name in enumerate(img_names):

            path_img = os.path.join(img_dir, img_name)

            # step 1/4 : path --> img
            img_rgb = Image.open(path_img).convert('RGB')

            # step 2/4 : img --> tensor
            img_tensor = img_transform(img_rgb, inference_transform) #3d
            img_tensor.unsqueeze_(0) #增加维度 4d
            img_tensor = img_tensor.to(device)

            # step 3/4 : tensor --> vector
            time_tic = time.time()
            outputs = resnet18(img_tensor)
            time_toc = time.time()

            # step 4/4 : visualization
            _, pred_int = torch.max(outputs.data, 1)
            pred_str = classes[int(pred_int)]

            if vis:
                img_list.append(img_rgb)
                img_pred.append(pred_str)

                if (idx+1) % (vis_row*vis_row) == 0 or num_img == idx+1:
                    for i in range(len(img_list)):
                        plt.subplot(vis_row, vis_row, i+1).imshow(img_list[i])
                        plt.title("predict:{}".format(img_pred[i]))
                    plt.show()
                    plt.close()
                    img_list, img_pred = list(), list()

            time_s = time_toc-time_tic
            time_total += time_s

            print('{:d}/{:d}: {} {:.3f}s '.format(idx + 1, num_img, img_name, time_s))

    print("\ndevice:{} total time:{:.1f}s mean:{:.3f}s".
          format(device, time_total, time_total/num_img))
    if torch.cuda.is_available():
        print("GPU name:{}".format(torch.cuda.get_device_name())) #查询GPU以及型号

#打印时间会发现，通常第一张图片使用的时间是最久的(这里是接近1s) 原因是模型参数在个gpu&cpu都要初始化

'''
关于ResNet18
Res--残差
Net--网络
18--网络中带权值参数的网络层的总数 (1+2x2+2x2+2x2+2x2+1)

50层+后有bottleneck
'''