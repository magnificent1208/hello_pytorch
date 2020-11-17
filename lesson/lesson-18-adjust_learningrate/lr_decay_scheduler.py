# -*- coding:utf-8 -*-
"""
@brief      : 学习率下降策略
gradient descent
w i+1 = w i - LR*g(w i)
学习率控制更新的步伐

torch调整学习率的基类
class_LRScheduler
optimizer 关联的优化器
last_epoch 记录epoch数
base_lrs 记录初始学习率

#####总结
有序调整
指标自适应调整
自定义调整
"""
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
torch.manual_seed(1)
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

LR = 0.1
iteration = 10
max_epoch = 200
# ------------------------------ fake data and optimizer  ------------------------------

weights = torch.randn((1), requires_grad=True)
target = torch.zeros((1))

optimizer = optim.SGD([weights], lr=LR, momentum=0.9)

# ------------------------------ 1 Step LR ------------------------------
'''
等间隔调整学习率
主要参数：
step_size 调整间隔数
gamma 调整系数
lr = lr*gamma
'''

flag = 0
# flag = 1
if flag:

    scheduler_lr = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)  # 设置学习率下降策略
    #step_size 就是每50epoch更新一次
    lr_list, epoch_list = list(), list()
    for epoch in range(max_epoch):

        # 获取当前lr，新版本用 get_last_lr()函数，旧版本用get_lr()函数，具体看UserWarning
        lr_list.append(scheduler_lr.get_last_lr())
        epoch_list.append(epoch)

        for i in range(iteration):

            loss = torch.pow((weights - target), 2)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

        scheduler_lr.step()

    plt.plot(epoch_list, lr_list, label="Step LR Scheduler")
    plt.xlabel("Epoch")
    plt.ylabel("Learning rate")
    plt.legend()
    plt.show()


# ------------------------------ 2 Multi Step LR ------------------------------
'''
设置里程碑 制定iter数字进行 decay
'''
flag = 0
# flag = 1
if flag:

    milestones = [50, 125, 160]
    scheduler_lr = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    lr_list, epoch_list = list(), list()
    for epoch in range(max_epoch):

        lr_list.append(scheduler_lr.get_lr())
        epoch_list.append(epoch)

        for i in range(iteration):

            loss = torch.pow((weights - target), 2)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

        scheduler_lr.step()

    plt.plot(epoch_list, lr_list, label="Multi Step LR Scheduler\nmilestones:{}".format(milestones))
    plt.xlabel("Epoch")
    plt.ylabel("Learning rate")
    plt.legend()
    plt.show()


# ------------------------------ 3 Exponential LR ------------------------------
'''
lr = lr * gamma ** epoch
'''

flag = 0
# flag = 1
if flag:

    gamma = 0.95
    scheduler_lr = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    lr_list, epoch_list = list(), list()
    for epoch in range(max_epoch):

        lr_list.append(scheduler_lr.get_lr())
        epoch_list.append(epoch)

        for i in range(iteration):

            loss = torch.pow((weights - target), 2)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

        scheduler_lr.step()

    plt.plot(epoch_list, lr_list, label="Exponential LR Scheduler\ngamma:{}".format(gamma))
    plt.xlabel("Epoch")
    plt.ylabel("Learning rate")
    plt.legend()
    plt.show()

# ------------------------------ 4 Cosine Annealing LR ------------------------------
'''
设置下降周期T max
'''
flag = 0
# flag = 1
if flag:

    t_max = 50
    scheduler_lr = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=0.)

    lr_list, epoch_list = list(), list()
    for epoch in range(max_epoch):

        lr_list.append(scheduler_lr.get_lr())
        epoch_list.append(epoch)

        for i in range(iteration):

            loss = torch.pow((weights - target), 2)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

        scheduler_lr.step()

    plt.plot(epoch_list, lr_list, label="CosineAnnealingLR Scheduler\nT_max:{}".format(t_max))
    plt.xlabel("Epoch")
    plt.ylabel("Learning rate")
    plt.legend()
    plt.show()


# ------------------------------ 5 Reduce LR On Plateau ------------------------------
'''
可以监控某个指标，当指标不再变化，则调整学习率  [实用]
参数：
    mode(str)- 模式选择，有 min 和 max 两种模式：
            min 表示当指标不再降低(如监测loss)， 
            max 表示当指标不再升高(如监测 accuracy)。
    factor(float)- 学习率调整倍数(等同于其它方法的 gamma)，即学习率更新为 lr = lr * factor
    patience(int)- 忍受该指标多少个 step 不变化，当忍无可忍时，调整学习率。
    verbose(bool)- 是否打印学习率信息， print(‘Epoch {:5d}: reducing learning rate of group {} to {:.4e}.’.format(epoch, i, new_lr))
    threshold_mode(str)- 选择判断指标是否达最优的模式，有两种模式， rel 和 abs。
            当 threshold_mode == rel，并且 mode == max 时， dynamic_threshold = best * ( 1 +threshold )；
            当 threshold_mode == rel，并且 mode == min 时， dynamic_threshold = best * ( 1 -threshold )；
            当 threshold_mode == abs，并且 mode== max 时， dynamic_threshold = best + threshold ；
            当 threshold_mode == rel，并且 mode == max 时， dynamic_threshold = best - threshold；
    threshold(float)- 配合 threshold_mode 使用。
    cooldown(int)- “冷却时间“，当调整学习率之后，让学习率调整策略冷静一下，让模型再训练一段时间，再重启监测模式。
    min_lr(float or list)- 学习率下限，可为 float，或者 list，当有多个参数组时，可用 list 进行设置。
    eps(float)- 学习率衰减的最小值，当学习率变化小于 eps 时，则不调整学习率。
'''

flag = 0
# flag = 1
if flag:
    loss_value = 0.5
    accuray = 0.9

    factor = 0.1
    mode = "min"
    patience = 10
    cooldown = 10
    min_lr = 1e-4
    verbose = True

    scheduler_lr = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=factor, mode=mode, patience=patience,
                                                        cooldown=cooldown, min_lr=min_lr, verbose=verbose)

    for epoch in range(max_epoch):
        for i in range(iteration):

            # train(...)

            optimizer.step()
            optimizer.zero_grad()

        if epoch == 5:
            loss_value = 0.4

        scheduler_lr.step(loss_value) #监控 loss value


# ------------------------------ 6 lambda ------------------------------
'''
自定义调整学习率
'''
# flag = 0
flag = 1
if flag:

    lr_init = 0.1

    weights_1 = torch.randn((6, 3, 5, 5))
    weights_2 = torch.ones((5, 5))
    #构建优化器
    optimizer = optim.SGD([
        {'params': [weights_1]},
        {'params': [weights_2]}], lr=lr_init)
    #构建lambda
    lambda1 = lambda epoch: 0.1 ** (epoch // 20)
    lambda2 = lambda epoch: 0.95 ** epoch

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda1, lambda2])

    lr_list, epoch_list = list(), list()
    for epoch in range(max_epoch):
        for i in range(iteration):

            # train(...)

            optimizer.step()
            optimizer.zero_grad()

        scheduler.step()

        lr_list.append(scheduler.get_lr())
        epoch_list.append(epoch)

        print('epoch:{:5d}, lr:{}'.format(epoch, scheduler.get_lr()))

    plt.plot(epoch_list, [i[0] for i in lr_list], label="lambda 1")
    plt.plot(epoch_list, [i[1] for i in lr_list], label="lambda 2")
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("LambdaLR")
    plt.legend()
    plt.show()
