# -*- coding:utf-8 -*-
"""
@brief      : 测试tensorboard可正常使用
"""
'''
Tensorboard
1. python脚本记录可视化数据
2. 硬盘中存储 event file
3. 终端的tensorboard读取硬盘类型&可视化
'''

import numpy as np
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(comment='test_tensorboard')


for x in range(100):
    writer.add_scalar('y=2x', x * 2, x)
    writer.add_scalar('y=pow(2, x)', 2 ** x, x)

    writer.add_scalars('data/scalar_group', {"xsinx": x * np.sin(x),
                                             "xcosx": x * np.cos(x),
                                             "arctanx": np.arctan(x)}, x)
writer.close()