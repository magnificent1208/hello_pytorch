先运行model_save.py，再运行model_load.py
先运行save_checkpoint.py，再运行checkpoint_resume.py

---------------Note---------------
内存------>硬盘------->内存
   (序列化)    (反序列化)

- 序列化与反序列换
序列化-把对象转换为字节序列，就是加载模型
反序列化-从内存中保存
模型的保存与加载
###############
torch.save()
    params:
    obj：对象
    f：输出路径
torch.load()
    params:
        f：文件路径
        map_location:制定存放的位置，cpu/gpu
###############
choice1 保存整个Module #耗时占内存
torch.save(net,path)

choice2 保存模型参数
state_dict = net.state_dict()
torch.save(state_dict,path)