'''
Pytorch的动态图机制
torch.autograd.backward(tensors,
                        grad_tensors=None, 用于求导的张量
                        retain_graph=None, 保存计算图
                        create_graph=False) 创建倒数计算图，用于高阶求导

'''

import torch
torch.manual_seed(10)

# =========retain_graph=========
#flag = True
flag = False
if flag:
    w = torch.tensor([1.],requires_grad=True)
    x = torch.tensor([2.],requires_grad=True)

    a = torch.add(w,x)
    b = torch.add(w,1)
    y = torch.mul(a,b)

    y.backward(retain_graph=True)
    y.backward() #prove 如果不保存 就不能进行两次 反向传播，会被释放掉
    print(w.grad)

# =========grad_tensor=========
#flag = True
flag = False
if flag:
    w = torch.tensor([1.],requires_grad=True)
    x = torch.tensor([2.],requires_grad=True)

    a = torch.add(w,x)
    b = torch.add(w,1)

    y0 = torch.mul(a,b) #y0 = (x+w)*(w+1)
    y1 = torch.add(a,b) #y1 = (x+w）*(w+1)   dy1/dw =2

    loss = torch.cat([y0,y1],dim=0)  #[y0,y1]
    grad_tensors = torch.tensor([1.,1.]) #用于多个梯度之间，权重设置。类似于backward方法中的grad_tensors

    loss.backward(gradient=grad_tensors) #在backward中的gradient
    print(w.grad)

# ========= autograd.grad=========
#求取梯度   example 使用autogradgrad方法实现2阶倒数

#flag = True
flag = False
if flag:
    x = torch.tensor([3.], requires_grad=True)
    y = torch.pow(x,2) #y=x**2

    grad_1 = torch.autograd.grad(y,x,create_graph=True) #grad_1 =dy/dx =2x =2*3 =6
    print(grad_1)
    grad_2 = torch.autograd.grad(grad_1[0], x)  #grad_2 = d(dy/dx) = d(2x)/dx = 2
    print(grad_2)

'''
tips#1 需要梯度清零，不然就会变成叠加
tips#2 依赖于叶子节点的节点，require_grad自动是true
tips#3 叶子节点不可以用inplace操作 原位操作
'''
#====tips example====

flag = True
#flag = False
if flag:
    w = torch.tensor([1.],requires_grad=True)
    x = torch.tensor([2.],requires_grad=True)

    a = torch.add(w,x)
    b = torch.add(w,1)
    y = torch.mul(a,b)

    w.add(1)


    y.backward()
    print(w.grad)