
for i,(images,target) in enumerate(train_loader):
    # 1.input output
    images = images.cuda(non_blocking=True)
    target = torch.from_numpy(np.array(target)).float().cuda(non_blocking=True)

    # 2.mixup
    alpha=config.alpha
    lam = np.random.beta(alpha,alpha)
    index = torch.randperm(images.size(0)).cuda()
    inputs = lam*images + (1-lam)*images[index,:]
    targets_a, targets_b = target, target[index]
    outputs = model(inputs)
    loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)

    # 3.backward
    optimizer.zero_grad()   # reset gradient
    loss.backward()
    optimizer.step()        # update parameters of net