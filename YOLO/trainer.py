import _data
import dataset
from net import MainNet
import os
import torch
import torch.nn as nn

def loss_fn(output, target, alpha):
    output = output.permute(0, 2, 3, 1)#.double()
    # print(output.shape)
    output = output.reshape(output.size(0), output.size(1), output.size(2), 3, -1)
    # print(output.shape)

    mask_obj = target[..., 0] > 0
    mask_noobj = target[..., 0] == 0

    loss_obj = torch.mean((output[mask_obj] - target[mask_obj]) ** 2)
    loss_noobj = torch.mean((output[mask_noobj] - target[mask_noobj]) ** 2)

    loss = alpha * loss_obj + (1 - alpha) * loss_noobj
    return loss


if __name__ == '__main__':
    # myDataset = dataset.MyDataset()
    myDataset = _data.Mydataset()
    train_loader = torch.utils.data.DataLoader(myDataset, batch_size=4, shuffle=True)

    net = MainNet().cuda()

    if os.path.exists(r'./module/p2.pt'):
        try:
            net.load_state_dict(torch.load(r'./module/p2.pt'))
            print("加载成功")
        except:
            print("加载不成功！")

    opt = torch.optim.Adam(net.parameters())
    count=0
    while True:
        for target_13, target_26, target_52, img_data in train_loader:
            output_13, output_26, output_52 = net(img_data.cuda())

            loss_13 = loss_fn(output_13, target_13.cuda(), 0.9)
            loss_26 = loss_fn(output_26, target_26.cuda(), 0.9)
            loss_52 = loss_fn(output_52, target_52.cuda(), 0.9)

            loss = loss_13 + loss_26 + loss_52

            opt.zero_grad()
            loss.backward()
            opt.step()
            print(loss.item())
        count+=1
        if count%20==0:
            print(count,loss.item())
            torch.save(net.state_dict(),r"./module/p2.pt")
            print("保存成功！")