import os
from data_load import Face_Dataset
import torch.utils.data as data
import torch
import torch.nn as nn
class Trainer():
    def __init__(self,net,save_path,data_path,isCuda=True):
        self.net=net
        self.save_path=save_path
        self.data_path=data_path
        self.isCuda=isCuda

        if self.isCuda:
            self.net.cuda()

        self.c_loss_fn=nn.BCELoss().cuda()
        self.off_loss_fn=nn.MSELoss().cuda()

        self.optimizerer=torch.optim.Adam(self.net.parameters(),lr=0.0001)#

        if os.path.exists(self.save_path):
            net.load_state_dict(torch.load(self.save_path))
    def train(self):
        dataset = Face_Dataset(self.data_path)
        data1 = data.DataLoader(dataset=dataset, batch_size=512, shuffle=True,num_workers=4)
        losses1=0.001
        while True:
            for i, (img, c, off) in enumerate(data1):
                if self.isCuda:
                    img=img.cuda()
                    c=c.cuda()
                    off=off.cuda()
                    out_category,out_offset=self.net(img)

                out_category=out_category.reshape(-1,1)
                out_offset=out_offset.reshape(out_offset.size(0),-1)
                category_mask=torch.lt(c,2)

                c=torch.masked_select(c,category_mask)

                out_category=torch.masked_select(out_category,category_mask)
                c_loss=self.c_loss_fn(out_category,c)

                off_mask=torch.gt(off,0)
                off=off[off_mask]
                out_offset=out_offset[off_mask]

                off_loss=self.off_loss_fn(out_offset,off)
                losses=c_loss+off_loss

                self.optimizerer.zero_grad()
                losses.backward()
                self.optimizerer.step()

                # if i%100==0:
                #     print("{}损失为{}".format(i, losses.item()))
                #     # if losses.item()<losses1:
                #     #     losses1 = losses.item()
                #     torch.save(self.net.state_dict(),self.save_path)
                #     print("保存成功")
            print("损失为{}".format(losses.item()))
            torch.save(self.net.state_dict(), self.save_path)
            print("保存成功")