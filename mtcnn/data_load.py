from torch.utils.data import Dataset
import torch.utils.data as data
import os
import numpy as np
import torch
from PIL import Image

class Face_Dataset(Dataset):

    def __init__(self, path):
        self.path = path
        self.dataset = []
        self.dataset.extend(open(os.path.join(path, "positive.txt")).readlines())
        self.dataset.extend(open(os.path.join(path, "negative.txt")).readlines())
        self.dataset.extend(open(os.path.join(path, "part.txt")).readlines())

    def __getitem__(self, index):
        strs = self.dataset[index].strip().split(" ")

        img_path = os.path.join(self.path, strs[0])
        cond = torch.Tensor([int(strs[1])])
        offset = torch.Tensor([float(strs[2]), float(strs[3]), float(strs[4]), float(strs[5])])# float(strs[6]), float(strs[7]), float(strs[8]), float(strs[9]),float(strs[10]), float(strs[11]), float(strs[12]), float(strs[13]),float(strs[14]),float(strs[15])
        img_data =torch.Tensor(np.array(Image.open(img_path)).transpose([2,0,1])/255-0.5)

        return img_data, cond, offset

    def __len__(self):
        return len(self.dataset)


if __name__ == '__main__':
    dataset=Face_Dataset(r"F:\48")
    data1=data.DataLoader(dataset=dataset,batch_size=5,shuffle=True)
    for i,(img,c,off)in enumerate (data1):
        # print(i)
        print(off.size())
        # print(c)


