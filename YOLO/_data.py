import torch
from torch.utils.data import Dataset
import torchvision
import numpy as np
from cfg import *
import os

from PIL import Image
import math

label_file_path = r".\data\label.txt"
Image_file_path = r".\data"
# label_file_path = r"train1/lable.txt"
# Image_file_path = r"train1/img"


transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])


def one_hot(cls_num, index):
    b = np.zeros(cls_num)
    b[index] = 1
    return b


class Mydataset(Dataset):
    def __init__(self):
        with open(label_file_path) as f:
            self.dataset = f.readlines()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        label = {}
        line = self.dataset[index]
        strs = line.split()
        _img_data = Image.open(os.path.join(Image_file_path, strs[0]))
        img_data = transforms(_img_data).float()
        _boxes = np.array([float(x) for x in strs[1:]])
        boxes = np.split(_boxes, len(_boxes)//5)
        print(boxes)
        for feature_size, anchors in ANCHORS_GROUP.items():
            label[feature_size] = np.zeros(shape=(feature_size, feature_size, 3, 5 + CLASS_NUM))

            for box in boxes:
                cls, cen_x, cen_y, w, h = box
                cx_offset, cx_index = math.modf(cen_x*feature_size/IMG_WIDTH)
                cy_offset, cy_index = math.modf(cen_y * feature_size / IMG_HEIGHT)

                for i, anchor in enumerate(anchors):
                    anchor_area = ANCHORS_GROUP_AREA[feature_size][i]
                    p_w, p_h = w/anchor[0], h/anchor[1]
                    area = w*h
                    iou = min(area, anchor_area)/max(area, anchor_area)
                    # print(np.log(p_h))
                    label[feature_size][int(cy_index), int(cx_index), i] = np.array(
                        [iou, cx_offset, cy_offset, np.log(p_w), np.log(p_h), *one_hot(CLASS_NUM, int(cls))])

        return torch.Tensor(label[13]), torch.Tensor(label[26]), torch.Tensor(label[52]), img_data

myDataset = Mydataset()
train_loader = torch.utils.data.DataLoader(myDataset, batch_size=3, shuffle=True)
for target_13, target_26, target_52, img_data in train_loader:
    print(target_13.size())
#     print(target_26.size())
#     print(target_52.size())

