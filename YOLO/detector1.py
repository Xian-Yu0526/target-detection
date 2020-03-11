from darknet53 import *
from cfg import *
from _data import *
import PIL.ImageDraw as draw
import torch.nn as nn
from ways import *
from PIL import ImageFont
import time

class Detector(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = MainNet()
        self.net.eval()
        try:
            self.net.load_state_dict(torch.load(r"./module/p.pt"))
            print("加载成功")
        except:
            print("加载失败")

    def _filter(self, output, thresh):
        output = output.permute(0, 2, 3, 1)
        output = output.reshape(output.size(0), output.size(1), output.size(2), 3, -1)

        mask = output[..., 0] > thresh
        idxs = torch.nonzero(mask)
        vecs = output[mask]
        return idxs, vecs

    def _parse(self, idxs ,vecs, t, anchors):
        anchors = torch.Tensor(anchors)

        n = idxs[:, 0]
        a = idxs[:, 3]
        pre_x = (idxs[:, 2].float()+vecs[:, 1])*t
        pre_y = (idxs[:, 1].float()+vecs[:, 2])*t
        iou = vecs[:, 0]
        kind = torch.argmax(vecs[:, 5:15], dim=1).float()

        pre_w = anchors[a, 0]*torch.exp(vecs[:, 3])
        pre_h = anchors[a, 1]*torch.exp(vecs[:, 4])
        pre_x1 = pre_x-0.5*pre_w
        pre_y1 = pre_y-0.5*pre_h
        pre_x2 = pre_x+0.5*pre_w
        pre_y2 = pre_y+0.5*pre_h

        return torch.stack([iou, pre_x1, pre_y1, pre_x2, pre_y2, kind], dim=1)

    def forward(self, x, thresh, anchors):
        output13, output26, output52 = self.net(x)

        idxs_13, vecs_13 = self._filter(output13, thresh)

        if idxs_13.shape[0] == 0:
            boxes_13 = []
        else:
            boxes_13 = self._parse(idxs_13, vecs_13, 32, anchors[13])

        idxs_26, vecs_26 = self._filter(output26, thresh)

        if idxs_26.shape[0] == 0:
            boxes_26 = []
        else:
            boxes_26 = self._parse(idxs_26, vecs_26, 16, anchors[26])

        idxs_52, vecs_52 = self._filter(output52, thresh)

        if idxs_52.shape[0] ==0:
            boxes_52 = []
        else:
            boxes_52 = self._parse(idxs_52, vecs_52, 8, anchors[52])
        return torch.stack([*boxes_13, *boxes_26, *boxes_52])


if __name__ == '__main__':
    detector = Detector()
    img_path = r".\data\imgs\01.jpg"
    color = ["cyan", "red", "orange", "green","yellow", "white","purple" ,"black","gray", "brown"]
    strs = ["else","Filling machine", "fighter plane", "fighter", "corvette", "aircraft carrier", "awacs", "shipboard aircraft","fighter", "else"]
    font = ImageFont.truetype("simkai", 10)
    with Image.open(img_path) as im:
          input = transforms(im).unsqueeze(0)
          start_time=time.time()
          boxes = detector(input, 0.5, ANCHORS_GROUP)
          end_time=time.time()
          print("all_time={}".format(end_time-start_time))
          _boxes = []
          for i in range(10):
              o_boxes = (boxes[boxes[:, 5] == i*1.0]).detach().numpy()
              r_boxes = NMS(o_boxes, 0.5)
              _boxes.append(r_boxes)
          img_draw = draw.ImageDraw(im)
          for i in range(10):
              if len(_boxes[i]) > 0:
                  for box in _boxes[i]:

                    x1 = int(box[1])
                    y1 = int(box[2])
                    x2 = int(box[3])
                    y2 = int(box[4])
                    img_draw.rectangle((x1, y1, x2, y2), outline=color[i])
                    img_draw.text((x1+5, y1+5), strs[i], (255, 0, 0), font=font)
          im.show()
          # im.save("./result/{0}.png".format(9))
