import torch
import numpy as np
import workNet
from PIL import Image,ImageDraw
import ways
from torchvision import transforms
import time

class Detector():
    def __init__(self,P_net_param="./module/P_module.pt",
                 R_net_param="./module/R_module.pt",
                 O_net_param="./module/O_module1.pt",isCuda=True):
        self.isCuda=isCuda
        self.P_net=workNet.P_net()
        self.R_net=workNet.R_net()
        self.O_net=workNet.O_net()

        if self.isCuda:
            self.P_net.cuda()
            self.R_net.cuda()
            self.O_net.cuda()

        self.P_net.load_state_dict(torch.load(P_net_param))
        self.R_net.load_state_dict(torch.load(R_net_param))
        self.O_net.load_state_dict(torch.load(O_net_param))

        self.P_net.eval()
        self.R_net.eval()
        self.O_net.eval()

        self._img_transfor=transforms.Compose([transforms.ToTensor()])

    def detect(self,image):
        start_time=time.time()
        Pnet_boxes=self._P_net_detect(image)

        if Pnet_boxes.shape[0]==0:
            return np.array([])
        end_time=time.time()
        P_times=end_time-start_time

        start_time=time.time()
        Rnet_boxes=self._Rnet_detect(image,Pnet_boxes)
        if Rnet_boxes.shape[0]==0:
            return np.array([])
        end_time=time.time()
        R_times=end_time-start_time

        start_time = time.time()
        Onet_boxes=self.O_net_detect(image,Rnet_boxes)
        if Onet_boxes.shape[0]==0:
            return np.array([])
        end_time=time.time()
        O_times=end_time-start_time
        times=P_times+R_times+O_times
        print("total{0},P_times{1},R_times{2},O_times{3}".format(times,P_times,R_times,O_times))
        return Onet_boxes

    def _P_net_detect(self,image):
        boxes=[]

        w,h=image.size
        min_side_len=min(w,h)
        scale=1

        while min_side_len > 12:

            img_data = self._img_transfor(image)

            if self.isCuda:
                img_data = img_data.cuda()
            img_data.unsqueeze_(0)  # 升维度，可省略

            _offest, _cls = self.P_net(img_data)

            cls, offest = _cls[0][0].cpu(), _offest[0].cpu()
            # print(offest.size())
            print(cls.shape)
            idxs = torch.nonzero(torch.gt(cls, 0.85))
            print(idxs.shape)
            for idx in idxs:
                boxes.append(self._box(idx, offest, cls[idx[0], idx[1]], scale))

            scale *= 0.9
            _w = int(w * scale)
            _h = int(h * scale)

            image = image.resize((_w, _h))
            min_side_len = min(_w, _h)

        return ways.NMS(np.stack(boxes), 0.5, FT=False)

    # 反算回归到原图
    def _box(self,start_index,offset,cls,scale,stride=2,side_len=12):
        offset=offset.detach().numpy()
        cls=cls.detach().numpy()

        _x1 = (start_index[1] * stride).float() / scale
        _y1 = (start_index[0] * stride).float() / scale
        _x2 = (start_index[1] * stride + side_len).float()/ scale
        _y2 = (start_index[0] * stride + side_len).float() / scale

        ow = _x2 - _x1
        oh = _y2 - _y1

        _offset = offset[:, start_index[0], start_index[1]]
        x1 = _x1 + ow * _offset[0]
        y1 = _y1 + oh * _offset[1]
        x2 = _x2 + ow * _offset[2]
        y2 = _y2 + oh * _offset[3]

        return [x1, y1, x2, y2, cls]

    def _Rnet_detect(self,image,Pnet_boxes):

        img_dataset=[]

        _Pnet_boxes=ways.re_size(Pnet_boxes)
        for _box1 in _Pnet_boxes:
            _x1=int(_box1[0])
            _y1=int(_box1[1])
            _x2=int(_box1[2])
            _y2=int(_box1[3])

            img=image.crop((_x1,_y1,_x2,_y2))
            img=img.resize((24,24))

            # print(img.size)
            img_data=self._img_transfor(img)
            img_dataset.append((img_data))

        img_datasets=torch.stack(img_dataset)

        if self.isCuda:
            img_datasets=img_datasets.cuda()

        _offset,_cls=self.R_net(img_datasets)
        cls, offset =_cls.cpu().detach().numpy(),_offset.cpu().detach().numpy()

        boxes=[]
        idxs, _ = np.where(cls > 0.8)
        for idx in idxs:
            _box = _Pnet_boxes[idx]
            _x1 = int(_box[0])
            _y1 = int(_box[1])
            _x2 = int(_box[2])
            _y2 = int(_box[3])

            ow = _x2 - _x1
            oh = _y2 - _y1

            x1 = _x1 + ow * offset[idx][0]
            y1 = _y1 + oh * offset[idx][1]
            x2 = _x2 + ow * offset[idx][2]
            y2 = _y2 + oh * offset[idx][3]

            boxes.append([x1, y1, x2, y2, cls[idx][0]])
        return ways.NMS(np.stack(boxes),0.3)

    def O_net_detect(self,image,Rnet_boxes):
        img_dataset = []
        _Rnet_boxes = ways.re_size(Rnet_boxes)

        for _box in _Rnet_boxes:
            _x1 = int(_box[0])
            _y1 = int(_box[1])
            _x2 = int(_box[2])
            _y2 = int(_box[3])

            img = image.crop((_x1, _y1, _x2, _y2))
            img = img.resize((48, 48))
            img_data = self._img_transfor(img)
            img_dataset.append((img_data))

            img_datasets = torch.stack(img_dataset)
            if self.isCuda:
                img_datasets = img_datasets.cuda()

            _offset, _cls = self.O_net(img_datasets)
            cls, offset = _cls.cpu().detach().numpy(), _offset.cpu().detach().numpy()

            boxes = []
            idxs, _ = np.where(cls > 0.99)
            for idx in idxs:
                _box = _Rnet_boxes[idx]
                _x1 = int(_box[0])
                _y1 = int(_box[1])
                _x2 = int(_box[2])
                _y2 = int(_box[3])

                ow = _x2 - _x1
                oh = _y2 - _y1

                x1 = _x1 + ow * offset[idx][0]
                y1 = _y1 + oh * offset[idx][1]
                x2 = _x2 + ow * offset[idx][2]
                y2 = _y2 + oh * offset[idx][3]

                boxes.append([x1, y1, x2, y2, cls[idx][0]])
        return ways.NMS(np.array(boxes),0.3,FT=True)

if __name__ == '__main__':
    image_file=r"G:\python\实验\mtcnn\test3.jpg"
    detect=Detector()
    with Image.open(image_file) as img:
        boxes=detect.detect(img)
        # print(boxes)
        # print(img.size)
        imgDraw=ImageDraw.Draw(img)

        for box in boxes:
            x1 = int(box[0])
            y1 = int(box[1])
            x2 = int(box[2])
            y2 = int(box[3])

            imgDraw.rectangle((x1,y1,x2,y2),outline='red')
        img.show()