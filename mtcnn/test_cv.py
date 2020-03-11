import torch
import numpy as np
import workNet
from PIL import Image,ImageDraw
import ways
import cv2
import time
from torchvision import transforms
class Detector():
    def __init__(self,P_net_param="./module/pnet.pt",
                 R_net_param="./module/rnet.pt",
                 O_net_param="./module/onet.pt",isCuda=True):
        self.isCuda=isCuda
        self.P_net=workNet.Pnet()
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

        self._image_transform = transforms.Compose([
            transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

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
        boxes=np.array([[1,1,1,1,0]])
        w,h=image.size
        min_side_len=min(w,h)
        scale=1
        img_data=self._image_transform(image)
        while min_side_len > 12:

            if self.isCuda:
                img_data = img_data.cuda()
            img_data.unsqueeze_(0)  # 升维度，可省略
            with torch.no_grad():

                _cls,_offest = self.P_net(img_data)

            cls, offest = _cls[0][0].cpu(), _offest[0].cpu()
            idxs = torch.nonzero(torch.gt(cls, 0.46))
            cls = cls[cls>0.46].detach().numpy()
            # cls = cls[cls>0.85]

            idx=np.array(idxs)

            boxes1=self._box(idx,offest,cls,scale)

            scale *= 0.7
            _w = int(w * scale)
            _h = int(h * scale)

            image1 = image.resize((_w, _h))
            min_side_len = min(_w, _h)
            img_data = torch.Tensor(np.array(image1).transpose([2, 0, 1]) / 255 - 0.5)
            boxes=np.vstack((boxes,boxes1))

        return ways.NMS(boxes, 0.7, FT=False)

    # 反算回归到原图
    def _box(self,start_index,offset,cls,scale,stride=2,side_len=12):
        offset=offset.detach().numpy()

        _x1 = (start_index[:,1] * stride) / scale
        _y1 = (start_index[:,0] * stride)/ scale
        _x2 = (start_index[:,1] * stride + side_len)/ scale
        _y2 = (start_index[:,0] * stride + side_len)/ scale

        ow = _x2 - _x1
        oh = _y2 - _y1

        _offset = offset[:, start_index[:,0], start_index[:,1]]
        x1 = _x1 + ow * _offset[0]
        y1 = _y1 + oh * _offset[1]
        x2 = _x2 + ow * _offset[2]
        y2 = _y2 + oh * _offset[3]

        return np.array([x1, y1, x2, y2, cls]).T

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
            img_data=self._image_transform(img)
            img_dataset.append((img_data))

        img_datasets=torch.stack(img_dataset)

        if self.isCuda:
            img_datasets=img_datasets.cuda()
        with torch.no_grad():
            _cls, _offset =self.R_net(img_datasets)
        cls, offset =_cls.cpu().detach().numpy(),_offset.cpu().detach().numpy()

        idxs, _ = np.where(cls >0.6)
        cls=cls[cls>0.6]
        _box = _Pnet_boxes[idxs]
        _x1 = _box[:,0]
        _y1 = _box[:,1]
        _x2 = _box[:,2]
        _y2 = _box[:,3]

        ow = _x2 - _x1
        oh = _y2 - _y1

        x1 = _x1 + ow * offset[idxs][:,0]
        y1 = _y1 + oh * offset[idxs][:,1]
        x2 = _x2 + ow * offset[idxs][:,2]
        y2 = _y2 + oh * offset[idxs][:,3]

        boxes=np.array([x1, y1, x2, y2, cls]).T

        return ways.NMS(boxes,0.3)

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
            img_data = self._image_transform(img)
            img_dataset.append((img_data))

            img_datasets = torch.stack(img_dataset)
            if self.isCuda:
                img_datasets = img_datasets.cuda()
            with torch.no_grad():
                _cls, _offset = self.O_net(img_datasets)
            cls, offset = _cls.cpu().detach().numpy(), _offset.cpu().detach().numpy()

            idxs, _ = np.where(cls > 0.9999)
            cls=cls[cls > 0.9999]
            _box = _Rnet_boxes[idxs]
            _x1 = _box[:, 0]
            _y1 = _box[:, 1]
            _x2 = _box[:, 2]
            _y2 = _box[:, 3]

            ow = _x2 - _x1
            oh = _y2 - _y1

            x1 = _x1 + ow * offset[idxs][:, 0]
            y1 = _y1 + oh * offset[idxs][:, 1]
            x2 = _x2 + ow * offset[idxs][:, 2]
            y2 = _y2 + oh * offset[idxs][:, 3]

            boxes = np.array([x1, y1, x2, y2, cls]).T
        return ways.NMS(np.array(boxes),0.3,FT=True)
if __name__ == '__main__':
    path =r"F:\xy.mp4"
    cap = cv2.VideoCapture(path)
    detect=Detector()
    while (True):
        ret, farm = cap.read()
        farm = farm[:, :, [2, 1, 0]]
        img = Image.fromarray(farm)
        a = img.width
        b = img.height
        count=2
        if count%2==0:
            boxes = detect.detect(img)
            imgDraw = ImageDraw.Draw(img)

        for box in boxes:
            x1 = int(box[0])
            y1 = int(box[1])
            x2 = int(box[2])
            y2 = int(box[3])

            imgDraw.rectangle((x1, y1, x2, y2), outline='red',width=3)
            pic_data = img.resize([a, b])
            frame = np.array(pic_data)[:, :, [2, 1, 0]]
            cv2.imshow('xy', frame)
            cv2.waitKey(1)
            if ret == False:
                break
        count+=1
    cv2.destroyAllWindows()