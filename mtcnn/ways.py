import numpy as np

def iou(box,boxes,FT=False):

    box_area=(box[2]-box[0])*(box[3]-box[1])
    boxes_area=(boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1])

    x1=np.maximum(box[0],boxes[:,0])
    y1=np.maximum(box[1],boxes[:,1])
    x2=np.minimum(box[2],boxes[:,2])
    y2=np.minimum(box[3],boxes[:,3])
    w=np.maximum(0,x2-x1)
    h=np.maximum(0,y2-y1)
    area=w*h

    if FT:
        percent=np.true_divide(area,np.minimum(box_area,boxes_area))
    else: # print(area)
        # print(box_area)
        # print(boxes_area)

        percent=np.true_divide(area,(box_area+boxes_area-area))
    return percent


def NMS(boxes,tresh=0.3,FT=False):

    if boxes.shape[0]==0:
        return np.array([])
    boxes = boxes[(-boxes[:, 4]).argsort()]
    c_boxes = []

    while boxes.shape[0]>1:
        a_box = boxes[0]
        b_boxes = boxes[1:]

        c_boxes.append(a_box)
        index=np.where(iou(a_box,b_boxes,FT)<tresh)
        boxes=b_boxes[index]

    if boxes.shape[0]>0:
        c_boxes.append(boxes[0])
    return np.stack(c_boxes)

def re_size(box):
    abox = box.copy()
    if box.shape[0] == 0:
        return np.array([])
    h = box[:, 3] - box[:, 1]
    w = box[:, 2] - box[:, 0]
    max_side = np.maximum(h, w)
    abox[:, 0] = box[:, 0] + w * 0.5 - max_side * 0.5
    abox[:, 1] = box[:, 1] + h * 0.5 - max_side * 0.5
    abox[:, 2] = abox[:, 0] + max_side
    abox[:, 3] = abox[:, 1] + max_side

    return abox

if __name__ == '__main__':

    bs = np.array([[1, 1, 10, 10, 40], [1, 1, 9, 9, 10], [9, 8, 13, 20, 15], [6, 11, 18, 17, 13]])
    print(NMS(bs))
