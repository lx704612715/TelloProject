import cv2
import numpy as np
from djitellopy import tello
import cvzone

thres = 0.55
nmsThres = 0.2
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

classNames = []
classFile = '../models/coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().split('\n')
print(classNames)
configPath = '../models/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = "../models/frozen_inference_graph.pb"

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize((320, 320))
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)


# me = tello.Tello()
# me.connect()
# print(me.get_battery())
# me.streamoff()
# me.streamon()
#
# me.takeoff()
# me.move_up(80)


def get_overlap_bbox(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        print("separate")
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = abs(x_right - x_left) * abs(y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = abs(bb1['x2'] - bb1['x1']) * abs(bb1['y2'] - bb1['y1'])
    bb2_area = abs(bb2['x2'] - bb2['x1']) * abs(bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area

    if np.isclose(intersection_area, bb2_area):
        print("contain bb2_area")
    elif np.isclose(intersection_area, bb1_area):
        print("contain bb1_area")
    elif np.isclose(intersection_area, 0.0):
        print("separate")
    else:
        print("overlap")

    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    return iou


bb1 = dict()
bb1["x1"] = 0
bb1["y1"] = 0
bb1["x2"] = 2
bb1["y2"] = 2

bb2 = dict()
bb2["x1"] = 0.5
bb2["y1"] = 0.5
bb2["x2"] = 1
bb2["y2"] = 1

# contain
iou1 = get_overlap_bbox(bb1, bb2)

bb1 = dict()
bb1["x1"] = 0
bb1["y1"] = 0
bb1["x2"] = 2
bb1["y2"] = 2

bb2 = dict()
bb2["x1"] = 1
bb2["y1"] = 1
bb2["x2"] = 3
bb2["y2"] = 3

# intersection
iou2 = get_overlap_bbox(bb1, bb2)

bb1 = dict()
bb1["x1"] = 0
bb1["y1"] = 0
bb1["x2"] = 2
bb1["y2"] = 2

bb2 = dict()
bb2["x1"] = 3
bb2["y1"] = 3
bb2["x2"] = 5
bb2["y2"] = 5

# separate
iou3 = get_overlap_bbox(bb1, bb2)

print("print")

while True:
    success, img = cap.read()
    # img = me.get_frame_read().frame
    classIds, confs, bbox = net.detect(img, confThreshold=thres, nmsThreshold=nmsThres)
    try:
        for classId, conf, box in zip(classIds.flatten(), confs.flatten(), bbox):
            cvzone.cornerRect(img, box)
            cv2.putText(img, f'{classNames[classId - 1].upper()} {round(conf * 100, 2)}',
                        (box[0] + 10, box[1] + 30), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                        1, (0, 255, 0), 2)
    except Exception as e:
        print(e)
        pass

    cv2.imshow("image", img)
    cv2.waitKey(1)
#
