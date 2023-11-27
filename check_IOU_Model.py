import numpy as np 
import cv2 
import os 
import time


def bb_intersection_over_union(boxA, boxB):
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	iou = interArea / float(boxAArea + boxBArea - interArea)
	return iou


def load_l_bbox(path_file):
    l_bbox = []
    file_annotation = open(path_file,'r')
    for line in file_annotation:
        line = line[:-1]
        if len(line) == 0:
            bbox = []
            l_bbox.append(bbox)
        else:
            xmin, ymin, xmax, ymax = line.split(" ")
            xmin = int(xmin)
            ymin = int(ymin)
            xmax = int(xmax)
            ymax = int(ymax)
            bbox = ((xmin, ymin), (xmax, ymax))
            l_bbox.append(bbox)
    return l_bbox



l_resnet_bbox = load_l_bbox("Log_ResNet.txt")
l_mobilenet_bbox = load_l_bbox("Log_MobileNet.txt")

average_score = []
for index, bbox in enumerate(l_mobilenet_bbox):
    if len(bbox) == 0:
        continue
    annotation = l_resnet_bbox[index]
    if len(annotation) == 0:
        continue
    cv_bbox = (bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1])
    cv_annotation = (annotation[0][0], annotation[0][1], annotation[1][0], annotation[1][1])
    IOU_score = bb_intersection_over_union(cv_bbox, cv_annotation)
    # print(IOU_score)
    average_score.append(IOU_score)
print(np.mean(average_score))