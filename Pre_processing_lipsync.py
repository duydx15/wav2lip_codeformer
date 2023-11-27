import cv2
import mediapipe as mp
import os
from tqdm import tqdm
import numpy as np
import math
from PIL import Image
import ffmpeg
import subprocess
from pipeline_mobile_resnet_2 import loadmodelface, detection_face
from common import  normalize_channels
import LandmarksProcessor
from FaceType import FaceType
import torch
from torch import nn
from torchvision import transforms as TF_s
from scipy.spatial import Delaunay
from scipy import spatial
import skimage
import json
from collections import OrderedDict
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import math
import segmentation_models_pytorch as smp


# gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.2)
# config=tf.compat.v1.ConfigProto(gpu_options=gpu_options)
# session = tf.compat.v1.Session(config=config)
LOGURU_FFMPEG_LOGLEVELS = {
    "trace": "trace",
    "debug": "debug",
    "info": "info",
    "success": "info",
    "warning": "warning",
    "error": "error",
    "critical": "fatal",
}


import warnings
warnings.filterwarnings("ignore")

class KalmanTracking(object):
    # init kalman filter object
    def __init__(self, point):
        deltatime = 1/30 # 30fps
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0],
         [0, 1, 0, 0]], np.float32)

        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 0, 1]], np.float32)

        self.kalman.processNoiseCov = np.array([[1, 0, 0, 0],
         [0, 1, 0, 0],
         [0, 0, 1, 0],
         [0, 0, 0, 1]], np.float32) * deltatime # 0.03
        # self.kalman.measurementNoiseCov = np.array([[1, 0],
        #                                             [0, 1]], np.float32) *0.1
        self.measurement = np.array((point[0], point[1]), np.float32)

    def getpoint(self, kp):
        self.kalman.correct(kp-self.measurement)
        # get new kalman filter prediction
        prediction = self.kalman.predict()
        prediction[0][0] = prediction[0][0] +  self.measurement[0]
        prediction[1][0] = prediction[1][0] +  self.measurement[1]

        return prediction

def binaryMaskIOU_(mask1, mask2):
    mask1_area = torch.count_nonzero(mask1)
    mask2_area = torch.count_nonzero(mask2)
    # print("mask1_area ", mask1_area)
    # print("mask2_area ", mask2_area)
    intersection = torch.count_nonzero(torch.logical_and(mask1, mask2))
    iou = intersection / (mask1_area + mask2_area - intersection)
    return iou.numpy()

class KalmanArray(object):
    def __init__(self):
        self.kflist = []
        # self.oldmask = None
        # self.resetVal = 1
        self.w = 0
        self.h = 0

    def noneArray(self):
        return len(self.kflist) == 0

    def setpoints(self, points, w=1920, h=1080):
        for value in points:
            intpoint = np.array([np.float32(value[0]), np.float32(value[1])], np.float32)
            self.kflist.append(KalmanTracking(intpoint))

        self.w = w
        self.h = h
        # self.oldmask = np.zeros(image_shape[0:2]+(1,),dtype=np.float32)
        # print('setpoints ', self.w)

    def getpoints(self, kps):
        # print('old ', kps[:3])
        # print("KPS:",len(kps),'\n', kps)
        # print("Kflist",len(self.kflist))
        orginmask = np.zeros((self.h,self.w),dtype=np.float32)
        orginmask = cv2.fillConvexPoly(orginmask, np.array(kps[:], np.int32), 1) #kps[:-18]
        kps_o = kps.copy()
        # print("kps",len(kps))
        # print("kflist",len(self.kflist))
        if len(kps) <= len(self.kflist):
            kps_final = len(kps)
        else:
            kps_final = len(self.kflist)
        for i in range(kps_final):
            # print(i)
            # kps[i] = kflist[i]
            intpoint = np.array([np.float32(kps[i][0]), np.float32(kps[i][1])], np.float32)
            tmp = self.kflist[i].getpoint(intpoint)
            kps[i] = (tmp[0][0], tmp[1][0])

        newmask = np.zeros((self.h,self.w),dtype=np.float32)
        newmask = cv2.fillConvexPoly(newmask, np.array(kps[:], np.int32), 1) #kps[:-18]
        # cv2.imwrite('orginmask.jpg' , orginmask*255)
        val = binaryMaskIOU_(torch.from_numpy(orginmask), torch.from_numpy(newmask))
        # print('binaryMaskIOU_ ', val)
        # distance = spatial.distance.cosine(orgindata, newdata)
        # print(distance)
        if val < 0.9:
            del self.kflist[:]
            # self.oldmask = None
            self.setpoints(kps_o,self.w, self.h)
            return kps_o
        # self.olddata = newdata
        # print('new ', kps[:3])
        return kps

def cross_point(line1, line2):
    x1 = line1[0]
    y1 = line1[1]
    x2 = line1[2]
    y2 = line1[3]

    x3 = line2[0]
    y3 = line2[1]
    x4 = line2[2]
    y4 = line2[3]

    k1 = (y2 - y1) * 1.0 / (x2 - x1)
    b1 = y1 * 1.0 - x1 * k1 * 1.0
    if (x4 - x3) == 0:
        k2 = None
        b2 = 0
    else:
        k2 = (y4 - y3) * 1.0 / (x4 - x3)
        b2 = y3 * 1.0 - x3 * k2 * 1.0
    if k2 == None:
        x = x3
    else:
        x = (b2 - b1) * 1.0 / (k1 - k2)
    y = k1 * x * 1.0 + b1 * 1.0
    return [x, y]

def point_line(point,line):
    x1 = line[0]
    y1 = line[1]
    x2 = line[2]
    y2 = line[3]

    x3 = point[0]
    y3 = point[1]

    k1 = (y2 - y1)*1.0 /(x2 -x1)
    b1 = y1 *1.0 - x1 *k1 *1.0
    k2 = -1.0/k1
    b2 = y3 *1.0 -x3 * k2 *1.0
    x = (b2 - b1) * 1.0 /(k1 - k2)
    y = k1 * x *1.0 +b1 *1.0
    return [x,y]

def point_point(point_1,point_2):
    x1 = point_1[0]
    y1 = point_1[1]
    x2 = point_2[0]
    y2 = point_2[1]
    # distance = ((x1-x2)**2 +(y1-y2)**2)**0.5
    distance = math.sqrt((x1-x2) ** 2 + (y1-y2) ** 2)
    # distance = math.hypot(x2-x2, y1-y2)
    # if distance == 0:
    #     distance = distance + 0.1
    return distance

def facePose(point1, point31, point51, point60, point72):
    crossover51 = point_line(point51, [point1[0], point1[1], point31[0], point31[1]])
    yaw_mean = point_point(point1, point31) / 2
    yaw_right = point_point(point1, crossover51)
    yaw = (yaw_mean - yaw_right) / yaw_mean
    if math.isnan(yaw):
        return None, None, None
    yaw = int(yaw * 71.58 + 0.7037)

    #pitch
    pitch_dis = point_point(point51, crossover51)
    if point51[1] < crossover51[1]:
        pitch_dis = -pitch_dis
    if math.isnan(pitch_dis):
        return None, None, None
    pitch = int(1.497 * pitch_dis + 18.97)

    #roll
    roll_tan = abs(point60[1] - point72[1]) / abs(point60[0] - point72[0])
    roll = math.atan(roll_tan)
    roll = math.degrees(roll)
    if math.isnan(roll):
        return None, None, None
    if point60[1] >point72[1]:
        roll = -roll
    roll = int(roll)

    return yaw, pitch, roll

def is_on_line(x1, y1, x2, y2, x3):
    slope = (y2 - y1) / (x2 - x1)
    return slope * (x3 - x1) + y1

debuglist = [8, 298, 301, 368, 347, 329, 437, 168, 217, 100, 118, 139, 71, 68, #sunglass
    164, 391, 436, 432, 430, 273, 409, 11, 185, 210, 212, 216, 165,  #mustache
    13, 311, 308, 402, 14, 178, 78, 81, #lips Inner
    0, 269, 291, 375, 405, 17, 181, 146, 61, 39, #lips Outer
    4, 429, 327, 2, 98, 209, #noise
    151, 200, 175, 65, 69, 194, 140, 205, 214, 135, 215,
    177, 137, 34, 295, 299, 418, 369,
    425, 434, 364, 435, 401, 366, 264, 43]

with open("/home/ubuntu/Duy_test_folder/Retinaface_Mediapipe/config_lipsync.json",'r') as f_lips:
	list_keypoint = json.load(f_lips)
streamer = "Dr"
right_threshold = list_keypoint[streamer]["right_threshold"]
left_threshold = list_keypoint[streamer]["left_threshold"]
straight_threshold = list_keypoint[streamer]["straight_threshold"]
pitch_up_threshold = list_keypoint[streamer]["pitch_up_threshold"]
FACEMESH_lips_1 = list_keypoint[streamer]["FACEMESH_lips_1"]
FACEMESH_lips = list_keypoint[streamer]["FACEMESH_lips"]
FACEMESH_lips_2 = list_keypoint[streamer]["FACEMESH_lips_2"]
FACEMESH_lips_2_face_up = list_keypoint[streamer]["FACEMESH_lips_2_face_up"]
FACEMESH_lips_2_up = list_keypoint[streamer]["FACEMESH_lips_2_up"]
FACEMESH_lips_2_left = list_keypoint[streamer]["FACEMESH_lips_2_left"]
FACEMESH_lips_2_right = list_keypoint[streamer]["FACEMESH_lips_2_right"]
FACEMESH_lips_intermediate_left = list_keypoint[streamer]["FACEMESH_lips_2"]
FACEMESH_lips_intermediate_right = list_keypoint[streamer]["FACEMESH_lips_2"]
FACEMESH_beard_face = list_keypoint[streamer]["FACEMESH_beard_face"]
FACEMESH_beard = list_keypoint[streamer]["FACEMESH_beard"]
FACEMESH_beard_inner = list_keypoint[streamer]["FACEMESH_beard_inner"]
FACEMESH_pose_estimation = list_keypoint[streamer]["FACEMESH_pose_estimation"]
landmark_points_68 = list_keypoint[streamer]["landmark_points_68"]
FACEMESH_bigmask = list_keypoint[streamer]["FACEMESH_bigmask"]
lips_pts_global = []
#"FACEMESH_lips_2":[326, 423,425 ,411,416, 430, 431, 262, 428, 199, 208, 32, 211, 210,192, 187, 205 , 203, 97],


def pointInRect(point, rect):
    x1, y1, x2, y2 = rect
    wbox = abs(x2-x1)
    xo = (x1+x2)/2
    yo = (y1+y2)/2
    x, y = point
    dist1 = math.hypot(x-xo, y-yo)

    aaa = dist1/wbox if wbox>0 else 1
    # print(dist1, ' ', wbox, ' ',aaa)
    # print('cur: ', point, '\told: ',  (xo,yo))
    # print('oldbox: ', rect)
    if (x1 < x and x < x2):
        if (y1 < y and y < y2):
            if aaa <= 0.06:
                return True
    return False

def facemeshTrackByBox(multi_face_landmarks,  w, h):
    # print(bbox)
    for faceIdx, face_landmarks in enumerate(multi_face_landmarks):
        listpoint = []
        for i in range(len(FACEMESH_lips_1)):
            idx = FACEMESH_lips_1[i]
            x = face_landmarks.landmark[idx].x
            y = face_landmarks.landmark[idx].y

            realx = x * w
            realy = y * h
            listpoint.append((realx, realy))

        video_leftmost_x = min(x for x, y in listpoint)
        video_bottom_y = min(y for x, y in listpoint)
        video_rightmost_x = max(x for x, y in listpoint)
        video_top_y = max(y for x, y in listpoint)

        # x = (video_leftmost_x+video_rightmost_x)/2
        y = (video_bottom_y+video_top_y)/2
        # point = (x,y)
        # print(point, ' ', h, w)
        if y < h/2:
            continue
        # if pointInRect(point, bbox):
        return faceIdx
    return -1


def facemeshTrackByBoxCrop(multi_face_landmarks,  w, h):
    for faceIdx, face_landmarks in enumerate(multi_face_landmarks):
        listpoint = []
        for i in range(len(FACEMESH_lips_1)):
            idx = FACEMESH_lips_1[i]
            x = face_landmarks.landmark[idx].x
            y = face_landmarks.landmark[idx].y

            realx = x * w
            realy = y * h
            listpoint.append((realx, realy))

        video_leftmost_x = min(x for x, y in listpoint)
        video_bottom_y = min(y for x, y in listpoint)
        video_rightmost_x = max(x for x, y in listpoint)
        video_top_y = max(y for x, y in listpoint)

        y = (video_bottom_y+video_top_y)/2
        # point = (x,y)
        # print(point, ' ', h, w)
        if y < h/2:
            continue
        # if pointInRect(point, bbox):
        return faceIdx
    return -1


def get_face_by_RetinaFace(facea, inimg, mobile_net, resnet_net, device, kf = None,kf_driving = None):

    h,w = inimg.shape[:2]
    count_loss_detect = 0
    padding_ratio = [0.3,0.1]
    detect_fail = False
    previous_model = 0
    crop_image = None
    #rgb_frame = cv2.cvtColor(inimg,cv2.COLOR_BGR2RGB)
    # results = detect_faces_retinaface_tensor(rgb_frame)
    # print(results)
    for r_idx in range(len(padding_ratio)):
        l_coordinate, detected_face,score_retina = detection_face(mobile_net,resnet_net, inimg, device,padding_ratio[r_idx])
        if not detected_face and r_idx == len(padding_ratio)-1:
            count_loss_detect = 1
            detect_fail = True
            # print("New_model loss")
            return None, None, None,None,None,detect_fail,count_loss_detect,previous_model,None,None,None,None
        elif not detected_face and r_idx < len(padding_ratio)-1:
          previous_model =  1
          continue
        face_landmarks = None
        bbox = None

        crop_images_coors = None
        # print('get_face_by_RetinaFace ', l_coordinate)
        for i in range(len(l_coordinate)):
            topleft, bottomright = l_coordinate[i]
            if bottomright[1] < h/3:
                continue

            crop_image = inimg[topleft[1]:bottomright[1], topleft[0]:bottomright[0],:]
            crop_images_coors = [topleft[1],bottomright[1], topleft[0],bottomright[0]]

            results = facea.process(cv2.cvtColor(crop_image, cv2.COLOR_BGR2RGB))
            if not results.multi_face_landmarks:
                continue
            face_landmarks = results.multi_face_landmarks[0]
            # for detection in results_2.detections:

            curbox = []
            curbox.append(topleft[0])
            curbox.append(topleft[1])
            curbox.append(bottomright[0])
            curbox.append(bottomright[1])
            bbox = curbox


        if not face_landmarks and r_idx == len(padding_ratio)-1:
            # print("Medipipe loss")
            return None, None, None, None,crop_images_coors, detect_fail,count_loss_detect,previous_model,None,None,None,None
        elif not face_landmarks and r_idx < len(padding_ratio)-1 :
            # print("Medipipe skip")
            continue

        # print('bbox ', bbox)
        # print("Medipipe detected")
        bbox_w = bbox[2] - bbox[0]
        bbox_h = bbox[3] - bbox[1]
        #Cal yaw_distance for face
        posePoint = []
        for i in range(len(FACEMESH_pose_estimation)):
            idx = FACEMESH_pose_estimation[i]
            x = face_landmarks.landmark[idx].x
            y = face_landmarks.landmark[idx].y

            realx = x * bbox_w + bbox[0]
            realy = y * bbox_h + bbox[1]
            posePoint.append((realx, realy))

        listpointLocal = []
        for i in range(len(FACEMESH_bigmask)):
            idx = FACEMESH_bigmask[i]
            x = face_landmarks.landmark[idx].x
            y = face_landmarks.landmark[idx].y

            # realx = x * bbox_w + bbox[0]
            # realy = y * bbox_h + bbox[1]
            listpointLocal.append((x, y))

        listpoint = []

        for i in range(len(listpointLocal)):
            # idx = FACEMESH_bigmask[i]
            x = listpointLocal[i][0]
            y = listpointLocal[i][1]

            realx = x * bbox_w + bbox[0]
            realy = y * bbox_h + bbox[1]
            listpoint.append((realx, realy))

        if kf is not None:
            if kf.noneArray():
                # kf.setpoints(listpointLocal, 1e-03)
                kf.setpoints(listpoint, w, h)

            else:
                listpoint = kf.getpoints(listpoint)

        srcpts = np.array(listpoint, np.int32)
        srcpts = srcpts.reshape(-1,1,2)

        yaw, pitch, roll = facePose(posePoint[0], posePoint[1], posePoint[2], posePoint[3], posePoint[4])
        # print("Pitch:",pitch)
        # Set threshold for yaw - face_angle
        #Get Lips_points based on mask of face's angle
        lips_pts = None
        if yaw is None or (yaw >= left_threshold and yaw < right_threshold) :
            face_angle = 0
            if pitch > pitch_up_threshold or pitch is None:
                lips_pts = FACEMESH_lips_2
            else:
                lips_pts = FACEMESH_lips_2_up
            lips_pts_global = lips_pts

        elif yaw >= right_threshold:
            face_angle = 1
            lips_pts = FACEMESH_lips_2_left
        elif yaw <  left_threshold:
            face_angle = 1
            lips_pts = FACEMESH_lips_2_right
        if  (~(straight_threshold-1) > yaw  and yaw >= left_threshold) or (right_threshold > yaw and yaw > straight_threshold):
            face_angle = 2
        cv2.putText(inimg, text='Angle: '+str(face_angle), org=(100, 150), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1.1, color=(0, 255, 0),thickness=2)
        # print("Yaw:",yaw,"Pitch:",pitch)
        # print(lips_pts)
        listpoint2 = []
        for i in range(len(lips_pts)):
            idx = lips_pts[i]
            x = face_landmarks.landmark[idx].x
            y = face_landmarks.landmark[idx].y

            realx = x * bbox_w + bbox[0]
            realy = y * bbox_h + bbox[1]
            listpoint2.append((realx,realy))

            # idx_out = FACEMESH_lips_3[i]
            # x_out = face_landmarks.landmark[idx_out].x
            # y_out = face_landmarks.landmark[idx_out].y
            #
            # realx_out = x_out * bbox_w + bbox[0]
            # realy_out = y_out * bbox_h + bbox[1]
            #
            # listpoint2.append(((realx+realx_out)/2, (realy+realy_out)//2))

        if kf_driving is not None:
            if kf_driving.noneArray():
                kf_driving.setpoints(listpoint2, w, h)
            else:
                listpoint2 = kf_driving.getpoints(listpoint2)
        srcpts2 = np.array(listpoint2, np.int32)
        # srcpts2 = srcpts2.reshape(-1,1,2)

        listpoint3 = []
        for i in range(len(FACEMESH_beard)):
            idx = FACEMESH_beard[i]
            x = face_landmarks.landmark[idx].x
            y = face_landmarks.landmark[idx].y

            realx = x * bbox_w + bbox[0]
            realy = y * bbox_h + bbox[1]
            listpoint3.append((realx, realy))

        # print(listpoint3.simplices)
        srcpts3 = np.array(listpoint3, np.int32)
        # srcpts3 = srcpts3.reshape(-1,1,2)

        lips_pts_small = FACEMESH_lips
        listpoint_small = []
        for i in range(len(lips_pts_small)):
            idx = lips_pts_small[i]
            x = face_landmarks.landmark[idx].x
            y = face_landmarks.landmark[idx].y

            realx = x * bbox_w + bbox[2]
            realy = y * bbox_h + bbox[0]
            listpoint_small.append((realx, realy))
        srcpts_lips_small = np.array(listpoint_small, np.int32)
        return srcpts, srcpts2, srcpts3,srcpts_lips_small, crop_images_coors, detect_fail,count_loss_detect,previous_model,yaw,face_angle,score_retina,pitch


def get_face(facea, inimg, yaw,pitch,kf=None, kfMount=None, iswide = False):
    listpoint = []
    yaws = yaw
    h,w = inimg.shape[:2]
    # print(inimg.shape[:2])

    results = facea.process(cv2.cvtColor(inimg, cv2.COLOR_BGR2RGB))
    if not results.multi_face_landmarks:
        return None, None,None,None

    face_landmarks = results.multi_face_landmarks[0]

    listpointLocal = []
    for i in range(len(FACEMESH_bigmask)):
        idx = FACEMESH_bigmask[i]
        x = face_landmarks.landmark[idx].x
        y = face_landmarks.landmark[idx].y

        # realx = x * bbox_w + bbox[0]
        # realy = y * bbox_h + bbox[1]
        listpointLocal.append((x, y))

    listpoint = []

    for i in range(len(listpointLocal)):
        # idx = FACEMESH_bigmask[i]
        x = listpointLocal[i][0]
        y = listpointLocal[i][1]

        realx = x * w
        realy = y * h
        listpoint.append((realx, realy))

    if kf is not None:
        if kf.noneArray():
            kf.setpoints(listpoint, w, h)

        else:
            listpoint = kf.getpoints(listpoint)

    srcpts = np.array(listpoint, np.int32)
    srcpts = srcpts.reshape(-1,1,2)

    # lips_pts = None

    if ~(straight_threshold-1)<= yaws and yaws <= straight_threshold:
        if pitch > pitch_up_threshold or pitch is None:
            lips_pts = FACEMESH_lips_2
        else:
            lips_pts = FACEMESH_lips_2_up
    elif  yaws >= right_threshold:
         lips_pts = FACEMESH_lips_2_left
    elif  yaws < left_threshold:
        lips_pts = FACEMESH_lips_2_right
    elif  ~(straight_threshold-1) > yaws  and yaws >= left_threshold:
        lips_pts = FACEMESH_lips_intermediate_right
    elif   right_threshold > yaws and yaws > straight_threshold:
        lips_pts = FACEMESH_lips_intermediate_left
    # print("FACE_angle:", face_angles,"List points: ", lips_pts)

    # lips_pts = FACEMESH_lips_2
    listpoint2 = []
    # print("Yaw",yaws)
    for i in range(len(lips_pts)):
        idx = lips_pts[i]
        x = face_landmarks.landmark[idx].x
        y = face_landmarks.landmark[idx].y

        realx = x * w
        realy = y * h

        listpoint2.append((realx,realy))
        # idx_out = FACEMESH_lips_3[i]
        # x_out = face_landmarks.landmark[idx_out].x
        # y_out = face_landmarks.landmark[idx_out].y
        #
        # realx_out = x_out * w
        # realy_out = y_out * h
        # listpoint2.append(((realx+realx_out)//2, (realy+realy_out)//2))

    # for idx in range(448):
    #     x = face_landmarks.landmark[idx].x
    #     y = face_landmarks.landmark[idx].y
    #
    #     realx = int(x * w)
    #     realy = int(y * h)
    #     cv2.circle(inimg,(realx,realy),2,(0,255,0),-1)
    #     cv2.putText(inimg,str(idx),(realx,realy),fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.35, color=(0, 255, 0),thickness=1)

    if kfMount is not None:
        if kfMount.noneArray():
            kfMount.setpoints(listpoint2, w, h)
        else:
            listpoint2 = kfMount.getpoints(listpoint2)

    srcpts2 = np.array(listpoint2, np.int32)
    # srcpts = np.concatenate((srcpts, srcpts2), axis=0)
    srcpts2 = srcpts2.reshape(-1,1,2)
    # cv2.polylines(inimg, [srcpts2],True,color=(0,0,255),thickness=2)
    # cv2.imwrite("FacE_extract.png",inimg)
    listpoint3 = []
    for i in range(len(FACEMESH_beard)):
        idx = FACEMESH_beard[i]
        x = face_landmarks.landmark[idx].x
        y = face_landmarks.landmark[idx].y

        realx = x * w
        realy = y * h
        listpoint3.append((realx, realy))

    # print(listpoint3.simplices)
    srcpts3 = np.array(listpoint3, np.int32)
    srcpts3 = srcpts3.reshape(-1,1,2)

    lips_pts_small = FACEMESH_lips
    listpoint_small = []
    for i in range(len(lips_pts_small)):
        idx = lips_pts_small[i]
        x = face_landmarks.landmark[idx].x
        y = face_landmarks.landmark[idx].y

        realx = x * w
        realy = y * h
        listpoint_small.append((realx, realy))
    srcpts_lips_small = np.array(listpoint_small, np.int32)
    srcpts_lips_small = srcpts_lips_small.reshape(-1,1,2)

    return srcpts, srcpts2,srcpts3,srcpts_lips_small

def getKeypointByMediapipe(face_mesh_wide, videoimg, kfVF, kfVM, mobile_net,resnet_net,device):
    videopts, videopts_out, videopts_big,videopts_lips_small, video_crops_coors,detect_fail,count_loss_detect_Retina,previous_model,yaw,face_angle,score_retina,pitch = get_face_by_RetinaFace(face_mesh_wide, videoimg, mobile_net, resnet_net, device, kfVF, kfVM)

    # if videopts is None:
    #     return None, None, None, None,None, None,None,None
    # facepts, facepts_out,facepts_big,facepts_lips_small = get_face(face_mesh_256, faceimg, yaw,pitch, kfFF, None)
    return videopts, videopts_out, videopts_big, video_crops_coors,detect_fail,face_angle,score_retina,pitch

def mask2box(mask2d):
    (y, x) = np.where(mask2d > 0)
    topy = np.min(y)
    topx = np.min(x)
    bottomy = np.max(y)
    bottomx = np.max(x)
    # (topy, topx) = (np.min(y), np.min(x))
    # (bottomy, bottomx) = (np.max(y), np.max(x))
    center_ = (int((topx+bottomx)/2),int((bottomy+topy)/2))

    return topy, topx, bottomy, bottomx, center_

def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    # print(dst.size)
    return dst
def get_concat_v(im1, im2):
    dst = Image.new('RGB', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst

def applyAffineTransform(src, srcTri, dstTri, size):
    # Given a pair of triangles, find the affine transform.
    warpMat = cv2.getAffineTransform(np.float32(srcTri), np.float32(dstTri))

    # Apply the Affine Transform just found to the src image
    dst = cv2.warpAffine(src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    return dst
def fillhole(input_image):
    # input_image = 255 - input_image

    im_flood_fill = input_image.copy()
    h, w = input_image.shape[:2]
    masktmp = np.zeros((h + 2, w + 2), np.uint8)
    im_flood_fill = im_flood_fill.astype("uint8")
    cv2.floodFill(im_flood_fill, masktmp, (0, 0), 255)
    im_flood_fill_inv = cv2.bitwise_not(im_flood_fill)
    img_out = input_image | im_flood_fill_inv

    # bighole = bwareaopen(im_flood_fill_inv, int(w/3))
    # img_out = img_out - bighole
    return img_out



def find_ratio_intersection(videoimg,crops_coors,videopts_out,FaceOcc_only = None,mask_mount=None):
    if not crops_coors is None:
        crop_image = videoimg[crops_coors[0]:crops_coors[1],crops_coors[2]:crops_coors[3],:]
        crop_image_occ = np.copy(crop_image)
        video_h,video_w, = videoimg.shape[:2]
        crop_image_occ = cv2.resize(crop_image_occ,(256,256)) # (256,256) is input size of FaceOcc

        #Predict mask by FaceOcc
        data = to_tensor(crop_image_occ).unsqueeze(0)
        data = data.to(device)
        with torch.no_grad():
            pred = model_Occ(data)
        pred_mask = (pred > 0).type(torch.int32)
        pred_mask = pred_mask.squeeze().cpu().numpy()
        mask_occ = cv2.resize(pred_mask,(crop_image.shape[1],crop_image.shape[0]),interpolation=cv2.INTER_LINEAR_EXACT) #Resize up need INTER_LINEAR_EXACT

        img_face_occ = np.zeros((video_h,video_w), np.uint8)
        img_face_occ[crops_coors[0]:crops_coors[1],crops_coors[2]:crops_coors[3]] = mask_occ
    else:
        crop_image_occ = np.copy(videoimg)
        video_h,video_w, = videoimg.shape[:2]
        crop_image_occ = cv2.resize(crop_image_occ,(256,256)) # (256,256) is input size of FaceOcc

        #Predict mask by FaceOcc
        data = to_tensor(crop_image_occ).unsqueeze(0)
        data = data.to(device)
        with torch.no_grad():
            pred = model_Occ(data)
        pred_mask = (pred > 0).type(torch.int32)
        pred_mask = pred_mask.squeeze().cpu().numpy()
        mask_occ = cv2.resize(pred_mask,(videoimg.shape[1],videoimg.shape[0]),interpolation=cv2.INTER_LINEAR_EXACT) #Resize up need INTER_LINEAR_EXACT

        img_face_occ = np.zeros((video_h,video_w), np.uint8)
        img_face_occ[:,:] = mask_occ
    new_mask_corners = None
    img_face_occ = fillhole(img_face_occ*255)
    # print("Max Face Occ:", np.max(img_face_occ)," ", img_face_occ.shape)
    #Check output condition
    if FaceOcc_only is True:
        img_face_occ = fillhole(img_face_occ)
        img_bgr_uint8_occ = normalize_channels(img_face_occ, 3)
        img_bgr_occ = img_bgr_uint8_occ.astype(np.int32)*255
        img_bgr_occ[img_bgr_occ>0] =255
        img_bgr_occ[:,:,0] = 0
        img_bgr_occ[:,:,2] = 0
        alpha_0 = 0.6
        cropimage = cv2.addWeighted(img_bgr_occ, alpha_0,videoimg, 1-alpha_0, 0, dtype = cv2.CV_32F)
        return cropimage
    else:
        var_middle_occ = np.copy(mask_mount[:,:])
        # var_middle_occ[var_middle_occ >0]=1
        img_mouth_mask_occ = cv2.bitwise_and(img_face_occ,var_middle_occ)
        # print("Max mouth Occ:", np.max(img_mouth_mask_occ)," ", img_mouth_mask_occ.shape)
        mount_mask_Occ = img_mouth_mask_occ.copy()


        #Find corner for new mask mouth by cv2.goodFeaturesToTrack(image, maxCorners, qualityLevel, minDistance, [,mask[,blockSize[,useHarrisDetector[,k]]]])
        # new_mask_corners = cv2.goodFeaturesToTrack(mount_mask_Occ,len(videopts_out),0.2,15)
        # if not new_mask_corners is None:
        #     new_mask_corners = np.int0(new_mask_corners)
            # new_mask_corners = sorted(new_mask_corners, key=clockwiseangle_and_distance)
        #cal ratio between 2 mouth mask Mediapipe and FaceOcc
        mask_mount = np.atleast_3d(mask_mount).astype(np.float) / 255.
        mask_mount[mask_mount != 1] = 0

        img_mouth_mask_occ = np.atleast_3d(img_mouth_mask_occ).astype(np.float) / 255.
        img_mouth_mask_occ[img_mouth_mask_occ != 1] = 0
        newval = len(img_mouth_mask_occ[img_mouth_mask_occ > 0])/len(mask_mount[mask_mount >0])
        return newval,mount_mask_Occ,img_face_occ,new_mask_corners

def get_yaw_pitch_roll(face_mesh, img_driving):
    visualization = (255 * img_driving).astype(np.uint8)
    results = face_mesh.process(cv2.cvtColor(visualization, cv2.COLOR_BGR2RGB))
    if not results.multi_face_landmarks:
        return (None, None, None)

    face_landmarks = results.multi_face_landmarks[0]
    posePoint = []
    for i in range(len(FACEMESH_pose_estimation)):
        idx = FACEMESH_pose_estimation[i]
        x = face_landmarks.landmark[idx].x
        y = face_landmarks.landmark[idx].y

        realx = x * 256
        realy = y * 256
        posePoint.append((realx, realy))

    # hullPoint = []
    # for i in range(len(FACEMESH_hull)):
    #     idx = FACEMESH_hull[i]
    #     x = face_landmarks.landmark[idx].x
    #     y = face_landmarks.landmark[idx].y
    #
    #     realx = x * 256
    #     realy = y * 256
    #     hullPoint.append((realx, realy))
    yaw, pitch, roll = facePose(posePoint[0], posePoint[1], posePoint[2], posePoint[3], posePoint[4])
    return (yaw, pitch, roll)

def load_speech_timestamp(fps):
    List_speak_frame = []
    json_path = '/home/ubuntu/Duy_test_folder/SP_Speechdetection/content/SpeakerDetection/Speech_detected_Dr2206_ES_full_voiceonly_tmp.json'
    with open(json_path,'r') as f:
        list_inital = json.load(f)
        List_speak_frame = []#list_inital['data'][:]
        for j in range(len(list_inital['data'])):
            start = int(float(list_inital['data'][j]['start'])*fps)-3
            stop = int(float(list_inital['data'][j]['end'])*fps)
            # print(start, stop)
            for i in range(start,stop+1):
                List_speak_frame.append(i)
    return np.unique(List_speak_frame)

def load_FaceOcc():
    ENCODER = 'resnet18'
    ENCODER_WEIGHTS = 'imagenet'
    CLASSES = 1
    ATTENTION = None
    ACTIVATION = None
    DEVICE = 'cuda:0'
    # root_mask = './Dataset/FaceOcc/COFW_test/mask'
    to_tensor = TF_s.ToTensor()
    model_Occ = smp.Unet(encoder_name=ENCODER,
                     encoder_weights=ENCODER_WEIGHTS,
                     classes=CLASSES,
                     activation=ACTIVATION)

    # model = nn.DataParallel(model.to(DEVICE), device_ids=[0, 1])
    weights = torch.load('/home/ubuntu/Duy_test_folder/FaceExtraction/FaceOcc/FaceOcc/epoch_16_best.ckpt')
    new_weights = OrderedDict()
    for key in weights.keys():
        new_key = '.'.join(key.split('.')[1:])
        new_weights[new_key] = weights[key]

    model_Occ.load_state_dict(new_weights)
    model_Occ.to(DEVICE)
    model_Occ.eval()
    return model_Occ,to_tensor

def update_frame(block,type_lips):
    global List_frame_ori,List_frame_fomm,List_frame_wav2lip,previous_lipsync,block_frames,block_tmp
    if type_lips =="ori":
            List_frame_ori = [*List_frame_ori,*block]

    elif type_lips == "fomm":
            List_frame_fomm = [*List_frame_fomm,*block]

    elif type_lips == "wav2lip":
            List_frame_wav2lip = [*List_frame_wav2lip,*block]

    block_frames =[]
    block_tmp = []
    previous_lipsync = type_lips
    # return np.array(List_frame)

font = cv2.FONT_HERSHEY_SIMPLEX

def mostFrequent(arr, n):
    # Sort the array
    arr.sort()
    # find the max frequency using
    # linear traversal
    max_count = 1
    res = arr[0]
    curr_count = 1
    for i in range(1, n):
        if (arr[i] == arr[i - 1]):
            curr_count += 1
        else:
            curr_count = 1
         # If last element is most frequent
        if (curr_count > max_count):
            max_count = curr_count
            res = arr[i - 1]

    return res

if __name__ == "__main__":
    #
    # # #Define path of model/videos/....
    FACE_PATH = '/home/ubuntu/Duy_test_folder/Retinaface_Mediapipe/Dr_video/3sourceheadMovCSV_22Aug_Dr_ES_gfpgan.mp4'
    FRAME_PATH = '/home/ubuntu/Duy_test_folder/Retinaface_Mediapipe/Dr_video/DrDisrespect-Falls-in-Love-with-Warzone-again-thanks-to-new-Game-Mode-30FPS.mp4'
    device = 'cuda'
    wavpath = 'speaker00es.wav'
    gfpgan_modelpath = '/home/ubuntu/Duy_test_folder/Retinaface_Mediapipe/GFPGAN/experiments/pretrained_models/GFPGANv1.3.pth'
    output_path = '/home/ubuntu/Duy_test_folder/Retinaface_Mediapipe/Dr_video/12Sep_Dr_ES_debug.mp4'

    #Load model
    model_Occ,to_tensor = load_FaceOcc()
    mobile_net, resnet_net = loadmodelface()
    #
    # with open("/home/ubuntu/Duy_test_folder/SP_Speechdetection/content/SpeakerDetection/Mel_chunks_29Aug_DrES.json", "r") as outfile:
    #     mel_chunks = json.load(outfile)

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh_wide = mp_face_mesh.FaceMesh(min_detection_confidence=0.5)
    facecout = 0
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))

    capFrame = cv2.VideoCapture(FRAME_PATH)
    capFace = cv2.VideoCapture(FACE_PATH)
    fps = capFrame.get(cv2.CAP_PROP_FPS)
    width_  = int(capFrame.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_ = int(capFrame.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width_Face  = capFace.get(cv2.CAP_PROP_FRAME_WIDTH)
    height_Face = capFace.get(cv2.CAP_PROP_FRAME_HEIGHT)

    totalF = int(capFrame.get(cv2.CAP_PROP_FRAME_COUNT))
    # pbar = tqdm(total=totalF)
    configOEF = {
        'freq': fps,  # Hz
        'mincutoff': 1.0,  # FIXME
        'beta': 1.0,  # FIXME
        'dcutoff': 1.0,  # this one should be ok
        'array_len': len(FACEMESH_bigmask)*2
    }

    trackkpVideoFace = KalmanArray()
    trackkpVideoMount = KalmanArray()
    trackkpFaceFace = KalmanArray()
    trackkpFaceMount = KalmanArray()
    # Read speech json timline file
    List_speak_frame = load_speech_timestamp(fps)

    check_not_speech = 0
    speak_check = 0

    #dynamic variable
    count_frame = 0
    # limit_frame = range(12860, 12900+1)
    num_frame_loss = 0
    num_mouth_frame_loss = 0
    block_frames_count = 0
    block_frames = []
    write_video = None
    lipsync_success = 0
    lipsync_fomm= 0
    lipsync_wav2lip = 0
    #
    # #const variable
    fps_per_block = 10
    num_frame_threshold = 1
    minute_start = 0
    second_start = 0
    minute_stop =10
    second_stop =40
    frame_start = int(minute_start*60*fps+second_start*fps)
    frame_stop = int(minute_stop*60*fps+second_stop*fps)
    # frame_start = 10650
    # frame_stop = 10680
    total_f = frame_stop-frame_start
    pbar = tqdm(total=total_f)
    output_fr =0
    center_beard = None
    videoimg_copy = []
    #
    List_frame_fomm = []
    List_frame_wav2lip = []
    List_frame_ori = []
    List_frame = []
    update_wav2lips = False
    update_fomm = False
    previous_lipsync ="ori"
    print("Pre-processing detect type of lipsync")
    while capFrame.isOpened():
        count_frame = count_frame+1
        okay1  , videoimg = capFrame.read()
        # okay2 , faceimg = capFace.read()
        # mel_chunk = np.array(mel_chunks[str(count_frame-1)])
        # faceimg = cv2.imread("/home/ubuntu/first-order-model/data/11.jpg")
        #cal frame for block_frames util reach num_frame_threshold
        # print(count_frame)
        pbar.update(1)

        if count_frame <= frame_start:
            continue
        elif count_frame > frame_stop:
            break
        elif count_frame > frame_start and count_frame <= frame_stop:
            # print("List_frame:",count_frame)
            # okay2 , faceimg = capFace.read()
            if not okay1 :
                print('Cant read the video , Exit!')
                break

            if not count_frame in List_speak_frame:
                check_not_speech +=1
                List_frame_ori.append(count_frame)
                List_frame.append("ori")

                continue
            else:
                block_frames_count += 1
                speak_check +=1

                video_h,video_w, = videoimg.shape[:2]
                # face_h,face_w, = faceimg.shape[:2]

                videopts, videopts_out, videopts_beard, video_crops_coors,detect_fail,face_angle, score_retina,pitch = getKeypointByMediapipe(
                face_mesh_wide, videoimg,  trackkpVideoFace, trackkpVideoMount,  mobile_net,resnet_net,device)
                if videopts is None  :
                    # print("Mediapipe loss:",count_frame,score_retina,videopts is None)
                    num_frame_loss +=1
                    List_frame_ori.append(count_frame)

                    List_frame.append("ori")
                else:
                    # try:
                        #get mask of mediapipe
                        img_video_mask_mount = np.zeros((video_h,video_w), np.uint8)
                        img_video_mask_face = np.zeros((video_h,video_w), np.uint8)
                        cv2.fillPoly(img_video_mask_mount, pts =[videopts_out], color=(255,255,255))
                        cv2.fillPoly(img_video_mask_face, pts =[videopts[:-13]], color=(255,255,255))
                        img_video_mask_mount_bk = np.zeros((video_h,video_w), np.uint8)
                        cv2.fillPoly(img_video_mask_mount_bk, pts =[videopts_out], color=(255,255,255))

                        m_topy, m_topx, m_bottomy, m_bottomx, center_mount = mask2box(img_video_mask_mount)
                        if m_topy <5 or m_topx <5 or m_bottomy > height_-5 or m_bottomx > width_-5:# or newval < 0.8902 or (newval < 0.83 and face_angle != 0):
                            # print("Loss mouth bbox",count_frame,"|",m_topy, m_topx, m_bottomy, m_bottomx)

                            List_frame.append("ori")
                        else:
                            #Cal mask and get new_mouth_point of videoimg by FaceOcc
                            newval,mount_mask_Occ,face_mask_Occ,video_newpoints_mouth = find_ratio_intersection(videoimg,video_crops_coors,videopts_out,\
                            FaceOcc_only = False,mask_mount = img_video_mask_mount_bk)
                            if newval > 0.8902 or  (newval > 0.835 and face_angle == 1) or (newval > 0.85 and face_angle !=1 and score_retina > 0.99): #(newval > 0.8902) or
                                if face_angle ==1  or pitch >=21 or (face_angle == 2 and pitch >= 4 and newval > 0.89):
                                    lipsync_wav2lip +=1
                                    List_frame.append("wav2lip")

                                    # print("Newval",newval, "- wav2lip:",count_frame)
                                    # if lipsync_wav2lip ==1 and previous_lipsync !="ori":
                                    #     update_fomm = True
                                elif face_angle ==0  or pitch <21 or (face_angle == 2 and pitch <4) or (face_angle == 2 and pitch >= 4 and newval < 0.89):   #Lipsync by FOMM
                                    # print("Newval",newval, "- fomm:",count_frame)
                                    lipsync_fomm += 1
                                    List_frame.append("fomm")
                                    # if lipsync_fomm ==1 and previous_lipsync !="ori":
                                    #     update_wav2lips = True
                                # if update_fomm:
                                #     List_frame_fomm = update_frame(block,List_frame_fomm)


                                lipsync_success +=1
                            else:
                                # print("Lost frame:",count_frame,"Ratio:", newval)
                                num_mouth_frame_loss += 1
                                List_frame_ori.append(count_frame)

                                List_frame.append("ori")

    pbar.close()
    data = {"List_frame":List_frame}
    json_object = json.dumps(data, indent = 4)
    # Writing to sample.json
    with open("/home/ubuntu/Duy_test_folder/Retinaface_Mediapipe/Dr_9p_lipsync_timestamp_raw2.json", "w") as outfile:
        outfile.write(json_object)
        # data = json.load(outfile)
    # List_frame = data["List_frame"]
    # Optimize list type_lipsync
    previous_lipsync = "ori"
    print("OPTIMIZE type of lipsync ",len(List_frame))
    idx_frame = 0
    skip_list =[]
    block_frames = []
    pbar_1 = tqdm(len(List_frame))
    for j in range(len(List_frame)):
        # pbar_1.update(1)


        idx_frame +=1
        print("Fr:",idx_frame)
        if j in skip_list:
            continue
        elif j == len(List_frame)-1:
            update_frame(block_frames,previous_lipsync)
        else:
            curr_lipsync = List_frame[j]
            if curr_lipsync == previous_lipsync:
                # print("check normal")
                block_frames.append(idx_frame)
            else:
                update_frame(block_frames,previous_lipsync)
                #check next block after update to prevent 1 fr swap
                start_fr = idx_frame-1
                stop_fr = idx_frame + int((fps_per_block)/2)-1
                # print(start_fr,"|",stop_fr)
                block_tmp = List_frame[start_fr:stop_fr]
                type_lipsync = mostFrequent(block_tmp,len(block_tmp))
                # print(type_lipsync)
                # if type_lipsync == curr_lipsync:
                #     # print("check good")
                #     continue
                # else:
                    # print("check bad")
                    # block_tmp[:] = type_lipsync
                update_frame([*range(start_fr+1,stop_fr+1)],type_lipsync)
                skip_list = [*range(start_fr,stop_fr)]
                # print("Skip list:", skip_list)
    # List_frame_ori = np.unique(List_frame_ori)
    # List_frame_fomm = np.unique(List_frame_fomm)
    # List_frame_wav2lip = np.unique(List_frame_wav2lip)
    # print(len(List_frame_ori),len(List_frame_fomm),len(List_frame_wav2lip),len(List_frame_ori)+len(List_frame_fomm)+len(List_frame_wav2lip))
    # print("Error fomm:",[fruit for fruit in List_frame_ori if fruit in List_frame_fomm])
    # print("Error wav2lip:",[fruit for fruit in List_frame_ori if fruit in List_frame_wav2lip])
    data = {"List_frame_ori":List_frame_ori,
            "List_frame_fomm":List_frame_fomm,
            "List_frame_wav2lip":List_frame_wav2lip}
    print(List_frame_wav2lip)
    json_object = json.dumps(data, indent = 4)
    # Writing to sample.json
    with open("/home/ubuntu/Duy_test_folder/Retinaface_Mediapipe/Dr_9p_lipsync_timestamp_2.json", "w") as outfile:
        outfile.write(json_object)


    # img_bgr_uint8_occ = normalize_channels(mount_mask_Occ, 3)
    # img_bgr_occ = img_bgr_uint8_occ.astype(np.int32)*255
    # img_bgr_occ[img_bgr_occ>0] =255
    # img_bgr_occ[:,:,0] = 0
    # img_bgr_occ[:,:,1] = 0
    # alpha_0 = 0.15
    # output_main = cv2.addWeighted(img_bgr_occ, alpha_0,output_main, 1-alpha_0, 0, dtype = cv2.CV_32F)
    print("=================\n","DONE!\n","=================")
