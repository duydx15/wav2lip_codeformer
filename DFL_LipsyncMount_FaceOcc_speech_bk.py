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
from color_transfer import color_transfer, color_hist_match,linear_color_transfer
from common import  normalize_channels
import LandmarksProcessor
from FaceType import FaceType
import torch
from torch import nn
from torchvision import transforms as TF_s
from scipy.spatial import Delaunay
# from filterpy.kalman import KalmanFilter
# from filterpy.common import Q_discrete_white_noise
from scipy import spatial
import skimage
import json
from collections import OrderedDict
import os
from GFPGAN.gfpgan.utils import GFPGANer
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
# from retinaface_h5.model import retinaface_model
# from retinaface_h5.commons import preprocess, postprocess
# from _wav2lip.gfp_gan_ import load_gfpgan_model
import math
import face_detection
from wav2lip_pipeline_bkforFOMM import my_cv2_resize,merge_color,gfpgan_img,gen_lipsync_img,get_fm_landmark,load_FaceOcc,load_speech_timestamp,load_model,lipsync_one_frame
import segmentation_models_pytorch as smp
# import tqdm


# gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.2)
# config=tf.compat.v1.ConfigProto(gpu_options=gpu_options)
# session = tf.compat.v1.Session(config=config)
# from oneEuroFilterArrayInput import OneEuroFilter
# from estimate_sharpness import estimate_sharpness
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


# FACEMESH_lips_1 = [206, 97, 326, 426, 2, 164, 94, 19, 61, 291, 13]
# ## Tho new mask mouth
# # FACEMESH_lips_2 = [326, 423,425 ,411,416, 430, 431, 262, 428, 199, 208, 32, 211, 210,192, 187, 205 , 203, 97]
#
# # new mask outter for Stone64
# FACEMESH_lips_2 = [326, 327,426,436,432,430,431,262, 428, 199, 208, 32, 211, 210, 212, 216,206,98,  2, 97]
# # new mask inner for Stone64
# FACEMESH_lips_3 = [164, 393,391,322,410,287,422,424, 418, 421, 200,201, 194, 204, 202,  57,186,92,165,167]
#
# # # big_mask origin
# # FACEMESH_bigmask = [197, 419, 399, 437, 355, 371, 266, 425, 411, 416,
# #                     394, 395,369, 396, 175, 171, 140, 170, 169,
# #                     192, 187, 205, 36, 142, 126, 217, 174, 196,
# #                     185, 40, 39, 37, 0, 267, 269, 270, 409,
# #                     186, 92, 165, 167, 164, 393, 391, 322, 410]
# FACEMESH_bigmask = [197, 419, 399, 437, 355, 371, 266, 425, 411, 416,
#                     394, 395,369, 396, 175, 171, 140, 170, 169,
#                     192, 187, 205, 36, 142, 126, 217, 174, 196,
#                     164, 391, 436, 432, 430, 273, 409, 11, 185, 210, 212, 216, 165]
#
# FACEMESH_pose_estimation = [34,264,168,33, 263]
#
# landmark_points_68 = [162,234,93,58,172,136,149,148,152,377,378,365,397,288,323,454,389,71,63,105,66,107,336,
#                   296,334,293,301,168,197,5,4,75,97,2,326,305,33,160,158,133,153,144,362,385,387,263,373,
#                   380,61,39,37,0,267,269,291,405,314,17,84,181,78,82,13,312,308,317,14,87,
#                   206, 97, 326, 426, 2, 164, 94, 19, 61, 291]
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
FACEMESH_lips_2_left = list_keypoint[streamer]["FACEMESH_lips_2"]
FACEMESH_lips_2_right = list_keypoint[streamer]["FACEMESH_lips_2"]
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

def green_blue_swap(img):
    # 3-channel image (no transparency)
    image = None
    if img.shape[2] == 3:
        image = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    # 4-channel image (with transparency)
    if img.shape[2] == 4:
        b,g,r,a = cv2.split(image)
        img[:,:,0] = r
        img[:,:,1] = g
        img[:,:,2] = b
        image = img[:,:,0:2]
    return image

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

# build retinaface model .h5
# def build_model():

#     global model #singleton design pattern

#     if not "model" in globals():

#         model = tf.function(
#             retinaface_model.build_model(),
#             input_signature=(tf.TensorSpec(shape=[None, None, None, 3], dtype=np.float32),)
#         )

#     return model


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
        # cv2.putText(inimg, text='Yaw: '+str(yaw), org=(100, 200), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1.1, color=(0, 255, 0),thickness=2)
        # cv2.putText(inimg, text='Pitch: '+str(pitch), org=(100, 250), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1.1, color=(0, 255, 0),thickness=2)
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
        # cv2.putText(inimg, text='Angle: '+str(face_angle), org=(100, 150), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1.1, color=(0, 255, 0),thickness=2)
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

def getKeypointByMediapipe(face_mesh_wide, videoimg,face_mesh_256, faceimg, kfVF, kfVM, kfFF, kfFM, mobile_net,resnet_net,device):
    videopts, videopts_out, videopts_big,videopts_lips_small, video_crops_coors,detect_fail,count_loss_detect_Retina,previous_model,yaw,face_angle,score_retina,pitch = get_face_by_RetinaFace(face_mesh_wide, videoimg, mobile_net, resnet_net, device, kfVF, kfVM)

    if videopts is None:
        return None, None, None, None,None, None,video_crops_coors,detect_fail,None,None,None,None
    facepts, facepts_out,facepts_big,facepts_lips_small = get_face(face_mesh_256, faceimg, yaw,pitch, kfFF, None)
    return videopts, videopts_out, videopts_big, facepts, facepts_out,facepts_big,facepts_lips_small, video_crops_coors,detect_fail,face_angle,score_retina,pitch

def ffmpeg_encoder(outfile, fps, width, height):
    frames = ffmpeg.input(
        "pipe:0",
        format="rawvideo",
        pix_fmt="rgb24",
        vsync="1",
        s='{}x{}'.format(width, height),
        r=fps,
    )

    encoder_ = subprocess.Popen(
        ffmpeg.compile(
            ffmpeg.output(
                frames,
                outfile,
                pix_fmt="yuv420p",
                vcodec="libx264",
                acodec="copy",
                r=fps,
                crf=17,
                vsync="1",
            )
            .global_args("-hide_banner")
            .global_args("-nostats")
            .global_args(
                "-loglevel",
                LOGURU_FFMPEG_LOGLEVELS.get(
                    os.environ.get("LOGURU_LEVEL", "INFO").lower()
                ),
            ),
            overwrite_output=True,
        ),
        stdin=subprocess.PIPE,
        # stdout=subprocess.DEVNULL,
        # stderr=subprocess.DEVNULL,
    )
    return encoder_

def blursharpen(img, sharpen_mode=0, kernel_size=3, amount=100):
    if kernel_size % 2 == 0:
        kernel_size += 1
    if amount > 0:
        if sharpen_mode == 1: #box
            kernel = np.zeros( (kernel_size, kernel_size), dtype=np.float32)
            kernel[ kernel_size//2, kernel_size//2] = 1.0
            box_filter = np.ones( (kernel_size, kernel_size), dtype=np.float32) / (kernel_size**2)
            kernel = kernel + (kernel - box_filter) * amount
            return cv2.filter2D(img, -1, kernel)
        elif sharpen_mode == 2: #gaussian
            blur = cv2.GaussianBlur(img, (kernel_size, kernel_size) , 0)
            img = cv2.addWeighted(img, 1.0 + (0.5 * amount), blur, -(0.5 * amount), 0)
            return img
    elif amount < 0:
        n = -amount
        while n > 0:

            img_blur = cv2.medianBlur(img, 5)
            if int(n / 10) != 0:
                img = img_blur
            else:
                pass_power = (n % 10) / 10.0
                img = img*(1.0-pass_power)+img_blur*pass_power
            n = max(n-10,0)

        return img
    return

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

def find_best_source(list_source, driving, face_mesh):
    bestidx = 0

    driving_ypr = get_yaw_pitch_roll(face_mesh, driving)
    if driving_ypr[0] is None:
        return None

    if driving_ypr[0] > 50:
        return 1
    elif driving_ypr[0] < -50:
        return 2
    else:
        return 0

    cosine_similarity = [np.dot(driving_ypr, value)/(norm(driving_ypr)*norm(value)) for value in list_source]
    bestidx = cosine_similarity.index(max(cosine_similarity))
    return bestidx

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
    if point60[1] > point72[1]:
        roll = -roll
    roll = int(roll)

    return yaw, pitch, roll
def init_source_list(list_img_source):
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
                            max_num_faces=1,
                            # min_tracking_confidence=0.5,
                            min_detection_confidence=0.2)

    source_image = []
    for value in list_img_source:

        yaw_pitch_roll = get_yaw_pitch_roll(face_mesh, value)
        # if yaw_pitch_roll[0] is not None:
        source_image.append(yaw_pitch_roll)

    source_image = [source_image[4],source_image[1],source_image[8]]
    print(source_image)
    return source_image

def add_image_by_mask(img1, img2, mask_):
    mask_not = cv2.bitwise_not(mask_)
    img2_no_mask = cv2.bitwise_and(img2, img2, mask=mask_not)
    img1_mask_only = cv2.bitwise_and(img1, img1, mask=mask_)
    return cv2.add(img2_no_mask, img1_mask_only)

def write_block(block,encoder):
    global num_frame_loss,original_frames,select_block,drawed_frames,\
    block_frames_count,num_mouth_frame_loss,check_not_speech,write_video,\
    speak_check, count_frame, lipsync_success,lipsync_fomm,lipsync_wav2lip

    for images in block:
        # output_fr +=1
        write_frame(images,encoder_video)
    num_frame_loss = 0
    original_frames = []
    select_block = []
    drawed_frames = []
    block_frames_count = 0
    num_mouth_frame_loss = 0
    check_not_speech =0
    write_video = False
    speak_check=0
    lipsync_success = 0
    lipsync_fomm= 0
    lipsync_wav2lip = 0

def mix_pixel(pix_1, pix_2, perc):
    return (perc/255 * pix_1) + ((255 - perc)/255 * pix_2)

# function for blending images depending on values given in mask
def blend_images_using_mask(img_orig, img_for_overlay, img_mask):
    if len(img_mask.shape) != 3:
        img_mask = cv2.cvtColor(img_mask, cv2.COLOR_GRAY2BGR)
    img_res = mix_pixel(img_orig, img_for_overlay, img_mask)
    return img_res.astype(np.uint8)

def lipsync_replace(videoimg,face_img,facepts_out,videopts_out,img_video_mask_mount,img_video_mask_face,alpha,center_mount,type_seamless=1):
    global center_beard,videoimg_copy,num_mouth_frame_loss
    center_cop = center_mount
    img_video_mask_face = cv2.bitwise_or(img_video_mask_face, img_video_mask_mount)
    topy, topx, bottomy, bottomx, center_face = mask2box(img_video_mask_face)

    result = add_image_by_mask(face_img, videoimg, img_video_mask_face)
    # cv2.imwrite("results_bk.png",result)
    alpha_0 = alpha
    # result[topy:bottomy, topx:bottomx] = np.mean(result[topy:bottomy, topx:bottomx])
    result[topy:bottomy, topx:bottomx] = cv2.addWeighted(result[topy:bottomy, topx:bottomx],\
    alpha_0, videoimg[topy:bottomy, topx:bottomx], 1-alpha_0, 0.0)

    img_bgr_uint8_1 = normalize_channels(result[topy:bottomy, topx:bottomx], 3)
    img_bgr_1 = img_bgr_uint8_1.astype(np.float32) / 255.0
    img_bgr_1 = np.clip(img_bgr_1, 0, 1)
    img_bgr_uint8_2 = normalize_channels(videoimg[topy:bottomy, topx:bottomx], 3)
    img_bgr_2 = img_bgr_uint8_2.astype(np.float32) / 255.0
    img_bgr_2 = np.clip(img_bgr_2, 0, 1)
    if type_seamless ==1:
        result_new = linear_color_transfer(img_bgr_1, img_bgr_2)
        final_img = color_hist_match(result_new, img_bgr_2, 255).astype(dtype=np.float32)

        result[topy:bottomy, topx:bottomx] = blursharpen((final_img*255).astype(np.uint8), 1, 5, 0.5)

    #addWeighted to results after match color
    alpha_0 = alpha
    result[topy:bottomy, topx:bottomx] = cv2.addWeighted(result[topy:bottomy, topx:bottomx],\
    alpha_0, videoimg[topy:bottomy, topx:bottomx], 1-alpha_0, 0.0)
    # print("Max mask mouth:",np.unique(img_video_mask_mount))
    # m_topy, m_topx, m_bottomy,/ m_bottomx, center_mount = mask2box(img_video_mask_mount)
    img_video_mask_mount = cv2.dilate(img_video_mask_mount, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)), iterations = 1 )
    img_video_mask_face_2 = cv2.erode(img_video_mask_face, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)), iterations = 1 )
    img_video_mask_mount_error = cv2.bitwise_xor(img_video_mask_mount, img_video_mask_face_2)
    img_video_mask_mount = img_video_mask_mount - img_video_mask_mount_error
    if type_seamless == 0:
        img_video_mask_mount[img_video_mask_mount<255]=0
    # print("center_mount before:",center_mount)
    m_topy, m_topx, m_bottomy, m_bottomx,center_mount = mask2box(img_video_mask_mount)
    if m_topy <5 or m_topx <5 or m_bottomy > height_-5 or m_bottomx > width_-5:
        num_mouth_frame_loss +=1
        return videoimg_copy
    if face_angle ==0:
        if pitch <=pitch_up_threshold :
            img_video_mask_mount[img_video_mask_mount<255]=0
            r_point_face = facepts_out.reshape(-1,2)[FACEMESH_lips_2.index(2)]
            r_point_video = videopts_out[FACEMESH_lips_2.index(2)]
            # print(r_point_face,r_point_video)
            del_y = int(r_point_video[1]-r_point_face[1])
            del_x = int(r_point_video[0]-r_point_face[0])
            center_mount = np.array(center_cop)
            # print("center_mount before:",center_mount)
            center_mount[0] = center_mount[0]+del_x
            center_mount[1] = center_mount[1]+del_y
            center_mount = tuple(center_mount)
    # print("center_mount lipsync:",center_mount)
    result = add_image_by_mask(result, videoimg, img_video_mask_mount)
    output_main = cv2.seamlessClone(result, videoimg, img_video_mask_mount, center_mount, cv2.NORMAL_CLONE)
    output_main = cv2.seamlessClone(result, output_main, img_video_mask_mount, center_mount, cv2.NORMAL_CLONE)

    return output_main

def write_frame(images,encoder_video):
    image_draw = cv2.cvtColor(images,cv2.COLOR_RGB2BGR)
    imageout = Image.fromarray(np.uint8(image_draw))
    encoder_video.stdin.write(imageout.tobytes())

font = cv2.FONT_HERSHEY_SIMPLEX

def load_gfpgan_model(model_path, device):
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
    bg_upsampler = RealESRGANer(
        scale=2,
        model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
        model=model,
        tile=200,
        tile_pad=10,
        pre_pad=0,
        half=False if device=='cpu' else True)  # need to set False in CPU mode
    arch = 'clean'
    channel_multiplier = 2
    model_name = 'GFPGANv1.3'
    scale =1
    restorer = GFPGANer(
                        model_path=model_path,
                        upscale=scale,
                        arch=arch,
                        channel_multiplier=channel_multiplier,
                        bg_upsampler=bg_upsampler)

    return restorer

if __name__ == "__main__":

    #Define path of model/videos/....
    FACE_PATH = '/home/ubuntu/first-order-model/Dr_ES_driving_16Aug_crop.mp4'
    # FACE_PATH = '/home/ubuntu/Duy_test_folder/Retinaface_Mediapipe/Dr_video/3sourceheadMovCSV_22Aug_Dr_ES_gfpgan.mp4'
    FRAME_PATH = '/home/ubuntu/Duy_test_folder/Retinaface_Mediapipe/Dr_video/DrDisrespect-Falls-in-Love-with-Warzone-again-thanks-to-new-Game-Mode-30FPS.mp4'
    device = 'cuda'
    wavpath = 'speaker00es.wav'
    wav2lip_modelpath = '/home/ubuntu/quyennv/DeepFake/_wav2lip/wav2lip_gan.pth'
    gfpgan_modelpath = '/home/ubuntu/Duy_test_folder/Retinaface_Mediapipe/GFPGAN/experiments/pretrained_models/GFPGANv1.3.pth'
    output_path = '/home/ubuntu/Duy_test_folder/Retinaface_Mediapipe/Dr_video/30Oct_Dr_ES_FOMM_debug.mp4'

    #Load model
    sfd_facedetector = face_detection.FaceAlignment(face_detection.LandmarksType._2D,
                                        flip_input=False, device='cuda')
    wav2lip_model = load_model(wav2lip_modelpath, device)
    model_Occ,to_tensor = load_FaceOcc()
    mobile_net, resnet_net = loadmodelface()
    gfpgan_model = load_gfpgan_model(gfpgan_modelpath, device)

    with open("/home/ubuntu/Duy_test_folder/SP_Speechdetection/content/SpeakerDetection/Mel_chunks_29Aug_DrES.json", "r") as outfile:
        mel_chunks = json.load(outfile)

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh_256 = mp_face_mesh.FaceMesh(min_detection_confidence=0.2, min_tracking_confidence=0.2,max_num_faces=1)
    face_mesh_wide = mp_face_mesh.FaceMesh(min_detection_confidence=0.5)
    face_mesh_wav2lip = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)
    facecout = 0
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))

    capFrame = cv2.VideoCapture(FRAME_PATH)
    capFace = cv2.VideoCapture(FACE_PATH)
    fps = capFrame.get(cv2.CAP_PROP_FPS)
    width_  = int(capFrame.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_ = int(capFrame.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width_Face  = capFace.get(cv2.CAP_PROP_FRAME_WIDTH)
    height_Face = capFace.get(cv2.CAP_PROP_FRAME_HEIGHT)

    encoder_video = ffmpeg_encoder(output_path, fps, width_, height_)
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
    original_frames = []
    drawed_frames = []
    select_block = []
    write_video = None
    lipsync_success = 0
    lipsync_fomm= 0
    lipsync_wav2lip = 0

    #const variable
    fps_per_block = 10
    num_frame_threshold = 1
    minute_start = 0
    second_start = 16
    minute_stop =0
    second_stop =45
    frame_start = int(minute_start*60*fps+second_start*fps)
    frame_stop = int(minute_stop*60*fps+second_stop*fps)
    # frame_start = 15200
    # frame_stop = 15400
    total_f = frame_stop-frame_start
    pbar = tqdm(total=total_f)
    output_fr =0
    center_beard = None
    videoimg_copy = []
    # skin = cv2.imread("/home/ubuntu/Duy_test_folder/Retinaface_Mediapipe/Dr_skin.png")

    while capFrame.isOpened():
        count_frame = count_frame+1
        okay1  , videoimg = capFrame.read()
        okay2 , faceimg = capFace.read()
        # mel_chunk = np.array(mel_chunks[str(count_frame-1)])
        # faceimg = cv2.imread("/home/ubuntu/first-order-model/data/11.jpg")
        # cv2.putText(videoimg, text='Fr:'+str(count_frame), org=(100, 40), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1.1, color=(0, 255, 0),thickness=2)
        videoimg_copy = videoimg.copy()
        #cal frame for block_frames util reach num_frame_threshold
        # print(count_frame)
        pbar.update(1)

        if count_frame <= frame_start:
            continue
        elif count_frame > frame_stop:
            if lipsync_success >=5:
                write_block(drawed_frames,encoder_video)
            else:
                write_block(original_frames,encoder_video)
            break
        elif count_frame > frame_start and count_frame <= frame_stop:
            # print("Lost frame:",count_frame)

            # okay2 , faceimg = capFace.read()
            if not okay1 or not okay2 :
                print('Cant read the video , Exit!')
                break

            if not count_frame in List_speak_frame:
                check_not_speech +=1
                if lipsync_success >=5:
                    write_block(drawed_frames,encoder_video)
                else:
                    write_block(original_frames,encoder_video)
                # print("Not speak",count_frame)
                output_fr +=1
                write_frame(videoimg_copy,encoder_video)
                # original_frames.append(videoimg_copy)
                continue
            else:
                block_frames_count += 1
                speak_check +=1

                video_h,video_w, = videoimg.shape[:2]
                face_h,face_w, = faceimg.shape[:2]

                videopts, videopts_out, videopts_beard, facepts, facepts_out, facepts_beard,facepts_lips_small, video_crops_coors,detect_fail,face_angle, score_retina,pitch = getKeypointByMediapipe(\
                face_mesh_wide, videoimg, face_mesh_256, faceimg, trackkpVideoFace, trackkpVideoMount, trackkpFaceFace, trackkpFaceMount, mobile_net,resnet_net,device)
                if videopts is None or facepts is None :
                    print("Mediapipe loss:",count_frame,score_retina,videopts is None,facepts is None)
                    num_frame_loss +=1
                    original_frames.append(videoimg_copy)
                    drawed_frames.append(videoimg_copy)

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
                            print("Loss mouth bbox",count_frame,"|",m_topy, m_topx, m_bottomy, m_bottomx)
                            drawed_frames.append(videoimg_copy)
                            original_frames.append(videoimg_copy)
                            num_mouth_frame_loss += 1

                        else:
                            #Cal mask and get new_mouth_point of videoimg by FaceOcc
                            newval,mount_mask_Occ,face_mask_Occ,video_newpoints_mouth = find_ratio_intersection(videoimg,video_crops_coors,videopts_out,\
                            FaceOcc_only = False,mask_mount = img_video_mask_mount_bk)
                            # cv2.putText(videoimg, text='Score:'+str(score_retina)+'-Occ ratio: '+str(newval), org=(100, 100), fontFace=cv2.FONT_HERSHEY_TRIPLEX,fontScale=1.1, color=(0, 255, 0),thickness=2)
                            if newval > 0.8902 or  (newval > 0.835 and face_angle == 1) or (newval > 0.84 and face_angle !=1 and score_retina > 0.99): #(newval > 0.8902) or
                                img_video_mask_mount = mount_mask_Occ
                                # img_video_mask_face = face_mask_Occ
                                M_face, _ = cv2.findHomography(facepts,videopts)
                                facepts_out = cv2.perspectiveTransform(facepts_out.astype(np.float32), M_face).astype(np.int32)
                                M_mount, _ = cv2.findHomography(facepts_out, videopts_out)

                                # if face_angle ==1  or pitch >=21 or (face_angle == 2 and pitch >= 4) or lipsync_wav2lip >=3:
                                #     lipsync_wav2lip +=1
                                #     print("Wav2lip:",pitch,face_angle)       #Lipsync by wav2lip
                                #     output_main = lipsync_one_frame(videoimg, mel_chunk, face_mesh_wav2lip, FACEMESH_lips_2,  wav2lip_model, gfpgan_model,mobile_net, resnet_net, device,to_tensor,model_Occ,sfd_facedetector)
                                # elif face_angle ==0  or pitch <21 or (face_angle == 2 and pitch <4) or lipsync_fomm >=3:   #Lipsync by FOMM
                                #     lipsync_fomm += 1
                                # print("FOMM:",pitch,face_angle)
                                #Cal mask and get new_mouth_point of faceimg by FaceOcc
                                # img_face_mask_mount_bk = np.zeros((face_h,face_w), np.uint8)
                                # cv2.fillPoly(img_face_mask_mount_bk, pts =[facepts_out], color=(255,255,25  5))
                                # newval_faceimg,faceimg_mount_mask_Occ,faceimg_face_mask_Occ,faceimg_newpoints_mouth = find_ratio_intersection(faceimg,None,facepts_out,\
                                # FaceOcc_only = False,mask_mount = img_face_mask_mount_bk)
                                #
                                # faceimg_mount_mask_Occ =cv2.warpPerspective(faceimg_mount_mask_Occ,M_face,(video_w, video_h))
                                facepts_beard = cv2.perspectiveTransform(facepts_beard.astype(np.float32), M_face).astype(np.int32)
                                facepts_lips_small = cv2.perspectiveTransform(facepts_lips_small.astype(np.float32), M_face).astype(np.int32)
                                face_img = cv2.warpPerspective(faceimg,M_face,(video_w, video_h))

                                facepts_beard = cv2.perspectiveTransform(facepts_beard.astype(np.float32), M_mount).astype(np.int32)
                                facepts_lips_small = cv2.perspectiveTransform(facepts_lips_small.astype(np.float32), M_mount).astype(np.int32)
                                face_img = cv2.warpPerspective(face_img,M_mount,(video_w, video_h), cv2.INTER_LINEAR)

                                output_main = lipsync_replace(videoimg,face_img,facepts_out,videopts_out,img_video_mask_mount,img_video_mask_face,0.95,center_mount,type_seamless = 1)

                                # cv2.polylines(output_main, [videopts_out],True,color=(0,0,255),thickness=2)
                                # cv2.polylines(output_main, [facepts_out],True,color=(0,255,0),thickness=2)

                                #Cal ratio Fake_mouth_mask and Real_mouth_mask
                                # real_mouth_mask = np.zeros((video_h,video_w), np.uint8)
                                # fak_mouth_mask = np.zeros((video_h,video_w), np.uint8)
                                # facepts_out = cv2.perspectiveTransform(facepts_out.astype(np.float32), M_mount).astype(np.int32)
                                # cv2.fillPoly(real_mouth_mask, pts =[videopts_out], color=(255,255,255))
                                # cv2.fillPoly(fak_mouth_mask, pts =[facepts_out], color=(255,255,255))
                                # real_mouth_mask = np.atleast_3d(real_mouth_mask).astype(np.float) / 255.
                                # real_mouth_mask[real_mouth_mask != 1] = 0
                                # fak_mouth_mask = np.atleast_3d(fak_mouth_mask).astype(np.float) / 255.
                                # fak_mouth_mask[fak_mouth_mask != 1] = 0
                                # F_R_ratio = len(fak_mouth_mask[fak_mouth_mask > 0])/len(real_mouth_mask[real_mouth_mask >0])
                                # cv2.putText(output_main, text='F_R_ratio:' + str(F_R_ratio), org=(100, 300), fontFace=cv2.FONT_HERSHEY_TRIPLEX,fontScale=1.1, color=(0, 255, 0),thickness=2)
                                drawed_frames.append(output_main)
                                lipsync_success +=1
                                original_frames.append(videoimg_copy)
                            else:
                                print("Lost frame:",count_frame,"Ratio:", newval)
                                videoimg_copy = videoimg
                                drawed_frames.append(videoimg_copy)
                                original_frames.append(videoimg_copy)
                                num_mouth_frame_loss += 1
                        # except Exception:
                        #     print("Failed to try")
                        #     drawed_frames.append(videoimg_copy)
                        #     original_frames.append(videoimg_copy)

                # if count_frame == 15:
                #     break
                if num_frame_loss > num_frame_threshold or num_mouth_frame_loss > num_frame_threshold:# or check_not_speech > num_frame_threshold:
                    select_block = original_frames
                    check_bl = "Normally"
                    write_video = True
                elif block_frames_count >= fps_per_block:
                    select_block = drawed_frames
                    check_bl = "Lipsynced"
                    write_video = True
                if write_video:
                    write_block(select_block,encoder_video)

    del  face_mesh_wav2lip, wav2lip_model, gfpgan_model, mel_chunks
    pbar.close()
    encoder_video.stdin.flush()
    encoder_video.stdin.close()

    # img_bgr_uint8_occ = normalize_channels(mount_mask_Occ, 3)
    # img_bgr_occ = img_bgr_uint8_occ.astype(np.int32)*255
    # img_bgr_occ[img_bgr_occ>0] =255
    # img_bgr_occ[:,:,0] = 0
    # img_bgr_occ[:,:,1] = 0
    # alpha_0 = 0.15
    # output_main = cv2.addWeighted(img_bgr_occ, alpha_0,output_main, 1-alpha_0, 0, dtype = cv2.CV_32F)
    print("=================\n","DONE!\n","=================")
