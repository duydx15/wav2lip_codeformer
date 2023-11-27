import cv2
import mediapipe as mp
import os
from tqdm import tqdm
import numpy as np
import math
from PIL import Image
import ffmpeg
import subprocess
from pipeline_mobile_resnet import loadmodelface, detection_face
from color_transfer import color_transfer, color_transfer_mix, color_transfer_sot, color_transfer_mkl, color_transfer_idt, color_hist_match, reinhard_color_transfer, linear_color_transfer
from common import random_crop, normalize_channels, cut_odd_image, overlay_alpha_image
import LandmarksProcessor
from FaceType import FaceType
import segmentation_models_pytorch as smp
import torch
from torch import nn
from torchvision import transforms as TF_s
from scipy.spatial import Delaunay
# from filterpy.kalman import KalmanFilter
# from filterpy.common import Q_discrete_white_noise
from scipy import spatial
# from estimate_sharpness import estimate_sharpness
import skimage
import json
from collections import OrderedDict
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from retinaface_h5.model import retinaface_model
from retinaface_h5.commons import preprocess, postprocess
# import tqdm


gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.3)
config=tf.compat.v1.ConfigProto(gpu_options=gpu_options)
session = tf.compat.v1.Session(config=config)
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
        orginmask = np.zeros((self.h,self.w),dtype=np.float32)
        orginmask = cv2.fillConvexPoly(orginmask, np.array(kps[:-18], np.int32), 1)
        kps_o = kps.copy()
        for i in range(len(kps)):
            # print(i)
            # kps[i] = kflist[i]
            intpoint = np.array([np.float32(kps[i][0]), np.float32(kps[i][1])], np.float32)
            tmp = self.kflist[i].getpoint(intpoint)
            kps[i] = (tmp[0][0], tmp[1][0])

        newmask = np.zeros((self.h,self.w),dtype=np.float32)
        newmask = cv2.fillConvexPoly(newmask, np.array(kps[:-18], np.int32), 1)
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


with open("/home/ubuntu/quyennv/DeepFake/Retinaface_Mediapipe/config_lipsync.json",'r') as f_lips:
    list_keypoint = json.load(f_lips)
streamer = "Dr"
FACEMESH_lips_1 = list_keypoint[streamer]["FACEMESH_lips_1"]
FACEMESH_lips_2 = list_keypoint[streamer]["FACEMESH_lips_2"]
FACEMESH_lips_3 = list_keypoint[streamer]["FACEMESH_lips_3"]
FACEMESH_pose_estimation = list_keypoint[streamer]["FACEMESH_pose_estimation"]
landmark_points_68 = list_keypoint[streamer]["landmark_points_68"]
FACEMESH_bigmask = list_keypoint[streamer]["FACEMESH_bigmask"]

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
def build_model():

    global model #singleton design pattern

    if not "model" in globals():

        model = tf.function(
            retinaface_model.build_model(),
            input_signature=(tf.TensorSpec(shape=[None, None, None, 3], dtype=np.float32),)
        )

    return model

def detect_faces_retinaface_tensor(img_path, threshold=0.9, model = None , allow_upscaling = True, ratio = None):

    img = img_path
    H,W = img.shape[:2]
    padding_size_ratio = ratio
    if model is None:
        model = build_model()
    nms_threshold = 0.4; decay4=0.5

    _feat_stride_fpn = [32, 16, 8]

    _anchors_fpn = {
        'stride32': np.array([[-248., -248.,  263.,  263.], [-120., -120.,  135.,  135.]], dtype=np.float32),
        'stride16': np.array([[-56., -56.,  71.,  71.], [-24., -24.,  39.,  39.]], dtype=np.float32),
        'stride8': np.array([[-8., -8., 23., 23.], [ 0.,  0., 15., 15.]], dtype=np.float32)
    }

    _num_anchors = {'stride32': 2, 'stride16': 2, 'stride8': 2}

    #---------------------------
    proposals_list = []
    scores_list = []
    landmarks_list = []
    im_tensor, im_info, im_scale = preprocess.preprocess_image(img, allow_upscaling)
    net_out = model(im_tensor)
    net_out = [elt.numpy() for elt in net_out]
    sym_idx = 0

    for _idx, s in enumerate(_feat_stride_fpn):
        _key = 'stride%s'%s
        scores = net_out[sym_idx]
        scores = scores[:, :, :, _num_anchors['stride%s'%s]:]

        bbox_deltas = net_out[sym_idx + 1]
        height, width = bbox_deltas.shape[1], bbox_deltas.shape[2]

        A = _num_anchors['stride%s'%s]
        K = height * width
        anchors_fpn = _anchors_fpn['stride%s'%s]
        anchors = postprocess.anchors_plane(height, width, s, anchors_fpn)
        anchors = anchors.reshape((K * A, 4))
        scores = scores.reshape((-1, 1))

        bbox_stds = [1.0, 1.0, 1.0, 1.0]
        bbox_deltas = bbox_deltas
        bbox_pred_len = bbox_deltas.shape[3]//A
        bbox_deltas = bbox_deltas.reshape((-1, bbox_pred_len))
        bbox_deltas[:, 0::4] = bbox_deltas[:,0::4] * bbox_stds[0]
        bbox_deltas[:, 1::4] = bbox_deltas[:,1::4] * bbox_stds[1]
        bbox_deltas[:, 2::4] = bbox_deltas[:,2::4] * bbox_stds[2]
        bbox_deltas[:, 3::4] = bbox_deltas[:,3::4] * bbox_stds[3]
        proposals = postprocess.bbox_pred(anchors, bbox_deltas)

        proposals = postprocess.clip_boxes(proposals, im_info[:2])

        if s==4 and decay4<1.0:
            scores *= decay4

        scores_ravel = scores.ravel()
        order = np.where(scores_ravel>=threshold)[0]
        proposals = proposals[order, :]
        scores = scores[order]

        proposals[:, 0:4] /= im_scale
        proposals_list.append(proposals)
        scores_list.append(scores)

        landmark_deltas = net_out[sym_idx + 2]
        landmark_pred_len = landmark_deltas.shape[3]//A
        landmark_deltas = landmark_deltas.reshape((-1, 5, landmark_pred_len//5))
        landmarks = postprocess.landmark_pred(anchors, landmark_deltas)
        landmarks = landmarks[order, :]

        landmarks[:, :, 0:2] /= im_scale
        landmarks_list.append(landmarks)
        sym_idx += 3

    proposals = np.vstack(proposals_list)
    if proposals.shape[0]==0:
        landmarks = np.zeros( (0,5,2) )
        return np.zeros( (0,5) ), landmarks
    scores = np.vstack(scores_list)
    scores_ravel = scores.ravel()
    order = scores_ravel.argsort()[::-1]

    proposals = proposals[order, :]
    scores = scores[order]
    landmarks = np.vstack(landmarks_list)
    landmarks = landmarks[order].astype(np.float32, copy=False)

    pre_det = np.hstack((proposals[:,0:4], scores)).astype(np.float32, copy=False)

    #nms = cpu_nms_wrapper(nms_threshold)
    #keep = nms(pre_det)
    keep = postprocess.cpu_nms(pre_det, nms_threshold)

    det = np.hstack( (pre_det, proposals[:,4:]) )
    det = det[keep, :]
    landmarks = landmarks[keep]
    l_coordinate = []
    dets = []

    resp = {}
    for idx, face in enumerate(det):

        label = 'face_'+str(idx+1)
        resp[label] = {}
        resp[label]["score"] = face[4]

        resp[label]["facial_area"] = list(face[0:4].astype(int))
        if idx ==0:
            dets = list(face[0:4].astype(int))
        resp[label]["landmarks"] = {}
        resp[label]["landmarks"]["right_eye"] = list(landmarks[idx][0])
        resp[label]["landmarks"]["left_eye"] = list(landmarks[idx][1])
        resp[label]["landmarks"]["nose"] = list(landmarks[idx][2])
        resp[label]["landmarks"]["mouth_right"] = list(landmarks[idx][3])
        resp[label]["landmarks"]["mouth_left"] = list(landmarks[idx][4])
    if len(dets)==0:
        check_detect = 0
    else:
        check_detect = 1
        for k in range(len(dets)):
            xmin = int(dets[0])
            ymin = int(dets[1])
            xmax = int(dets[2])
            ymax = int(dets[3])
            bbox = ((xmin, ymin , xmax, ymax))
            topleft = (int(bbox[0]), int(bbox[1]))
            bottomright = (int(bbox[2]), int(bbox[3]))
            padding_X = int((bottomright[0] - topleft[0]) * padding_size_ratio)
            padding_Y = int((bottomright[1] - topleft[1]) * padding_size_ratio)
            padding_topleft = (max(0, topleft[0] - padding_X), max(0, topleft[1]- padding_Y))
            padding_bottomright = (min(W, bottomright[0] + padding_X), min(H, bottomright[1] + padding_Y))
            coordinate = (padding_topleft, padding_bottomright)
            l_coordinate.append(coordinate)

    return l_coordinate, check_detect

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
        l_coordinate, detected_face = detection_face(mobile_net,resnet_net, inimg, device,padding_ratio[r_idx])
        if not detected_face and r_idx == len(padding_ratio)-1:
            count_loss_detect = 1
            detect_fail = True
            # print("New_model loss")
            return None, None, None,None,detect_fail,count_loss_detect,previous_model,None
        elif not detected_face and r_idx < len(padding_ratio)-1:
          previous_model =  1
          # # print("New_model_used")
          # l_coordinate, check_fail = detect_faces_retinaface_tensor(inimg, ratio = padding_ratio[r_idx])
          # if check_fail and r_idx == len(padding_ratio)-1:
          #     count_loss_detect = count_loss_detect+ 1
          #     detect_fail = True
          #     # print("New_model loss")
          #     return None, None, None,None,detect_fail,count_loss_detect,previous_model,None
          # elif check_fail and r_idx < len(padding_ratio)-1:
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
            curbox = []
            curbox.append(topleft[0])
            curbox.append(topleft[1])
            curbox.append(bottomright[0])
            curbox.append(bottomright[1])
            bbox = curbox


        if not face_landmarks and r_idx == len(padding_ratio)-1:
            # print("Medipipe loss")
            return None, None, None, crop_images_coors, detect_fail,count_loss_detect,previous_model,None
        elif not face_landmarks and r_idx < len(padding_ratio)-1 :
            # print("Medipipe skip")
            continue
        # print('bbox ', bbox)
        # print("Medipipe detected")
        bbox_w = bbox[2] - bbox[0]
        bbox_h = bbox[3] - bbox[1]

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

        listpoint2 = []
        for i in range(len(FACEMESH_lips_2)):
            idx = FACEMESH_lips_2[i]
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
        for i in range(len(landmark_points_68)):
            idx = landmark_points_68[i]
            x = face_landmarks.landmark[idx].x
            y = face_landmarks.landmark[idx].y

            realx = x * bbox_w + bbox[0]
            realy = y * bbox_h + bbox[1]
            listpoint3.append((realx, realy))

        # print(listpoint3.simplices)
        srcpts3 = np.array(listpoint3, np.int32)
        posePoint = []
        for i in range(len(FACEMESH_pose_estimation)):
            idx = FACEMESH_pose_estimation[i]
            x = face_landmarks.landmark[idx].x
            y = face_landmarks.landmark[idx].y

            realx = x * bbox_w + bbox[0]
            realy = y * bbox_h + bbox[1]
            posePoint.append((realx, realy))

        yaw, pitch, roll = facePose(posePoint[0], posePoint[1], posePoint[2], posePoint[3], posePoint[4])
        if yaw is None:
            face_angle = 0
        if yaw > 50:
            face_angle = 1
        elif yaw < -50:
            face_angle = 2
        else:
            face_angle = 0
        return srcpts, srcpts2, srcpts3, crop_images_coors, detect_fail,count_loss_detect,previous_model,face_angle


def get_face(facea, inimg, kf=None, kfMount=None, iswide = False):
    listpoint = []
    h,w = inimg.shape[:2]
    # print(inimg.shape[:2])
    results = facea.process(cv2.cvtColor(inimg, cv2.COLOR_BGR2RGB))
    if not results.multi_face_landmarks:
        return None, None

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

    listpoint2 = []
    for i in range(len(FACEMESH_lips_2)):
        idx = FACEMESH_lips_2[i]
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
    if kfMount is not None:
        if kfMount.noneArray():
            kfMount.setpoints(listpoint2, w, h)
        else:
            listpoint2 = kfMount.getpoints(listpoint2)

    srcpts2 = np.array(listpoint2, np.int32)
    # srcpts = np.concatenate((srcpts, srcpts2), axis=0)
    srcpts2 = srcpts2.reshape(-1,1,2)


    return srcpts, srcpts2

def getKeypointByMediapipe(face_mesh_wide, videoimg,face_mesh_256, faceimg, kfVF, kfVM, kfFF, kfFM, mobile_net,resnet_net,device):
    videopts, videopts_out, videopts_big, video_crops_coors,detect_fail,count_loss_detect_Retina,previous_model,face_angle = get_face_by_RetinaFace(face_mesh_wide, videoimg, mobile_net, resnet_net, device, kfVF, kfVM)

    if videopts is None:
        return None, None, None, None, None,video_crops_coors,detect_fail,None
    facepts, facepts_out = get_face(face_mesh_256, faceimg, kfFF, None)
    return videopts, videopts_out, videopts_big, facepts, facepts_out, video_crops_coors,detect_fail,face_angle

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

def warpTriangle(img1, img2, t1, t2):
    # Find bounding rectangle for each triangle
    r1 = cv2.boundingRect(np.float32(t1))
    r2 = cv2.boundingRect(np.float32(t2))

    # Offset points by left top corner of the respective rectangles
    t1Rect = []
    t2Rect = []
    t2RectInt = []

    for i in range(0, 3):
        # print(t1[i][0] - r1[0])
        # print(i, ' ' , t1[i][0][1] )
        # print(r1[1])
        t1Rect.append(((t1[i][0][0] - r1[0]), (t1[i][0][1] - r1[1])))
        t2Rect.append(((t2[i][0][0] - r2[0]), (t2[i][0][1] - r2[1])))
        t2RectInt.append(((t2[i][0][0] - r2[0]), (t2[i][0][1] - r2[1])))

    # Get mask by filling triangle
    mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2RectInt), (1.0, 1.0, 1.0), 16, 0)

    # Apply warpImage to small rectangular patches
    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    size = (r2[2], r2[3])
    img2Rect = applyAffineTransform(img1Rect, t1Rect, t2Rect, size)
    img2Rect = img2Rect * mask

    # Copy triangular region of the rectangular patch to the output image
    img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] * ((1., 1., 1.) - mask)
    img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] + img2Rect

def find_ratio_intersection(videoimg,crops_coors,videopts_out,videopts_big,FaceOcc_only = None,img_video_mask_mount=None):
    crop_image = videoimg[crops_coors[0]:crops_coors[1],crops_coors[2]:crops_coors[3],:]
    crop_image_occ = np.copy(crop_image)
    video_h,video_w, = videoimg.shape[:2]
    crop_image_occ = cv2.resize(crop_image_occ,(256,256)) # (256,256) is input size of FaceOcc

    #Predict mask by FaceOcc
    data = to_tensor(crop_image_occ).unsqueeze(0)
    data = data.to(DEVICE)
    with torch.no_grad():
        pred = model_Occ(data)
    pred_mask = (pred > 0).type(torch.int32)
    pred_mask = pred_mask.squeeze().cpu().numpy()
    mask_occ = cv2.resize(pred_mask,(crop_image.shape[1],crop_image.shape[0]),interpolation=cv2.INTER_LINEAR_EXACT) #Resize up need INTER_LINEAR_EXACT

    img_mouth_mask_occ = np.zeros((video_h,video_w), np.uint8)
    img_mouth_mask_occ[crops_coors[0]:crops_coors[1],crops_coors[2]:crops_coors[3]] = mask_occ
    #Check output condition
    if FaceOcc_only:
        img_mouth_mask_occ = fillhole(img_mouth_mask_occ)
        img_bgr_uint8_occ = normalize_channels(img_mouth_mask_occ, 3)
        img_bgr_occ = img_bgr_uint8_occ.astype(np.int32)*255
        img_bgr_occ[img_bgr_occ>0] =255
        img_bgr_occ[:,:,0] = 0
        img_bgr_occ[:,:,2] = 0
        alpha_0 = 0.6
        cropimage = cv2.addWeighted(img_bgr_occ, alpha_0,videoimg, 1-alpha_0, 0, dtype = cv2.CV_32F)
        return cropimage
    else:
        var_middle_occ = np.copy(img_video_mask_mount[:,:])
        var_middle_occ[var_middle_occ ==255]=1
        face_Occ = img_mouth_mask_occ.copy()
        face_Occ = fillhole(face_Occ)
        img_mouth_mask_occ = img_mouth_mask_occ*var_middle_occ*255
        img_mouth_mask_occ = fillhole(img_mouth_mask_occ)
        #cal ratio between 2 mouth mask Mediapipe and FaceOcc
        img_video_mask_mount = np.atleast_3d(img_video_mask_mount).astype(np.float) / 255.
        img_video_mask_mount[img_video_mask_mount != 1] = 0

        img_mouth_mask_occ = np.atleast_3d(img_mouth_mask_occ).astype(np.float) / 255.
        img_mouth_mask_occ[img_mouth_mask_occ != 1] = 0
        newval = len(img_mouth_mask_occ[img_mouth_mask_occ > 0])/len(img_video_mask_mount[img_video_mask_mount >0])
        img_bgr_uint8_occ = normalize_channels(face_Occ, 3)
        img_bgr_occ = img_bgr_uint8_occ.astype(np.int32)*255
        img_bgr_occ[img_bgr_occ>0] =255
        img_bgr_occ[:,:,0] = 0
        img_bgr_occ[:,:,2] = 0

        alpha_0 = 0.6
        cropimage = cv2.addWeighted(img_bgr_occ, alpha_0,videoimg, 1-alpha_0, 0, dtype = cv2.CV_32F)
        cv2.polylines(cropimage, [videopts_out],True,color=(0,0,255),thickness=1)
        cv2.polylines(cropimage, [videopts_big],True,color=(255,0,255),thickness=1)
        cv2.putText(cropimage, text='Inter ratio'+str(newval), org=(100, 100), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1.1, color=(0, 255, 0),thickness=2)
        return newval,cropimage

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
# def gen_fomm_final_face(videoimg,faceimg,videopts,videopts_out,facepts,facepts_out):
#
#     return output_main
def add_image_by_mask(img1, img2, mask_):
    mask_not = cv2.bitwise_not(mask_)
    img2_no_mask = cv2.bitwise_and(img2, img2, mask=mask_not)
    img1_mask_only = cv2.bitwise_and(img1, img1, mask=mask_)
    return cv2.add(img2_no_mask, img1_mask_only)

font = cv2.FONT_HERSHEY_SIMPLEX

if __name__ == "__main__":
    #
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
    #
    # # model = nn.DataParallel(model.to(DEVICE), device_ids=[0, 1])
    weights = torch.load('/home/ubuntu/Duy_test_folder/FaceExtraction/FaceOcc/FaceOcc/epoch_16_best.ckpt')
    new_weights = OrderedDict()
    for key in weights.keys():
        new_key = '.'.join(key.split('.')[1:])
        new_weights[new_key] = weights[key]

    model_Occ.load_state_dict(new_weights)
    model_Occ.to(DEVICE)
    model_Occ.eval()
    # FRAME_PATH = 'in.mp4'
    FACE_PATH = '/home/ubuntu/Duy_test_folder/Retinaface_Mediapipe/Dr_video/3sourceheadMovCSV_3Aug_Dr_ES_newdriving_newlogic_gfpgan2.mp4'
    # FACE_PATH = '/home/ubuntu/first-order-model/3sourceheadMovCSV_26July_Tim_test1.mp4'
    FRAME_PATH = '/home/ubuntu/Duy_test_folder/Retinaface_Mediapipe/Dr_video/DrDisrespect-Falls-in-Love-with-Warzone-again-thanks-to-new-Game-Mode-30FPS.mp4'
    # FRAME_PATH = '/home/ubuntu/Duy_test_folder/Retinaface_Mediapipe/Tim_video/Tim_source_2.mp4'
    mobile_net, resnet_net = loadmodelface()
    device = torch.device("cuda")

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh_256 = mp_face_mesh.FaceMesh(min_detection_confidence=0.2, min_tracking_confidence=0.2,max_num_faces=1)
    face_mesh_wide = mp_face_mesh.FaceMesh(min_detection_confidence=0.5)

    facecout = 0
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))

    capFrame = cv2.VideoCapture(FRAME_PATH)
    capFace = cv2.VideoCapture(FACE_PATH)
    fps = capFrame.get(cv2.CAP_PROP_FPS)
    width_  = capFrame.get(cv2.CAP_PROP_FRAME_WIDTH)
    height_ = capFrame.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(fps)
    # encoder = ffmpeg_encoder('vidout.mp4', fps, int(width_), int(height_))
    # encoder = ffmpeg_encoder('vidout256.mp4', fps, 256, 256)
    encoder_video = ffmpeg_encoder('Dr_video/3Aug_Dr_testmask_logicmask_8p23.mp4', fps, int(width_), int(height_))
    # encoder_mask = ffmpeg_encoder('1July_DR_PT_test_left_blockratio0301_addweighted_visualize_newmask_v4.mp4', fps,int(width_), int(height_))
    #cropped_image_folder_path = '/content/gdrive/MyDrive/Deep_fake/video_test_mediapipe_ratio_0_dot_45'

    totalF = int(capFrame.get(cv2.CAP_PROP_FRAME_COUNT))
    # pbar = tqdm(total=totalF)
    print("Length input video: ", int(capFrame.get(cv2.CAP_PROP_FRAME_COUNT)))
    print("Length 3d video: ",int(capFace.get(cv2.CAP_PROP_FRAME_COUNT)))
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

    List_speak_frame = []
    json_path = '/home/ubuntu/Duy_test_folder/SP_Speechdetection/content/SpeakerDetection/Speech_detected_Dr2206_ES_full_voiceonly.json'
    with open(json_path,'r') as f:
        list_inital = json.load(f)
        List_speak_frame = []#list_inital['data'][:]
        for j in range(len(list_inital['data'])):
            start = int(float(list_inital['data'][j]['start'])*fps)-3
            stop = int(float(list_inital['data'][j]['end'])*fps)
            print(start, stop)
            for i in range(start,stop+1):
                List_speak_frame.append(i)
    List_speak_frame = np.unique(List_speak_frame)        
    #dynamic variable
    count_frame = 0
    # limit_frame = range(12860, 12900+1)
    num_frame_loss = 0
    num_mouth_frame_loss = 0
    block_frames_count = 0
    original_frames = []
    drawed_frames = []
    select_block = []
    drawed_frames_visualize = []

    #const variable
    fps_per_second = 10
    num_frame_threshold = 1
    minute_start = 8
    second_start = 23
    minute_stop = 8
    second_stop = 32
    frame_start = int(minute_start*60*fps+second_start*fps)
    frame_stop = int(minute_stop*60*fps+second_stop*fps)
    total_f = frame_stop-frame_start
    pbar = tqdm(total=totalF)

    while capFrame.isOpened():
        #cal frame for block_frames util reach num_frame_threshold
        count_frame = count_frame+1
        okay1  , videoimg = capFrame.read()
        okay2 , faceimg = capFace.read()
        videoimg_copy = np.copy(videoimg)
        pbar.update(1)
        if count_frame <= frame_start:
            continue
        elif count_frame > frame_stop+1:
            break
        elif count_frame > frame_start and count_frame <= frame_stop+1:
            # if block_frames_count == fps_per_second:
            #     # if num_frame_loss <= num_frame_threshold and num_mouth_frame_loss <= num_frame_threshold:
            #     #     select_block = drawed_frames
            #     # else:
            #     #     select_block = original_frames
            #
            #     for i in range(block_frames_count):
            #         output = select_block[i]
            #         image_draw = cv2.cvtColor(output,cv2.COLOR_RGB2BGR)
            #         imageout = Image.fromarray(np.uint8(image_draw))
            #         encoder_video.stdin.write(imageout.tobytes())
            #
            #         # output_visualize = drawed_frames_visualize[i]
            #         # image_draw_visualize= green_blue_swap(output_visualize)
            #         # image_draw_ = cv2.cvtColor(image_draw_visualize,cv2.COLOR_RGB2BGR)
            #         # imageout_visalize = Image.fro/home/ubuntu/Duy_test_folder/Retinaface_Mediapipe/Tim_videomarray(np.uint8(image_draw_visualize))
            #         # encoder_mask.stdin.write(imageout_visalize.tobytes())
            #     #reset dynamic val
            #     num_frame_loss = 0
            #     original_frames = []
            #     select_block = []
            #     drawed_frames = []
            #     drawed_frames_visualize = []
            #     block_frames_count = 0
            #     num_mouth_frame_loss = 0
            if not count_frame in List_speak_frame:
                # print("Not speak")
                # image_draw_visualize= green_blue_swap(videoimg_copy)
                # image_draw_ = cv2.cvtColor(image_draw_visualize,cv2.COLOR_RGB2BGR)
                encoder_video.stdin.write(videoimg_copy.tobytes())
                continue
            else:
                # block_frames_count = block_frames_count + 1

                if not okay1 or not okay2:
                    print('Cant read the video , Exit!')
                    break

                video_h,video_w, = videoimg.shape[:2]

                videopts, videopts_out, videopts_big, facepts, facepts_out, crops_coors,detect_fail,face_angle= getKeypointByMediapipe(\
                face_mesh_wide, videoimg, face_mesh_256, faceimg, trackkpVideoFace, trackkpVideoMount, trackkpFaceFace, trackkpFaceMount, mobile_net,resnet_net,device)
                if videopts is None or facepts is None :
                    if detect_fail is True or crops_coors is None:
                        encoder_video.stdin.write(videoimg_copy.tobytes())
                    # else:
                    #     cropimage = find_ratio_intersection(videoimg,crops_coors,videopts,FaceOcc_only = True,img_video_mask_mount=None)
                    #     image_draw_visualize= green_blue_swap(cropimage)
                    #     image_draw_ = cv2.cvtColor(image_draw_visualize,cv2.COLOR_RGB2BGR)
                    #     encoder_video.stdin.write(image_draw_.tobytes())
                    print("Mediapipe not detected - ", count_frame)
                    # num_frame_loss +=1
                    # original_frames.append(videoimg_copy)
                    # drawed_frames.append(videoimg_copy)
                    # drawed_frames_visualize.append(cropimage)

                    continue
                else:
                    try:
                        #get mask of mediapipe
                        # M_face, _ = cv2.findHomography(facepts, videopts)
                        # face_img = cv2.warpPerspective(faceimg,M_face,(video_w, video_h))
                        #
                        # facepts_out = cv2.perspectiveTransform(facepts_out.astype(np.float32), M_face).astype(np.int32)
                        # M_mount, _ = cv2.findHomography(facepts_out, videopts_out)
                        # face_img = cv2.warpPerspective(face_img,M_mount,(video_w, video_h), cv2.INTER_LINEAR)

                        img_video_mask_mount = np.zeros((video_h,video_w), np.uint8)
                        cv2.fillPoly(img_video_mask_mount, pts =[videopts_out], color=(255,255,255))
                        img_video_mask_mount_bk = np.zeros((video_h,video_w), np.uint8)
                        cv2.fillPoly(img_video_mask_mount_bk, pts =[videopts_out], color=(255,255,255))

                        img_video_mask_face = np.zeros((video_h,video_w), np.uint8)
                        cv2.fillPoly(img_video_mask_face, pts =[videopts[:-13]], color=(255,255,255))
                        img_video_mask_face = cv2.bitwise_or(img_video_mask_face, img_video_mask_mount)
                        topy, topx, bottomy, bottomx, center_face = mask2box(img_video_mask_face)

                        # result = add_image_by_mask(face_img, videoimg, img_video_mask_face)
                        # # videoimg = cv2.seamlessClone(result, videoimg, img_video_mask_mount, center_mount, cv2.MIXED_CLONE)
                        # #Use addWeighted to combine results box and videoimg box
                        #
                        # alpha_0 = 0.95
                        # result[topy:bottomy, topx:bottomx] = cv2.addWeighted(result[topy:bottomy, topx:bottomx],\
                        #  alpha_0, videoimg[topy:bottomy, topx:bottomx], 1-alpha_0, 0.0)
                        #
                        # img_bgr_uint8_1 = normalize_channels(result[topy:bottomy, topx:bottomx], 3)
                        # img_bgr_1 = img_bgr_uint8_1.astype(np.float32) / 255.0
                        # img_bgr_1 = np.clip(img_bgr_1, 0, 1)
                        # img_bgr_uint8_2 = normalize_channels(videoimg[topy:bottomy, topx:bottomx], 3)
                        # img_bgr_2 = img_bgr_uint8_2.astype(np.float32) / 255.0
                        # img_bgr_2 = np.clip(img_bgr_2, 0, 1)
                        #
                        # result_new = linear_color_transfer(img_bgr_1, img_bgr_2)
                        # final_img = color_hist_match(result_new, img_bgr_2, 255).astype(dtype=np.float32)
                        #
                        # result[topy:bottomy, topx:bottomx] = blursharpen((final_img*255).astype(np.uint8), 1, 5, 0.5)
                        # # addWeighted to results after match color
                        # alpha_0 = 0.95
                        # result[topy:bottomy, topx:bottomx] = cv2.addWeighted(result[topy:bottomy, topx:bottomx],\
                        #  alpha_0, videoimg[topy:bottomy, topx:bottomx], 1-alpha_0, 0.0)
                        #
                        # img_video_mask_mount = cv2.dilate(img_video_mask_mount, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)), iterations = 1 )
                        # img_video_mask_face = cv2.erode(img_video_mask_face, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)), iterations = 1 )
                        # img_video_mask_mount_error = cv2.bitwise_xor(img_video_mask_mount, img_video_mask_face)
                        # img_video_mask_mount = img_video_mask_mount - img_video_mask_mount_error
                        #
                        # result = add_image_by_mask(result, videoimg, img_video_mask_mount)
                        # m_topy, m_topx, m_bottomy, m_bottomx, center_mount = mask2box(img_video_mask_mount)
                        #
                        #
                        # output_main = cv2.seamlessClone(result, videoimg, img_video_mask_mount, center_mount, cv2.NORMAL_CLONE)
                        # cv2.rectangle(output_main, (m_topx, m_topy), (m_bottomx,m_bottomy), (0,255,0), 4)
                        # if m_topy <2 or m_topx <2 or m_bottomy > int(height_)-2 or m_bottomx > int(width_)-2 :
                        #     drawed_frames.append(videoimg_copy)
                        #     original_frames.append(videoimg_copy)
                        #
                        #     # drawed_frames_visualize.append(videoimg_copy)
                        #     continue
                        #Cal mask by FaceOcc
                        newval, cropimage = find_ratio_intersection(videoimg,crops_coors,videopts_out,videopts_big,\
                        FaceOcc_only = False,img_video_mask_mount = img_video_mask_mount_bk)
                        # cv2.polylines(cropimage, [videopts_out],True,color=(0,0,255),thickness=   2)
                        # image_draw_visualize= green_blue_swap(cropimage)
                        if not cropimage is None:
                            # path = "Dr_video/faceocc_test/img_"+str(count_frame)+".png"
                            # cv2.imwrite(path,cropimage)
                            image_draw_ = cv2.cvtColor(cropimage,cv2.COLOR_RGB2BGR)
                            images = Image.fromarray(np.uint8(image_draw_))
                            encoder_video.stdin.write(images.tobytes())
                            # break
                        # # print("Face angle ", face_angle)
                        # if newval > 0.9:
                        #     # print("Mask good!")
                        #     drawed_frames.append(output_main)
                        #     original_frames.append(videoimg_copy)
                        # elif newval > 0.8 and face_angle != 0:
                        #     # print("Face angle!")
                        #     drawed_frames.append(output_main)
                        #     original_frames.append(videoimg_copy)
                        # else:
                        #     num_mouth_frame_loss += 1
                        #     drawed_frames.append(videoimg_copy)
                        #     original_frames.append(videoimg_copy)

                        # drawed_frames.append(output_main)
                        # original_frames.append(videoimg_copy)

                    except Exception:
                        print("Try fail", count_frame)
                        # drawed_frames.append(videoimg_copy)
                        # original_frames.append(videoimg_copy)
                        # image_draw_visualize= videoimg_copy
                        # image_draw_ = cv2.cvtColor(image_draw_visualize,cv2.COLOR_RGB2BGR)
                        encoder_video.stdin.write(videoimg_copy.tobytes())
                        continue

    pbar.close()
    encoder_video.stdin.flush()
    encoder_video.stdin.close()
    # encoder_mask.stdin.flush()
    # encoder_mask.stdin.close()
    print("=================\n","DONE!\n","=================")
