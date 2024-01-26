"""
"""
import sys
import json
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)
import audio
import numpy as np
import torch
from _wav2lip.models import Wav2Lip
from _wav2lip.Crop_aligned_face import crop_aligned_face,merge_wav2lip
import cv2
from tqdm import tqdm
import math
import imutils
import mediapipe as mp
import ffmpeg
import subprocess
from PIL import Image
import warnings
from pipeline_mobile_resnet_wav2lip import loadmodelface, detection_face_wav2lips ,args
from torchvision import transforms as TF_s
from collections import OrderedDict
import segmentation_models_pytorch as smp
from color_transfer import color_transfer, color_hist_match,linear_color_transfer
from common import  normalize_channels
warnings.filterwarnings("ignore")
import time
# from CodeFormer.load_codeformer import process_img,load_codeformer_model
import argparse
from datetime import datetime
from signal import signal, SIGPIPE, SIG_DFL
signal(SIGPIPE, SIG_DFL)
import multiprocessing
from multiprocessing import Process, Manager
import string
import random
import uuid
from urllib.parse import urlsplit
import requests
import shutil
import boto3
import logging
import threading

PATH = os.path.dirname(__file__)
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# Replace boomtv access key here
s3 = boto3.client('s3',
                  aws_access_key_id="",
                  aws_secret_access_key="/")
# Replace StarPower team api key here
api_key = ""

endpoints = {
    'wav2lip_model': {
            "url": "https://api.runpod.ai/v2/rtc6xf7jldz2xl/run",
            "id": "rtc6xf7jldz2xl"
    }
}

def generate_random_filename(length=10, extension=''):
    characters = string.ascii_letters + string.digits
    random_name = ''.join(random.choice(characters) for _ in range(length))
    unique_id = str(uuid.uuid4().hex)
    return random_name + '_' + unique_id + extension

def get_bucket_name(s3_url):
    # Parse the URL
    url_parts = urlsplit(s3_url)
    # Check if the URL is an S3 URL
    if url_parts.scheme == "https" and url_parts.netloc.endswith("amazonaws.com"):
        bucket_name = url_parts.netloc.split('.')[0]
        return bucket_name
    else:
        raise ValueError("Not a valid S3 URL")

def download_asset(tmp_folder,url_link):
    bucket_name = get_bucket_name(url_link)
    fileDirName = url_link.split("com/")[1]
    file_path = f"/tmp/{tmp_folder}/" + url_link.split("/")[-1]
    try :
        print("Download asset from ",url_link," to ",file_path )
        s3.download_file(bucket_name,fileDirName,file_path)
    except Exception as e:
        print("Error: ",e)
    return file_path

def upload_video(file_path):
    tmp_img_fn = generate_random_filename(extension=os.path.splitext(file_path)[-1])
    print("S3 filename: ",tmp_img_fn)
    shutil.copyfile(file_path, tmp_img_fn)
    bucket_name = 'vi-lang-pipeline-public'
    object_key = f'runpod/output/wav2lip_model/{tmp_img_fn}'
    s3.upload_file(
        tmp_img_fn, bucket_name, object_key
    )
    os.remove(tmp_img_fn)
    file_url = f"https://vi-lang-pipeline-public.s3.us-west-2.amazonaws.com/{object_key}"
    # s3.generate_presigned_url(
    #     'get_object',
    #     Params={'Bucket': bucket_name, 'Key': object_key}
    #     # ExpiresIn=3600  # URL validity duration in seconds (1 hour in this example)
    # )
    return file_url

PATH = os.path.dirname(__file__)
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
five_points_template = np.array([(192.98138, 239.94708), (318.90277, 240.1936), (256.63416, 314.01935),
                               (201.26117, 371.41043), (313.08905, 371.15118)])

def warpaffine_face_wav2lip(input_img,landmarks_5,face_size,border_mode):
    # face_template = np.array([[192.98138, 239.94708], [318.90277, 240.1936], [256.63416, 314.01935],
    #                                [201.26117, 371.41043], [313.08905, 371.15118]])
    face_template = np.array([[141, 145], [266, 143], [200, 216],
                                  [153, 303], [251, 303]], np.float32) ## default size = 400
    face_template = face_template+[float(face_size[0]-400)/2,float(face_size[1]-400)/2]
    landmark = landmarks_5
    landmark = np.vstack((landmark[:2],landmark[3:]))
    face_template = np.vstack((face_template[:2],face_template[3:]))
    # print("landmarks_5:",landmarks_5)
    # print("landmarks_5:",face_template)
        # print("Lanfmark",landmark)

        # use 5 landmarks to get affine matrix
        # use cv2.LMEDS method for the equivalence to skimage transform
        # ref: https://blog.csdn.net/yichxi/article/details/115827338
    affine_matrix = cv2.estimateAffinePartial2D(landmark, face_template, method=cv2.LMEDS)[0]

    # skin_img = cv2.imread("/home/ubuntu/Duy_test_folder/Retinaface_Mediapipe/Dr_skin.png",0)
    # skin_img = my_cv2_resize(skin_img,face_size[0],face_size[1])
    # warp and crop faces
    if border_mode == 'constant':
        border_mode = cv2.BORDER_CONSTANT
    elif border_mode == 'reflect101':
        border_mode = cv2.BORDER_REFLECT101
    elif border_mode == 'reflect':
        border_mode = cv2.BORDER_REFLECT
    elif border_mode == 'transparent':
        border_mode = cv2.BORDER_REPLICATE
    cropped_face = cv2.warpAffine(input_img, affine_matrix,face_size,flags=cv2.INTER_NEAREST, borderMode=border_mode,borderValue=0)
    return cropped_face,affine_matrix
 
def warpPerspective_face_wav2lip(input_img,landmarks_5,face_size,border_mode):
    # face_template = np.array([[192.98138, 239.94708], [318.90277, 240.1936], [256.63416, 314.01935],
    #                                [201.26117, 371.41043], [313.08905, 371.15118]])
    face_template_perspective = np.array([(141, 145), (266, 143),
                                  (153, 303), (251, 303)], np.float32)*(face_size/400)  ## default size = 400
    face_template = np.array([(141, 145), (266, 143), (200, 216),
                                  (153, 303), (251, 303)], np.float32)*(face_size/400)
    face_template = np.vstack((face_template[:2],face_template[3:]))
    landmark = landmarks_5
    landmark = np.vstack((landmark[:2],landmark[3:]))
    # print("landmarks_5:",landmark)
    # print("landmarks_5:",face_template)
        # print("Lanfmark",landmark)

        # use 5 landmarks to get affine matrix
        # use cv2.LMEDS method for the equivalence to skimage transform
        # ref: https://blog.csdn.net/yichxi/article/details/115827338
    affine_matrix= cv2.findHomography(landmark.reshape(-1,1,2), face_template.reshape(-1,1,2))
    # warp and crop faces

    cropped_face = cv2.warpAffine(input_img, affine_matrix[0][:2],(face_size,face_size),borderMode=cv2.BORDER_TRANSPARENT)
    return cropped_face,affine_matrix
 

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



class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def get_mel_chunks(wav_path,fps_s):
    """
    each mel chunk is corresponding with one frame
    """
    mel_step_size = 16
    fps = fps_s
    wav = audio.load_wav(wav_path, 16000)
    mel = audio.melspectrogram(wav)
    if np.isnan(mel.reshape(-1)).sum() > 0:
        raise ValueError('Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')
    mel_chunks = []
    mel_idx_multiplier = 80./fps
    i = 0
    while 1:
        start_idx = int(i * mel_idx_multiplier)
        if start_idx + mel_step_size > len(mel[0]):
            mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
            break
        mel_chunks.append(mel[:, start_idx : start_idx + mel_step_size])
        i += 1

    return mel_chunks

def load_wav2lip_model(path, device):
    # checkpoint
    if device == 'cuda':
        checkpoint = torch.load(path)
    else:
        checkpoint = torch.load(path,
                                map_location=lambda storage, loc: storage)
    # model
    model = Wav2Lip()
    # print("Load checkpoint from: {}".format(path))
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s)

    model = model.to(device)
    return model.eval()

def my_cv2_resize(img, w, h):
    if img.size > w*h:
        return cv2.resize(img, (w, h), cv2.INTER_AREA)
    else:
        return cv2.resize(img, (w, h), cv2.INTER_CUBIC)

def gen_lipsync_img(img, mel_chunk, model, box, device):
    """
    """
    resize_factor = 1
    # box = face_detect([frame], sfd_facedetector)[0]
    # print(img.shape[1]//resize_factor, img.shape[0]//resize_factor)
    # img = cv2.resize(img, (int(img.shape[1]//resize_factor), int(img.shape[0]//resize_factor)))
    # frame = cv2.resize(frame, (1280,720))
    MODEL_INPUT_IMG_SIZE = 96
    # print(box)
    y_ori1, y_ori2, x_ori1, x_ori2 = box
    y1, y2, x1, x2 = (np.array(box)//resize_factor).astype(np.int32)

    roi = img[y1:y2, x1:x2, :]
    # roi = cv2.GaussianBlur(roi,(7,7),0)

    # roi = cv2.resize(roi, (int(roi.shape[1]//1.5), int(roi.shape[0]//1.5)))
    # roi = my_cv2_resize(roi, (x_ori2-x_ori1), (y_ori2-y_ori1))

    roi = my_cv2_resize(roi, MODEL_INPUT_IMG_SIZE, MODEL_INPUT_IMG_SIZE)

    img_batch, mel_batch = np.asarray([roi]), np.asarray([mel_chunk])
    img_masked = img_batch.copy()
    img_masked[:, MODEL_INPUT_IMG_SIZE//2:] = 0

    img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
    mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

    img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
    mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)
    with torch.no_grad():
        pred = model(mel_batch, img_batch)

    pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.
    pred = pred.astype(np.uint8)[0]
    pred = my_cv2_resize(pred, (x_ori2-x_ori1), (y_ori2-y_ori1))
    return pred

def init_face_mesh():
    mp_face_mesh = mp.solutions.face_mesh
    return mp_face_mesh.FaceMesh(min_detection_confidence=0.5)
    #static_image_mode=True, max_num_faces=1, refine_landmarks=True,

def get_fm_landmark(img, facemesh):
    h, w = img.shape[:2]
    results = facemesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if not results.multi_face_landmarks:
        return None,None
    else:
        face_landmarks = results.multi_face_landmarks[0]
        points = results.multi_face_landmarks[0].landmark
        points = [[point.x * w, point.y * h] for point in points]
    return points,face_landmarks

def get_fm_mask(img, kp_ids, landmark):
    """
    """
    # get mask
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    kps = [landmark[id] for id in kp_ids]
    kps = np.array(kps, dtype=np.int32)
    cv2.fillPoly(mask, [kps], 255)

    # get center point
    x, y = np.where(mask > 0)
    top = min(y)
    bot = max(y)
    left = min(x)
    right = max(x)
    center =  ((top+bot)//2, (left+right)//2)

    return mask, center

def get_box_by_RetinaFace(inimg, mobile_net_wav2lip, resnet_net_wav2lip, device):

    h,w = inimg.shape[:2]
    padding_ratio = [0]
    padding_size_ratio = 0.1
    for r_idx in range(len(padding_ratio)):
        l_coordinate, detected_face,five_points= detection_face_wav2lips(mobile_net_wav2lip,resnet_net_wav2lip, inimg, device,padding_ratio[r_idx])
        if not detected_face and r_idx == len(padding_ratio)-1:
            return None,None,None,None
        elif not detected_face and r_idx < len(padding_ratio)-1:
            continue
        bbox = None
        for i in range(len(l_coordinate)):
            topleft, bottomright = l_coordinate[i]
            curbox = []
            if bottomright[1] < h/3:
                continue
            h_w_dis = int((bottomright[1]-topleft[1]) - (bottomright[0]-topleft[0]))//2
            topleft = (topleft[0],max(0, topleft[1]+h_w_dis))
            curbox.append(topleft[0])
            curbox.append(topleft[1])
            curbox.append(bottomright[0])
            curbox.append(bottomright[1])

        ##Expand all box
            padding_X = int((bottomright[0] - topleft[0]) * padding_size_ratio)
            padding_Y = int((bottomright[1] - topleft[1]) * padding_size_ratio)
            padding_topleft = (max(0, topleft[0] - padding_X), max(0, topleft[1]- padding_Y))
            padding_bottomright = (min(w, bottomright[0] + padding_X), min(h, bottomright[1] + padding_Y))

            # #Expand bottom only_center_face
            # padding_Y = int((bottomright[1] - topleft[1]) * padding_size_ratio)
            # padding_topleft = (max(0, topleft[0]), max(0, topleft[1]))
            # padding_bottomright = (min(W, bottomright[0]), min(H, bottomright[1] + padding_Y))
            coordinate = (padding_topleft, padding_bottomright)

            # x1, y1, x2, y2 = curbox

        if len(curbox) > 0:
            landmarks_5 = np.array([[five_points[0],five_points[1]],[five_points[2],five_points[3]],\
            [five_points[4],five_points[5]],[five_points[6],five_points[7]],[five_points[8],five_points[9]]])

            landmarks_5_tuple = np.array([(five_points[0],five_points[1]),(five_points[2],five_points[3]),\
            (five_points[4],five_points[5]),(five_points[6],five_points[7]),(five_points[8],five_points[9])])
            return curbox,landmarks_5,coordinate,landmarks_5_tuple
        else:
            return None,None,None,None


def get_keypoint_mouth(input_img,face_landmarks,bbox,kf_mouth=None,kf=None,kf_68=None):
    with open(f"{os.path.dirname(__file__)}/_wav2lip/config_lipsync_wav2lip.json",'r') as f_lips:
        list_keypoint = json.load(f_lips)
    streamer = "Dr"
    lips_pts = list_keypoint[streamer]["FACEMESH_lips_2_up"]
    lips_pts_small = list_keypoint[streamer]["FACEMESH_lips"]
    lips_pts_face = list_keypoint[streamer]["FACEMESH_bigmask"]
    lips_pts_beard = list_keypoint[streamer]["FACEMESH_beard"]
    landmark_points_68 = list_keypoint[streamer]["landmark_points_68"]
    lips_pts_beard_inner = list_keypoint[streamer]["FACEMESH_beard_inner"]
    # lips_pts_face = list_keypoint[streamer]["FACEMESH_"]
    listpoint2 = []
    bbox_w = bbox[3] - bbox[2]
    bbox_h = bbox[1] - bbox[0]
    # print("Box ", bbox_w,bbox_h)
    h,w = input_img.shape[:2]
    # print(face_landmarks.landmark[:])
    for i in range(len(lips_pts)):
        idx = lips_pts[i]
        x = face_landmarks.landmark[idx].x
        y = face_landmarks.landmark[idx].y

        realx = x * bbox_w + bbox[2]
        realy = y * bbox_h + bbox[0]
        listpoint2.append((realx,realy))
    if kf_mouth is not None:
        # print("Use ")
        if kf_mouth.noneArray():
            # kf.setpoints(listpointLocal, 1e-03)
            kf_mouth.setpoints(listpoint2, w, h)

        else:
            listpoint2 = kf_mouth.getpoints(listpoint2)
    point_mouth = np.array(listpoint2, np.int32)
    point_mouth = point_mouth.reshape(-1,1,2)
    listpoint = []
    for i in range(len(lips_pts_face)):
        idx = lips_pts_face[i]
        x = face_landmarks.landmark[idx].x
        y = face_landmarks.landmark[idx].y

        realx = x * bbox_w + bbox[2]
        realy = y * bbox_h + bbox[0]
        listpoint.append((realx, realy))
    if kf is not None:
        if kf.noneArray():
            # kf.setpoints(listpointLocal, 1e-03)
            kf.setpoints(listpoint, w, h)

        else:
            listpoint = kf.getpoints(listpoint)
    point_face = np.array(listpoint, np.int32)
    point_face = point_face.reshape(-1,1,2)

    list_points_68 = []
    for i in range(len(landmark_points_68)):
        idx = landmark_points_68[i]
        x = face_landmarks.landmark[idx].x
        y = face_landmarks.landmark[idx].y

        realx = x * bbox_w + bbox[2]
        realy = y * bbox_h + bbox[0]
        list_points_68.append((realx, realy))
    if kf_68 is not None:
        if kf_68.noneArray():
            # kf.setpoints(listpointLocal, 1e-03)
            kf_68.setpoints(list_points_68, w, h)

        else:
            list_points_68 = kf_68.getpoints(list_points_68)

    list_points_68 = np.array(list_points_68, np.int32)

    return point_mouth,point_face,list_points_68#,srcpts_beard,srcpts_beard_inner,srcpts_lips_small

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

def add_image_by_mask(img1, img2, mask_):
    mask_not = cv2.bitwise_not(mask_)
    img2_no_mask = cv2.bitwise_and(img2, img2, mask=mask_not)
    img1_mask_only = cv2.bitwise_and(img1, img1, mask=mask_)
    return cv2.add(img2_no_mask, img1_mask_only)

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

def merge_color(videopts_out,videopts,frame,box,pred,alpha):
    # global pts

    video_h,video_w = frame.shape[:2]
    frame_gan_raw = frame.copy()

    #### Image Wav2lip
    # y1, y2, x1, x2 = box
    # frame_gan_raw[y1:y2, x1:x2, :] = pred

    ##Image codeformer
    frame_gan_raw = pred

    mask_mount =  np.zeros((video_h,video_w), np.uint8)
    cv2.fillPoly(mask_mount,pts = [videopts_out], color=(255,255,255))
    mask_face =  np.zeros((video_h,video_w), np.uint8)
    cv2.fillPoly(mask_face,pts = [videopts[:-13]], color=(255,255,255))

    mask_face = cv2.bitwise_or(mask_face, mask_mount)
    topy, topx, bottomy, bottomx, center_face = mask2box(mask_face)

    # alpha_0 = alpha
    # result[topy:bottomy, topx:bottomx] = cv2.addWeighted(result[topy:bottomy, topx:bottomx],\
    # alpha_0, frame[topy:bottomy, topx:bottomx], 1-alpha_0, 0.0)
    result = add_image_by_mask(frame_gan_raw, frame, mask_face)
    #
    #
    # result = final_img
    ###addWeighted to results after match color
    # alpha_0 = alpha
    # result[topy:bottomy, topx:bottomx] = cv2.addWeighted(result[topy:bottomy, topx:bottomx],\
    # alpha_0, frame[topy:bottomy, topx:bottomx], 1-alpha_0, 0.0)
    m_topy, m_topx, m_bottomy, m_bottomx,center_mount = mask2box(mask_mount)
    border_th = 10
    # print(m_topy, m_topx, m_bottomy, m_bottomx)
    if m_topy <border_th or m_topx <border_th or m_bottomy > video_h-border_th or m_bottomx > video_w-border_th:
        img_bgr_uint8_1 = normalize_channels(result[topy:bottomy, topx:bottomx], 3)
        img_bgr_1 = img_bgr_uint8_1.astype(np.float32) / 255.0
        img_bgr_1 = np.clip(img_bgr_1, 0, 1)
        img_bgr_uint8_2 = normalize_channels(frame[topy:bottomy, topx:bottomx], 3)
        img_bgr_2 = img_bgr_uint8_2.astype(np.float32) / 255.0
        img_bgr_2 = np.clip(img_bgr_2, 0, 1)

        result_new = linear_color_transfer(img_bgr_1, img_bgr_2)
        final_img = color_hist_match(result_new, img_bgr_2, 255).astype(dtype=np.float32)
        result[topy:bottomy, topx:bottomx] = blursharpen((final_img*255).astype(np.uint8), 1, 5, 0.5)
    result = add_image_by_mask(result, frame, mask_mount)
    # mask_mount = cv2.GaussianBlur(mask_mount, (11,11), 0)
    # output_main = blend_images_using_mask(result,frame,mask_mount)
    output_main = cv2.seamlessClone(result, frame, mask_mount, center_mount, cv2.NORMAL_CLONE)
    return output_main

def mix_pixel(pix_1, pix_2, perc):
    return (perc/255 * pix_1) + ((255 - perc)/255 * pix_2)

# function for blending images depending on values given in mask
def blend_images_using_mask(img_orig, img_for_overlay, img_mask):
    if len(img_mask.shape) != 3:
        img_mask = cv2.cvtColor(img_mask, cv2.COLOR_GRAY2BGR)
    img_res = mix_pixel(img_orig, img_for_overlay, img_mask)
    return img_res.astype(np.uint8)

def dist_point(p1,p2):
    distance = math.sqrt( ((p1[0]-p2[0])**2)+((p1[1]-p2[1])**2) )
    return distance

def mask2box(mask2d):
    (y, x) = np.where(mask2d > 0)
    topy = np.min(y)
    topx = np.min(x)
    bottomy = np.max(y)
    bottomx = np.max(x)
    # print()
    # (topy, topx) = (np.min(y), np.min(x))
    # (bottomy, bottomx) = (np.max(y), np.max(x))
    center_ = (int((topx+bottomx)/2),int((bottomy+topy)/2))

    return topy, topx, bottomy, bottomx, center_

def gen_lipsync_img_v2(img, mel_chunk, model, device):
    """
    """
    resize_factor = 1
    MODEL_INPUT_IMG_SIZE = 96
    cr_h,cr_w = img.shape[:2]
    roi = img    # roi = cv2.GaussianBlur(roi,(7,7),0)
    roi = my_cv2_resize(roi, MODEL_INPUT_IMG_SIZE, MODEL_INPUT_IMG_SIZE)

    img_batch, mel_batch = np.asarray([roi]), np.asarray([mel_chunk])
    img_masked = img_batch.copy()
    img_masked[:, MODEL_INPUT_IMG_SIZE//2:] = 0

    img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
    mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])
    # start_2 = time.time()
    img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
    mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)
    with torch.no_grad():
        pred = model(mel_batch, img_batch)
    # time_GPu.append(time.time() - start_2)
    pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.
    pred = pred.astype(np.uint8)[0]
    pred = my_cv2_resize(pred, cr_w, cr_h)
    return pred

def lipsync_one_frame(frame, frame_ori,mel_chunk, facemesh, MASK_KP_IDS,  wav2lip_model,\
                        mobile_net_wav2lip, resnet_net_wav2lip, device,\
                        kf,kf_mouth,kf_68):
    """
    """
    blur_threshold = 10.69
    # box = face_detect([frame], sfd_facedetector)[0]
    frame_ori_cop = frame_ori.copy()
    frame_cop =  frame.copy()
    video_h,video_w = frame.shape[:2]
    
    box, five_points,padding_box,five_points_tuple = get_box_by_RetinaFace(frame, mobile_net_wav2lip, resnet_net_wav2lip, device)
    
    if box==None:
        print("Not detect")
        return frame_ori_cop, None

    x1, y1, x2, y2 = box
    bbox =box
    size_box_face = np.abs((box[0]-box[2])*(box[1]-box[3]))
    if size_box_face/(video_w*video_h) <= 0.0112:  ## Size box of face conditions ###
        print("Small face box")
        return frame_ori_cop, None
    # num_mouth_frame_loss += 1

    box = [y1,y2,x1,x2]
    roi = frame[y1:y2,x1:x2, :]
    padding_box_cop = padding_box
    roi_2 = frame[padding_box[0][1]:padding_box[1][1], padding_box[0][0]:padding_box[1][0],:]
    landmark,face_landmarks = get_fm_landmark(roi_2, facemesh)
    if landmark==None or face_landmarks == None:
        print("Not detect landmark")
        return frame_ori_cop,None
    padding_box = [padding_box[0][1],padding_box[1][1], padding_box[0][0],padding_box[1][0]]
    videopts_out,videopts,list_points_68= get_keypoint_mouth(frame,face_landmarks,padding_box,kf_mouth,kf,kf_68)
    mask_box = np.zeros((video_h,video_w), np.uint8)
    cv2.fillPoly(mask_box, pts =[videopts_out], color=(255,255,255))
    m_topy, m_topx, m_bottomy, m_bottomx,_ = mask2box(mask_box)
    border_th = 13

    # print(m_topy, m_topx, m_bottomy, m_bottomx)
    # if m_topy <border_th or m_topx <border_th or m_bottomy > video_h-border_th or m_bottomx > video_w-border_th:
    #     print('Box touched', list_points_68[57][:])
    #     return frame_ori_cop
    # newval,_,_ = find_ratio_intersection(frame,box,videopts_out,to_tensor,model_Occ)
    # if newval < 0.6:
    #     return frame_ori

    # pred = gen_lipsync_img(frame_ori, mel_chunk, wav2lip_model, box, device)
    # restored_img  = pred
    # frame[y1:y2, x1:x2, :] = restored_img

    # # frame_cop = merge_color(videopts_out,videopts,frame_ori,box,frame_cop,0.95,facemesh)
    # cv2.imwrite("frame_cop.png",frame_cop)
    # x1_ex,y1_ex, x2_ex,y2_ex = expand_box_face(frame,box,padding_ratio=.15)
    # restored_img = frame_cop[y1_ex:y2_ex, x1_ex:x2_ex, :]
    # # roi_2 = frame[y1_ex:y2_ex, x1_ex:x2_ex, :]
    # # expand_box = [y1_ex,y2_ex,x1_ex,x2_ex]
    # frame_cop = frame

    ###Wav2Lip
    crop_size =(350,400)
    crop_img,affine_matrix_wav2lip = warpaffine_face_wav2lip(frame_cop,five_points,face_size=crop_size,border_mode='transparent')
    # cv2.imwrite("Crop_image.png",crop_img)
    
    pred = gen_lipsync_img_v2(crop_img, mel_chunk, wav2lip_model, device)
    
    mask_wav2lip_warp = 255 * np.ones((crop_size[1],crop_size[0]), np.uint8)
    inverse_affine_wav2lip = cv2.invertAffineTransform(affine_matrix_wav2lip)
    restored_img = cv2.warpAffine(pred, inverse_affine_wav2lip, (video_w, video_h))
    mask_wav2lip_warp = cv2.warpAffine(mask_wav2lip_warp, inverse_affine_wav2lip, (video_w, video_h))
    frame_cop = merge_color(videopts_out,videopts,frame,box,restored_img,0.95)
    #
    # frame_cop = add_image_by_mask(restored_img,frame_cop,mask_wav2lip_warp)
    # #
    # # # # ###Code former process
    # restored_img,affine_matrix = warp_face_codeformer(frame_cop,five_points,face_size=512,border_mode='constant')
    # restored_img = process_img(restored_img,codeformer,face_helper,bg_upsampler, device,fidelity_weight=1.0,\
    # upscale =1,has_aligned=True, draw_box = False,face_upsample = False,only_center_face = False)
    # # #### Warp back
    # inverse_affine = cv2.invertAffineTransform(affine_matrix)
    # mask_codeformer_warp = 255 * np.ones((512,512), np.uint8)
    # mask_codeformer_warp = cv2.warpAffine(mask_codeformer_warp, inverse_affine, (video_w, video_h))
    # restored_img = cv2.warpAffine(restored_img, inverse_affine, (video_w, video_h))
    # frame = merge_color(videopts_out,videopts,frame,box,restored_img,0.95)
    return frame_cop,list_points_68
 
def mix_pixel(pix_1, pix_2, perc):
    return (perc/255 * pix_1) + ((255 - perc)/255 * pix_2)

# function for blending images depending on values given in mask
def blend_images_using_mask(img_orig, img_for_overlay, img_mask):
    if len(img_mask.shape) != 3:
        img_mask = cv2.cvtColor(img_mask, cv2.COLOR_GRAY2BGR)
    img_res = mix_pixel(img_orig, img_for_overlay, img_mask)
    return img_res.astype(np.uint8)

def get_pad_codeformer(input_img,all_landmarks_5,border_mode,face_size):
    face_template = np.array([[192.98138, 239.94708], [318.90277, 240.1936], [256.63416, 314.01935],
                                   [201.26117, 371.41043], [313.08905, 371.15118]])
    for idx, landmark in enumerate(all_landmarks_5):
        # use 5 landmarks to get affine matrix
        # use cv2.LMEDS method for the equivalence to skimage transform
        # ref: https://blog.csdn.net/yichxi/article/details/115827338
        affine_matrix = cv2.estimateAffinePartial2D(landmark, face_template, method=cv2.LMEDS)[0]
        # affine_matrices.append(affine_matrix)
        # warp and crop faces
        if border_mode == 'constant':
            border_mode = cv2.BORDER_CONSTANT
        elif border_mode == 'reflect101':
            border_mode = cv2.BORDER_REFLECT101
        elif border_mode == 'reflect':
            border_mode = cv2.BORDER_REFLECT
        # if pad_blur:
        #     input_img = pad_input_imgs[idx]
        # else:
        #     input_img = input_img
        cropped_face = cv2.warpAffine( input_img, affine_matrix, face_size, \
        borderMode=border_mode, borderValue=(135, 133, 132))
    return cropped_face
def ffmpeg_encoder(outfile, fps, width, height):
    LOGURU_FFMPEG_LOGLEVELS = {
        "trace": "trace",
        "debug": "debug",
        "info": "info",
        "success": "info",
        "warning": "warning",
        "error": "error",
        "critical": "fatal",
        }

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
                # vcodec="libx265",
                vcodec="h264_nvenc",
                acodec="copy",
                r=fps,
                cq=16,
                maxrate="5M",
                minrate= "5M",
                bufsize= "5M",
                # vsync="1",
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
            cmd="/home/ubuntu/anaconda3/envs/gazo/bin/ffmpeg",
        ),
        stdin=subprocess.PIPE,
        # stdout=subprocess.DEVNULL,
        # stderr=subprocess.DEVNULL,
    )
    return encoder_


def load_speech_timestamp(fps,data):
    # print("DATA TIMESTAMP: ", data , " - ", type(data))
    List_speak_frame = []
    
    list_inital = json.loads(data)
    # print("DATA ", data[:2])
    if data[:2] == "[[":
        # print("group")
        flattened_list = [item for sublist in list_inital for item in sublist]
        list_inital = flattened_list
    # print(list_inital)
    for j in range(len(list_inital)):
        # if list_inital[j]['character'] == 'DrDisrespect':
        start = int(float(list_inital[j]['timestamp'])/1000*fps)
        stop = start + int(float(list_inital[j]['duration'])/1000*fps)
        # print(j,"-",start, stop)
        for i in range(start,stop):
            List_speak_frame.append(i)
        # else:
        #     continue
    # print(Lis)
    return np.unique(List_speak_frame)

def write_frame(images,encoder_video):
    image_draw = cv2.cvtColor(images,cv2.COLOR_RGB2BGR)
    imageout = Image.fromarray(np.uint8(image_draw))
    encoder_video.stdin.write(imageout.tobytes())

def expand_box_face(image,box,padding_ratio):
    padding_size_ratio = padding_ratio
    H,W = image.shape[:2]
    bbox = [box[2],box[0],box[3],box[1]]#((xmin, ymin , xmax, ymax))
    topleft = (int(bbox[0]), int(bbox[1]))
    bottomright = (int(bbox[2]), int(bbox[3]))
    hw_ratio = (bottomright[1] - topleft[1])/(bottomright[0] - topleft[0])
    # print("HW_ratio:",hw_ratio)
    # center = (int((bbox[0]+bbox[2])/2), int((bbox[1]+bbox[3])/2))
    padding_X = int((bottomright[0] - topleft[0]) * padding_size_ratio*hw_ratio*1.2)
    padding_Y = int((bottomright[1] - topleft[1]) * padding_size_ratio/hw_ratio)
    padding_topleft = (max(0, topleft[0] - padding_X), max(0, topleft[1]- padding_Y))
    padding_bottomright = (min(W, bottomright[0] + padding_X), min(H, bottomright[1] + int(padding_Y/1.3)))
    # coordinate = (padding_topleft, padding_bottomright)
    return [padding_topleft[0],padding_topleft[1],padding_bottomright[0],padding_bottomright[1]]
    # l_coordinate.append(coordinate)

def render_Wav2lip_CPU_preprocess(listimages,list_videopts,list_videopts_out,list_box,list_matrixwarp,merge_images_preprocess,list_points_68_wav2lip,start,end,List_silence_frame):
    device = 'cuda'
    start_load_wav2lip = time.time()
    mobile_net_wav2lip, resnet_net_wav2lip = loadmodelface()    #Face Occlusion
    # wav2lip_modelpath = f'{os.path.dirname(__file__)}/wav2lip_model/wav2lip_gan.pth'
    MASK_KP_IDS = [2,326, 423,425 ,411,416, 430, 431, 262, 428, 199, 208, 32, 211, 210,192, 187, 205 , 203, 97]
    facemesh = init_face_mesh()
    # wav2lip_model = load_wav2lip_model(wav2lip_modelpath, device)
    
    print("Load model WAv2lip in: ", time.time() - start_load_wav2lip)
    trackkpVideoFace = KalmanArray()
    trackkpVideoMount = KalmanArray()
    trackkpVideo_68 = KalmanArray()
    start_x = time.time()
    # Start render wav2lip
    for idx in tqdm(range(start, end), 'Wav2lip preprocess'):
        if idx in List_silence_frame:
            merge_images_preprocess[idx] = listimages[idx]
            list_points_68_wav2lip[idx] = None
        else: 
            # res, list_points_68 = lipsync_one_frame(listimages[idx],listimages[idx], mel_chunk, facemesh, MASK_KP_IDS, \
            #                         wav2lip_model, mobile_net_wav2lip, resnet_net_wav2lip,\
            #                         device,trackkpVideoFace,trackkpVideoMount,trackkpVideo_68) #sfd_facedetector,codeformer, face_helper,bg_upsampler,
            blur_threshold = 10.69
            # box = face_detect([frame], sfd_facedetector)[0]
            frame_ori_cop = listimages[idx].copy()
            frame_cop =  listimages[idx].copy()
            video_h,video_w = listimages[idx].shape[:2]
            
            box, five_points,padding_box,five_points_tuple = get_box_by_RetinaFace(listimages[idx], mobile_net_wav2lip, resnet_net_wav2lip, device)
            
            if box==None:
                print("Not detect")
                return frame_ori_cop, None

            x1, y1, x2, y2 = box
            bbox =box
            size_box_face = np.abs((box[0]-box[2])*(box[1]-box[3]))
            if size_box_face/(video_w*video_h) <= 0.0112:  ## Size box of face conditions ###
                print("Small face box")
                return frame_ori_cop, None
            # num_mouth_frame_loss += 1

            box = [y1,y2,x1,x2]
            roi = listimages[idx][y1:y2,x1:x2, :]
            padding_box_cop = padding_box
            roi_2 = listimages[idx][padding_box[0][1]:padding_box[1][1], padding_box[0][0]:padding_box[1][0],:]
            landmark,face_landmarks = get_fm_landmark(roi_2, facemesh)
            if landmark==None or face_landmarks == None:
                print("Not detect landmark")
                return frame_ori_cop,None
            padding_box = [padding_box[0][1],padding_box[1][1], padding_box[0][0],padding_box[1][0]]
            videopts_out,videopts,list_points_68= get_keypoint_mouth(listimages[idx],face_landmarks,padding_box,trackkpVideoFace,trackkpVideoMount,trackkpVideo_68)
            mask_box = np.zeros((video_h,video_w), np.uint8)
            cv2.fillPoly(mask_box, pts =[videopts_out], color=(255,255,255))
            m_topy, m_topx, m_bottomy, m_bottomx,_ = mask2box(mask_box)
            border_th = 13
            ###Wav2Lip
            crop_size =(350,400)
            crop_img,affine_matrix_wav2lip = warpaffine_face_wav2lip(frame_cop,five_points,face_size=crop_size,border_mode='transparent')
            # cv2.imwrite("Crop_image.png",crop_img)
            
            # pred = gen_lipsync_img_v2(crop_img, mel_chunk, wav2lip_model, device)
            
            # mask_wav2lip_warp = 255 * np.ones((crop_size[1],crop_size[0]), np.uint8)
            # inverse_affine_wav2lip = cv2.invertAffineTransform(affine_matrix_wav2lip)
            # restored_img = cv2.warpAffine(pred, inverse_affine_wav2lip, (video_w, video_h))
            # mask_wav2lip_warp = cv2.warpAffine(mask_wav2lip_warp, inverse_affine_wav2lip, (video_w, video_h))
            # frame_cop = merge_color(videopts_out,videopts,listimages[idx],box,restored_img,0.95)
            
            merge_images_preprocess[idx] = crop_img
            list_videopts[idx] = videopts.tolist()
            list_videopts_out[idx] = videopts_out.tolist()
            list_matrixwarp[idx] = affine_matrix_wav2lip
            list_box[idx] = box
    del mobile_net_wav2lip, resnet_net_wav2lip,facemesh 
    # print("Time CPU: ", time.time()-start_x )
 

def render_Wav2lip_CPU_final(listimages,listimages_predict,list_videopts,list_videopts_out,list_box,list_matrixwarp,merge_images_final,start,end,List_silence_frame):
    MASK_KP_IDS = [2,326, 423,425 ,411,416, 430, 431, 262, 428, 199, 208, 32, 211, 210,192, 187, 205 , 203, 97]
    # Start render wav2lip final
    video_h,video_w = listimages[0].shape[:2]
    for idx in tqdm(range(start, end), 'Wav2lip process'):
        if idx in List_silence_frame:
            merge_images_final[idx] = listimages[idx]
        else: 

            crop_size =(350,400)
            
            # affine_matrix_wav2lip
            # crop_img,affine_matrix_wav2lip = warpaffine_face_wav2lip(frame_cop,five_points,face_size=crop_size,border_mode='transparent')
            # cv2.imwrite("Crop_image.png",crop_img)
            
            # pred = gen_lipsync_img_v2(crop_img, mel_chunk, wav2lip_model, device)
            
            mask_wav2lip_warp = 255 * np.ones((crop_size[1],crop_size[0]), np.uint8)
            inverse_affine_wav2lip = cv2.invertAffineTransform(list_matrixwarp[idx])
            restored_img = cv2.warpAffine(listimages_predict[idx], inverse_affine_wav2lip, (video_w, video_h))
            mask_wav2lip_warp = cv2.warpAffine(mask_wav2lip_warp, inverse_affine_wav2lip, (video_w, video_h))
            frame_cop = merge_color(np.array(list_videopts_out[idx]),np.array(list_videopts[idx]),listimages[idx],list_box[idx],restored_img,0.95)
            
            merge_images_final[idx] = frame_cop
            
    # print("Time CPU: ", time.time()-start_x )

def status_checking(response, endpoint_id):
    response = json.loads(response.text)
    job_id = response['id']

    status_url = f"https://api.runpod.ai/v2/{endpoint_id}/status/{job_id}"

    headers = {
        "accept": "application/json",
        "authorization": api_key
    }

    response = requests.get(status_url, headers=headers)

    return response

def inference(endpoint, video_url, audio_url):
    payload = {
    "input":{
        "video":video_url,
        "audio":audio_url,
        "silence":[],
    }    
}
    print(payload)
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "authorization": api_key
    }
    logging.info(f"Request body: ${payload}")
    response = requests.post(endpoint['url'], json=payload, headers=headers)

    response = status_checking(response, endpoint_id=endpoint['id'])
    response_json = json.loads(response.text)
    logging.info(f"Process ID runpod: {response_json['id']}")

    while response_json['status'] != 'COMPLETED':
        response = status_checking(response, endpoint_id=endpoint['id'])
        response_json = json.loads(response.text)
        if response_json['status'] == 'IN_PROGRESS' or response_json['status'] == 'IN_QUEUE':
            time.sleep(1)
        if response_json['status'] == 'FAILED':
            return 'FAILED', response
    return 'COMPLETED', response

def genWav2lip(
        video_path: str,
        audio_path: str,
        output: str,
):
    endpoint=endpoints['wav2lip_model']
    # upload image, get image url
    tmp_video = generate_random_filename(extension=os.path.splitext(video_path)[-1])
    # logging.info(f"Video file name s3: {tmp_video}")
    shutil.copyfile(video_path, tmp_video)
    bucket_name = 'vi-lang-pipeline'
    object_key = f'runpod/inputs/rmbg/tests/{tmp_video}'
    s3.upload_file(
        tmp_video, bucket_name, object_key
    )
    os.remove(tmp_video)
    video_url = f"https://vi-lang-pipeline.s3.us-west-2.amazonaws.com/runpod/inputs/rmbg/tests/{tmp_video}"
    
    # Upload audio to s3 and get url
    tmp_audio_fn = generate_random_filename(extension=os.path.splitext(audio_path)[-1])
    logging.info(f"Audio file name s3: {tmp_audio_fn}")
    shutil.copyfile(audio_path, tmp_audio_fn)
    bucket_name = 'vi-lang-pipeline'
    object_key = f'runpod/inputs/rmbg/tests/{tmp_audio_fn}'
    s3.upload_file(
        tmp_audio_fn, bucket_name, object_key
    )
    os.remove(tmp_audio_fn)
    audio_url = f"https://vi-lang-pipeline.s3.us-west-2.amazonaws.com/runpod/inputs/rmbg/tests/{tmp_audio_fn}"

    status, response = inference(endpoint, video_url,audio_url)
    result_json = json.loads(response.text)
    logging.info(f"Runpod response: {result_json}")
    if status == 'COMPLETED':
        result_url = result_json['output']["video_url"]
        fileDirName = result_url.split("com/")[1]
        s3.download_file('vi-lang-pipeline-public',fileDirName,output)
        time.sleep(1)
        # return output
        # r = requests.get(result_url)
        # with open(output_path, 'wb') as file:
        #     for chunk in r.iter_content(chunk_size=8192):
        #         if chunk:
        #             file.write(chunk)
    else:
        logging.info(f"error")
 


def processVideo(input_video,output_video,input_audio,savepathListpoint68,silence):
    start_1 = time.time()
    # Load video frame and mel chunks audio
    listimage = []
    cam = cv2.VideoCapture(input_video)
    fps = cam.get(cv2.CAP_PROP_FPS)
    width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # mel_chunks = get_mel_chunks(input_audio,fps)
    List_silence_frame = load_speech_timestamp(fps,silence)
    savepath_nonsound = output_video.split(".mp4")[0]+"_tmp.mp4"
    st = time.time()
    while(True): 
        ret,frame = cam.read() 
        if ret: 
            listimage.append(frame)
        else: 
            break
    cam.release()
    readvideotime = time.time()
    print("Load video to RAM in: ", readvideotime -st, " s")
    
    # Check VRAM and get num threads
    # total_memory =  torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
    # print(f"Total GPU Memory: {total_memory} MB")
    num_threads = 1#np.min([np.floor(total_memory/4000).astype(int),4])  # Define num_threads here
    
    # Start processes render
    multi_thread = []
    count = len(listimage)
    images_per_thread = count // num_threads
    extra = count % num_threads 
    with Manager() as manager:
        merge_images_preprocess = manager.dict()
        merge_images_final = manager.dict()
        list_points_68_wav2lip = manager.dict()
        # time_GPu =  manager.list()
        list_videopts = manager.dict()
        list_videopts_out = manager.dict()
        list_box = manager.dict()
        list_matrixwarp = manager.dict()
        time_2 = time.time()
        logging.info(f"Prepare data before process completed in :{time_2-start_1}")
        #Gen face preprocess
        for i in range(num_threads):
            start = i * images_per_thread + min(i, extra)
            if i < extra:
                end = start + images_per_thread + 1
            else:
                end = start + images_per_thread
            p = threading.Thread(target=render_Wav2lip_CPU_preprocess,
                        args=(listimage,list_videopts,list_videopts_out,list_box,list_matrixwarp,merge_images_preprocess,list_points_68_wav2lip,start,end,List_silence_frame))
            multi_thread.append(p)
        for p in multi_thread:
            p.start()
        for p in multi_thread:
            p.join()
        time_3 = time.time()
        logging.info(f"Run Wav2lip preprocess completed in :{time_3-time_2}")
        # Merge preprocess video
        print("Start merge preprocess frame to video")
        encoder_video = ffmpeg_encoder(savepath_nonsound, fps,350, 400)
        # print("Time using GPU: ",sum(time_GPu))
        start_merge = time.time()
        for i in range(len(merge_images_preprocess)):
            write_frame(merge_images_preprocess[i],encoder_video)
        end_merge = time.time()
        encoder_video.stdin.flush()
        encoder_video.stdin.close()
        logging.info(f"Time merge preprocess frame to video: {end_merge - start_merge}")
        
        # # Gen wav2lip output
        genWav2lip(input_video, input_audio,output_video)
        # # Merge final frame
        time_4 = time.time()
        
        listimages_predict = []
        cam = cv2.VideoCapture(output_video)
        while(True): 
            ret,frame = cam.read() 
            if ret: 
                listimages_predict.append(frame)
            else: 
                break
        cam.release()
        multi_thread = []
        for i in range(num_threads):
            start = i * images_per_thread + min(i, extra)
            if i < extra:
                end = start + images_per_thread + 1
            else:
                end = start + images_per_thread
            p = threading.Thread(target=render_Wav2lip_CPU_final,
                        args=(listimage,listimages_predict,list_videopts,list_videopts_out,list_box,list_matrixwarp,merge_images_final,start,end,List_silence_frame))
            multi_thread.append(p)
        for p in multi_thread:
            p.start()
        for p in multi_thread:
            p.join()
        time_5 = time.time()   
        logging.info(f"Run merge Wav2lip predict to final frame completed in :{time_5-time_4}")
        
        print("Start merge final frame to video")
        encoder_video = ffmpeg_encoder(savepath_nonsound, fps,width, height)
        # print("Time using GPU: ",sum(time_GPu))
        start_merge = time.time()
        for i in range(len(merge_images_preprocess)):
            write_frame(merge_images_final[i],encoder_video)
        end_merge = time.time()
        encoder_video.stdin.flush()
        encoder_video.stdin.close()
        logging.info(f"Time merge final frame to video:{end_merge - start_merge}")
        # time_GPu.append(end_merge - start_merge)
        
        # print(list_points_68_wav2lip)
        # Write listpoint68 to json file
        
        with open(savepathListpoint68,'w') as f:
            json.dump(list_points_68_wav2lip.copy(), f)
        print("Write listpoint 68 to json file")
        ffmpeg_cmd = f"""ffmpeg -y  -hide_banner -loglevel quiet -i {savepath_nonsound} -i '{input_audio}' -c:v copy {output_video}"""
        print(ffmpeg_cmd)
        os.system(ffmpeg_cmd)
        time.sleep(1)


if __name__=='__main__':
    date = datetime.today().strftime('%Y-%m-%d')
    logging.basicConfig(filename=f'/tmp/wav2lip-runpod-{date}.log',filemode = 'a', level=logging.INFO,format='%(asctime)s - %(levelname)s- %(message)s')
    multiprocessing.set_start_method("spawn")
    video_path = args.input_video
    savepathWav2lip= args.output_video
    audio_path = args.input_audio
    silence = args.silences
    savepathListpoint68 = args.input_video.split(".mp4")[0] + "_list68.json"
    
    processVideo(video_path,savepathWav2lip,audio_path,savepathListpoint68,silence)