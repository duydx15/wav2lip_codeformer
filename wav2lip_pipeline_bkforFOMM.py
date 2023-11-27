"""
"""
import sys
import json
import os
import audio
import numpy as np
import torch
from _wav2lip.models import Wav2Lip
import cv2
from tqdm import tqdm
import face_detection
import math
# from gfp_gan_wav2lip import load_gfpgan_model
# from vqfr_ import load_vqfr_model
import mediapipe as mp
import ffmpeg
import subprocess
from PIL import Image
import warnings
from pipeline_mobile_resnet_wav2lip import loadmodelface, detection_face_wav2lips
from torchvision import transforms as TF_s
from collections import OrderedDict
import segmentation_models_pytorch as smp
from color_transfer import color_transfer, color_hist_match,linear_color_transfer
from common import  normalize_channels
warnings.filterwarnings("ignore")
import tensorflow as tf

# gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.2)
# config=tf.compat.v1.ConfigProto(gpu_options=gpu_options)
# session = tf.compat.v1.Session(config=config)

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

def load_model(path, device):
    # checkpoint
    if device == 'cuda':
        checkpoint = torch.load(path)
    else:
        checkpoint = torch.load(path,
                                map_location=lambda storage, loc: storage)
    # model
    model = Wav2Lip()
    print("Load checkpoint from: {}".format(path))
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s)

    model = model.to(device)
    return model.eval()

def face_detect(images, detector):
    predictions = []
    predictions.extend(detector.get_detections_for_batch(np.array(images)))

    results = []
    pady1, pady2, padx1, padx2 = 0, 0, 0, 0
    for rect, image in zip(predictions, images):
        if rect is None:
            return [None]
        else:
            y1 = max(0, rect[1] - pady1)
            y2 = min(image.shape[0], rect[3] + pady2)
            x1 = max(0, rect[0] - padx1)
            x2 = min(image.shape[1], rect[2] + padx2)

            results.append([x1, y1, x2, y2])

    boxes = np.array(results)
    results = [[y1, y2, x1, x2] for (x1, y1, x2, y2) in boxes]

    return results

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
    y_ori1, y_ori2, x_ori1, x_ori2 = box
    y1, y2, x1, x2 = (np.array(box)//resize_factor).astype(np.int32)

    roi = img[y1:y2, x1:x2, :]
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
    return mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

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

def find_ratio_intersection(videoimg,crops_coors,videopts_out,to_tensor,model_Occ):

    crop_image = videoimg[crops_coors[0]:crops_coors[1],crops_coors[2]:crops_coors[3],:]
    crop_image_occ = np.copy(crop_image)
    video_h,video_w, = videoimg.shape[:2]
    crop_image_occ = cv2.resize(crop_image_occ,(256,256)) # (256,256) is input size of FaceOcc

    video_h,video_w, = videoimg.shape[:2]
    mask_mount =  np.zeros((video_h,video_w), np.uint8)
    cv2.fillPoly(mask_mount,pts = [videopts_out], color=(255,255,255))
    #Predict mask by FaceOcc
    data = to_tensor(crop_image_occ).unsqueeze(0)
    data = data.to('cuda:0')
    with torch.no_grad():
        pred = model_Occ(data)
    pred_mask = (pred > 0).type(torch.int32)
    pred_mask = pred_mask.squeeze().cpu().numpy()
    mask_occ = cv2.resize(pred_mask,(crop_image.shape[1],crop_image.shape[0]),interpolation=cv2.INTER_LINEAR_EXACT) #Resize up need INTER_LINEAR_EXACT

    img_face_occ = np.zeros((video_h,video_w), np.uint8)
    img_face_occ[crops_coors[0]:crops_coors[1],crops_coors[2]:crops_coors[3]] = mask_occ

    img_face_occ = fillhole(img_face_occ*255)

    var_middle_occ = np.copy(mask_mount[:,:])
    img_mouth_mask_occ = cv2.bitwise_and(img_face_occ,var_middle_occ)
    mount_mask_Occ = img_mouth_mask_occ.copy()
    mask_mount = np.atleast_3d(mask_mount).astype(np.float) / 255.
    mask_mount[mask_mount != 1] = 0

    img_mouth_mask_occ = np.atleast_3d(img_mouth_mask_occ).astype(np.float) / 255.
    img_mouth_mask_occ[img_mouth_mask_occ != 1] = 0
    newval = len(img_mouth_mask_occ[img_mouth_mask_occ > 0])/len(mask_mount[mask_mount >0])
    return newval,mount_mask_Occ,img_face_occ

def get_box_by_RetinaFace(inimg, mobile_net_wav2lip, resnet_net_wav2lip, device):

    h,w = inimg.shape[:2]
    padding_ratio = [0]
    padding_size_ratio = 0.1
    for r_idx in range(len(padding_ratio)):
        l_coordinate, detected_face,five_points= detection_face_wav2lips(mobile_net_wav2lip,resnet_net_wav2lip, inimg, device,padding_ratio[r_idx])
        if not detected_face and r_idx == len(padding_ratio)-1:
            return None,None,None
        elif not detected_face and r_idx < len(padding_ratio)-1:
            continue
        bbox = None
        for i in range(len(l_coordinate)):
            topleft, bottomright = l_coordinate[i]
            curbox = []
            if bottomright[1] < h/3:
                continue
            h_w_dis = int((bottomright[1]-topleft[1]) - (bottomright[0]-topleft[0]))//2
            topleft = (max(0, topleft[0]-h_w_dis),topleft[1])
            bottomright = (min(w, bottomright[0]+h_w_dis),bottomright[1])
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
            return curbox,landmarks_5,coordinate
        else:
            return None,None,None

def get_keypoint_mouth(input_img,face_landmarks,bbox,kf_mouth=None,kf=None,kf_68=None):
    with open("/home/ubuntu/Duy_test_folder/Retinaface_Mediapipe/_wav2lip/config_lipsync_wav2lip.json",'r') as f_lips:
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

    h,w = input_img.shape[:2]

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

    return point_mouth,point_face,list_points_68#

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

def merge_color(videopts_out,videopts,frame,box,pred,alpha,facemesh):
    # global pts

    video_h,video_w = frame.shape[:2]
    frame_gan_raw = frame.copy()
    y1, y2, x1, x2 = box

    #### Image Wav2lip
    frame_gan_raw[y1:y2, x1:x2, :] = pred

    ##Image codeformer
    # frame_gan_raw = pred

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

    img_bgr_uint8_1 = normalize_channels(result[topy:bottomy, topx:bottomx], 3)
    img_bgr_1 = img_bgr_uint8_1.astype(np.float32) / 255.0
    img_bgr_1 = np.clip(img_bgr_1, 0, 1)
    img_bgr_uint8_2 = normalize_channels(frame[topy:bottomy, topx:bottomx], 3)
    img_bgr_2 = img_bgr_uint8_2.astype(np.float32) / 255.0
    img_bgr_2 = np.clip(img_bgr_2, 0, 1)

    # result_new = linear_color_transfer(img_bgr_1, img_bgr_2)
    # final_img = color_hist_match(result_new, img_bgr_2, 255).astype(dtype=np.float32)
    # result[topy:bottomy, topx:bottomx] = blursharpen((final_img*255).astype(np.uint8), 1, 5, 0.5)
    # result = final_img
    #addWeighted to results after match color
    # alpha_0 = alpha
    # result[topy:bottomy, topx:bottomx] = cv2.addWeighted(result[topy:bottomy, topx:bottomx],\
    # alpha_0, frame[topy:bottomy, topx:bottomx], 1-alpha_0, 0.0)
    m_topy, m_topx, m_bottomy, m_bottomx,center_mount = mask2box(mask_mount)
    border_th = 13
    # print(m_topy, m_topx, m_bottomy, m_bottomx)
    if m_topy <border_th or m_topx <border_th or m_bottomy > video_h-border_th or m_bottomx > video_w-border_th:
        print("Box touched")
        return frame
    result = add_image_by_mask(result, frame, mask_mount)
    # mask_mount = cv2.GaussianBlur(mask_mount, (31,31), 0)
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

def lipsync_one_frame(frame, frame_ori,mel_chunk, facemesh, MASK_KP_IDS,  wav2lip_model,\
                        mobile_net_wav2lip, resnet_net_wav2lip, device,kf,kf_mouth,kf_68):
    """
    """
    # globaloldFrame,pts
    # box = face_detect([frame], sfd_facedetector)[0]
    # frame_ori_cop = frame_ori.copy()
    frame_cop =  frame.copy()
    box, five_points,padding_box = get_box_by_RetinaFace(frame, mobile_net_wav2lip, resnet_net_wav2lip, device)
    if box==None:
        print("Not detect")
        return frame_ori


    x1, y1, x2, y2 = box
    box = [y1, y2, x1, x2]
    roi = frame[y1:y2,x1:x2, :]
    print("Size box", roi.shape)
    roi_2 = frame[padding_box[0][1]:padding_box[1][1], padding_box[0][0]:padding_box[1][0],:]
    landmark,face_landmarks = get_fm_landmark(roi_2, facemesh)
    if landmark==None or face_landmarks == None:
        return frame_ori
    padding_box = [padding_box[0][1],padding_box[1][1], padding_box[0][0],padding_box[1][0]]
    videopts_out,videopts,list_points_68= get_keypoint_mouth(frame,face_landmarks,padding_box,kf_mouth,kf,kf_68)
    # newval,_,_ = find_ratio_intersection(frame,box,videopts_out,to_tensor,model_Occ)
    # if newval < 0.6:
    #     return frame_ori

    # frame[y1:y2, x1:x2, :] = restored_img
    # # ## Preprocessing of Wav2lip output - Blured-Mask cover


    pred = gen_lipsync_img(frame, mel_chunk, wav2lip_model, box, device)
    # frame_cop[y1:y2, x1:x2, :] = pred
    # pred = gen_lipsync_img(frame_cop, mel_chunk, wav2lip_model, box, device)
    restored_img  = pred
    # restored_img = gfpgan_img(restored_img, gfpgan_model)
    frame = merge_color(videopts_out,videopts,frame,box,restored_img,0.95,facemesh)
    cv2.rectangle(frame,( x1,y1),(x2,y2),(255,0,0),2)
    cv2.rectangle(frame,( padding_box[2], padding_box[0]),( padding_box[3], padding_box[1]),(0,255,0),2)
    cv2.polylines(frame,[videopts_out],True,color=(0,0,255),thickness=2)
    # frame[y1:y2, x1:x2, :] = restored_img

    return frame

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


# def load_speech_timestamp(fps):
#     List_speak_frame = []
#     json_path = '/home/ubuntu/Duy_test_folder/SP_Speechdetection/content/SpeakerDetection/demo7-es-US_20221013.json'
#     with open(json_path,'r') as f:
#         list_inital = json.load(f)
#         List_speak_frame = []#list_inital['data'][:]
#         for j in range(len(list_inital['data'])):
#             start = int(float(list_inital['data'][j]['start'])*fps)-3
#             stop = int(float(list_inital['data'][j]['end'])*fps)
#             # print(start, stop)
#             for i in range(start,stop+1):
#                 List_speak_frame.append(i)
#     return np.unique(List_speak_frame)

def write_frame(images,encoder_video):
    image_draw = cv2.cvtColor(images,cv2.COLOR_RGB2BGR)
    imageout = Image.fromarray(np.uint8(image_draw))
    encoder_video.stdin.write(imageout.tobytes())



def main():
    vidpath = 'Dr_video/DrDisrespect-Falls-in-Love-with-Warzone-again-thanks-to-new-Game-Mode.mp4'
    frame_path = 'Dr_video/DrDisrespect-Falls-in-Love-with-Warzone-again-thanks-to-new-Game-Mode.mp4'
    savepath = 'Wav2lip_gan_60FPS_debug.mp4'
    wavpath = '_wav2lip/speaker00es.wav'
    device = 'cuda'
    wav2lip_modelpath = '/home/ubuntu/quyennv/DeepFake/_wav2lip/wav2lip_gan.pth'
    # gfpgan_modelpath = '/home/ubuntu/Duy_test_folder/Retinaface_Mediapipe/GFPGAN/experiments/pretrained_models/GFPGANv1.3.pth'
    vqfr_modelpath  = '/home/ubuntu/Duy_test_folder/Retinaface_Mediapipe/VQFR/experiments/pretrained_models/VQFR_v2.pth'
    MASK_KP_IDS = [2,326, 423,425 ,411,416, 430, 431, 262, 428, 199, 208, 32, 211, 210,192, 187, 205 , 203, 97]

    # load all models
    # sfd_facedetector = face_detection.FaceAlignment(face_detection.LandmarksType._2D,
    #                                     flip_input=False, device=device)
    facemesh = init_face_mesh()
    wav2lip_model = load_model(wav2lip_modelpath, device)
    trackkpVideoFace = KalmanArray()
    trackkpVideoMount = KalmanArray()
    trackkpVideo_68 = KalmanArray()
    # model_Occ,to_tensor = load_FaceOcc()
    mobile_net_wav2lip, resnet_net_wav2lip = loadmodelface()
    # print(mel_chunks)

    # read video
    cap = cv2.VideoCapture(vidpath)
    cap_ori =  cv2.VideoCapture(frame_path)
    fps = int(np.round(cap.get(cv2.CAP_PROP_FPS)))
    print("FPS: ",fps)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    encoder_video = ffmpeg_encoder(savepath, fps, width, height)
    # List_speak_frame = load_speech_timestamp(fps)

    mel_chunks = get_mel_chunks(wavpath,fps)
    # print(" - ", mel_chunks.type)
    # json_object = json.dumps(mel_chunks)
    # with open("/home/ubuntu/Duy_test_folder/SP_Speechdetection/content/SpeakerDetection/Mel_chunks_23Nov_DrES_60fps.json", "w") as outfile:
    #     outfile.write(json_object)



    minute_start =0
    second_start =18
    minute_stop =0
    second_stop =45
    frame_start = int(minute_start*60*fps+second_start*fps)
    frame_stop = int(minute_stop*60*fps+second_stop*fps)
    # frame_start = 500
    # frame_stop = 1200
    total_output_frames =  min(total_frames, len(mel_chunks))
    count_frame = 0

    for i in tqdm(range(total_output_frames)):
        count_frame += 1
        ret, frame = cap.read()
        ret2,frame_ori = cap_ori.read()
        if not ret2:
            break
        mel_chunk = mel_chunks[i]
        # mel_chunk = np.array(mel_chunks[str(i)])
        if count_frame <= frame_start:
            continue
        elif count_frame > frame_stop:
            break
        # elif count_frame == 10023:
        elif count_frame > frame_start and count_frame <= frame_stop:

            # ret, frame = cap.read()
            # if not ret:
            #     break
            # print("Make ancoder done")
            # cv2.putText(frame, text='Fr:'+str(count_frame), org=(100, 40), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1.1, color=(0, 255, 0),thickness=2)
            # if not count_frame in List_speak_frame:
            #     write_frame(frame_ori,encoder_video)
            # else:
            res = lipsync_one_frame(frame,frame_ori, mel_chunk, facemesh, MASK_KP_IDS,\
              wav2lip_model, mobile_net_wav2lip, resnet_net_wav2lip, device,\
              trackkpVideoFace,trackkpVideoMount,trackkpVideo_68)
            write_frame(res,encoder_video)
        # if count_frame == 1000:
        #     break
    # del all models
    del  facemesh, wav2lip_model, mel_chunks
    # release
    cap.release()
    cap_ori.release()
    encoder_video.stdin.flush()
    encoder_video.stdin.close()

if __name__=='__main__':
    main()
