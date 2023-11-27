"""
"""
import sys
import json
import os
import audio
import numpy as np
import torch
from models import Wav2Lip
import cv2
from tqdm import tqdm
# import face_detection
import math
from gfp_gan_ import load_gfpgan_model
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

gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.2)
config=tf.compat.v1.ConfigProto(gpu_options=gpu_options)
session = tf.compat.v1.Session(config=config)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def get_mel_chunks(wav_path):
    """
    each mel chunk is corresponding with one frame
    """
    mel_step_size = 16
    fps = 30
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

def gen_lipsync_img(img, mel_chunk, model, box, device,gfpgan_model):
    """
    """
    resize_factor = 2.25
    # box = face_detect([frame], sfd_facedetector)[0]
    # print(img.shape[1]//resize_factor, img.shape[0]//resize_factor)
    img = cv2.resize(img, (int(img.shape[1]//resize_factor), int(img.shape[0]//resize_factor)))

    # frame = cv2.resize(frame, (1280,720))
    MODEL_INPUT_IMG_SIZE = 96
    # print(box.type)
    y_ori1, y_ori2, x_ori1, x_ori2 = box
    y1, y2, x1, x2 = (np.array(box)//resize_factor).astype(np.int32)


    roi = img[y1:y2, x1:x2, :]
    roi_cop = roi.copy()
    # roi = gfpgan_img(roi, gfpgan_model)
    # roi = cv2.flip(roi,1)
    cv2.imwrite("Face_crop_gfp.png",roi)
    roi = cv2.GaussianBlur(roi,(7,7),0)
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
    # pred = gfpgan_img(pred, gfpgan_model)
    pred = my_cv2_resize(pred, (x_ori2-x_ori1), (y_ori2-y_ori1))
    # pred = cv2.flip(pred,1)
    cv2.imwrite("Pred_crop_gfp.png",pred)
    # img_bgr_uint8_1 = normalize_channels(pred, 3)
    # img_bgr_1 = img_bgr_uint8_1.astype(np.float32) / 255.0
    # img_bgr_1 = np.clip(img_bgr_1, 0, 1)
    # img_bgr_uint8_2 = normalize_channels(roi_cop, 3)
    # img_bgr_2 = img_bgr_uint8_2.astype(np.float32) / 255.0
    # img_bgr_2 = np.clip(img_bgr_2, 0, 1)
    #
    # result_new = linear_color_transfer(img_bgr_1, img_bgr_2)
    # final_img = color_hist_match(result_new, img_bgr_2, 255).astype(dtype=np.float32)
    # pred = blursharpen((final_img*255).astype(np.uint8), 1, 5, 0.5)
    return pred

def gfpgan_img(img, gfpgan_model):
    """
    """
    _, _,restored_img = gfpgan_model.enhance(
    img, has_aligned=False, only_center_face=False, paste_back=True)
    return restored_img

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

def get_face_by_RetinaFace(inimg, mobile_net, resnet_net, device):

    h,w = inimg.shape[:2]
    padding_ratio = [0]
    for r_idx in range(len(padding_ratio)):
        l_coordinate, detected_face= detection_face_wav2lips(mobile_net,resnet_net, inimg, device,padding_ratio[r_idx])
        if not detected_face and r_idx == len(padding_ratio)-1:
            return None
        elif not detected_face and r_idx < len(padding_ratio)-1:
            continue
        bbox = None

        for i in range(len(l_coordinate)):
            topleft, bottomright = l_coordinate[i]
            curbox = []
            if bottomright[1] < h/3:
                continue
            curbox.append(topleft[0])
            curbox.append(topleft[1])
            curbox.append(bottomright[0])
            curbox.append(bottomright[1])

            x1, y1, x2, y2 = curbox
        if len(curbox) > 0:
            return y1, y2, x1, x2
        else:
            return None

with open("/home/ubuntu/Duy_test_folder/Retinaface_Mediapipe/_wav2lip/config_lipsync_wav2lip.json",'r') as f_lips:
	list_keypoint = json.load(f_lips)
streamer = "Dr"
lips_pts = list_keypoint[streamer]["FACEMESH_lips_2"]
lips_pts_small = list_keypoint[streamer]["FACEMESH_lips"]
lips_pts_face = list_keypoint[streamer]["FACEMESH_bigmask"]
lips_pts_beard = list_keypoint[streamer]["FACEMESH_beard"]
lips_pts_beard_inner = list_keypoint[streamer]["FACEMESH_beard_inner"]
def get_keypoint_mouth(face_landmarks,bbox):

    # lips_pts_face = list_keypoint[streamer]["FACEMESH_"]
    listpoint2 = []
    bbox_w = bbox[3] - bbox[2]
    bbox_h = bbox[1] - bbox[0]
    for i in range(len(lips_pts)):
        idx = lips_pts[i]
        x = face_landmarks.landmark[idx].x
        y = face_landmarks.landmark[idx].y

        realx = x * bbox_w + bbox[2]
        realy = y * bbox_h + bbox[0]
        listpoint2.append((realx,realy))
    srcpts2 = np.array(listpoint2, np.int32)

    listpoint = []
    for i in range(len(lips_pts_face)):
        idx = lips_pts_face[i]
        x = face_landmarks.landmark[idx].x
        y = face_landmarks.landmark[idx].y

        realx = x * bbox_w + bbox[2]
        realy = y * bbox_h + bbox[0]
        listpoint.append((realx, realy))
    srcpts = np.array(listpoint, np.int32)

    listpoint_beard = []
    for i in range(len(lips_pts_beard)):
        idx = lips_pts_beard[i]
        x = face_landmarks.landmark[idx].x
        y = face_landmarks.landmark[idx].y

        realx = x * bbox_w + bbox[2]
        realy = y * bbox_h + bbox[0]
        listpoint_beard.append((realx, realy))
    srcpts_beard = np.array(listpoint_beard, np.int32)

    listpoint_beard_inner = []
    for i in range(len(lips_pts_beard_inner)):
        idx = lips_pts_beard_inner[i]
        x = face_landmarks.landmark[idx].x
        y = face_landmarks.landmark[idx].y

        realx = x * bbox_w + bbox[2]
        realy = y * bbox_h + bbox[0]
        listpoint_beard_inner.append((realx, realy))
    srcpts_beard_inner = np.array(listpoint_beard_inner, np.int32)

    listpoint_small = []
    for i in range(len(lips_pts_small)):
        idx = lips_pts_small[i]
        x = face_landmarks.landmark[idx].x
        y = face_landmarks.landmark[idx].y

        realx = x * bbox_w + bbox[2]
        realy = y * bbox_h + bbox[0]
        listpoint_small.append((realx, realy))
    srcpts_lips_small = np.array(listpoint_small, np.int32)

    return srcpts2,srcpts,srcpts_beard,srcpts_beard_inner,srcpts_lips_small

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

def merge_color(videopts_out,videopts,videopts_beard,videopts_beard_inner,frame,box,pred,alpha,facemesh):
    # landmark_gan,face_landmarks_gan = get_fm_landmark(pred, facemesh)
    # if landmark_gan is None or face_landmarks_gan is None:
    #     return frame
    # ganpts_out,ganpts,ganpts_beard,ganpts_beard_inner,_ = get_keypoint_mouth(face_landmarks_gan,box)
    video_h,video_w = frame.shape[:2]
    frame_gan_raw = frame.copy()
    y1, y2, x1, x2 = box
    frame_gan_raw[y1:y2, x1:x2, :] = pred
    mask_mount =  np.zeros((video_h,video_w), np.uint8)
    cv2.fillPoly(mask_mount,pts = [videopts_out], color=(255,255,255))
    mask_face =  np.zeros((video_h,video_w), np.uint8)
    cv2.fillPoly(mask_face,pts = [videopts[:-13]], color=(255,255,255))
    mask_beard =  np.zeros((video_h,video_w), np.uint8)
    cv2.fillPoly(mask_beard,pts = [videopts_beard], color=(255,255,255))
    # mask_beard_inner =  np.zeros((video_h,video_w), np.uint8)
    # cv2.fillPoly(mask_beard_inner,pts = [videopts_beard_inner], color=(255,255,255))

    mask_face = cv2.bitwise_or(mask_face, mask_mount)
    topy, topx, bottomy, bottomx, center_face = mask2box(mask_face)
    # _,_,_,_, center_beard = mask2box(mask_beard)

    result = add_image_by_mask(frame_gan_raw, frame, mask_face)
    beard_tmp = cv2.addWeighted(frame, 0.8, result, 0.2, 0.0)
    beard_tmp_cop = beard_tmp

    # cv2.polylines(frame, [videopts_beard_inner],True,color=(0,255,0),thickness=1)
    # mask_beard[mask_beard>0]=255
    #warp image
    # beard_inner = videopts_beard_inner
    # M_beard,_ = cv2.findHomography(videopts_beard.reshape(-1,1,2),ganpts_beard)
    # beard_tmp = cv2.warpPerspective(beard_tmp,M_beard,(video_w, video_h))
    # mask_beard_inner = cv2.warpPerspective(mask_beard_inner,M_beard,(video_w, video_h),cv2.INTER_NEAREST)
    # videopts_beard_inner = cv2.perspectiveTransform(videopts_beard_inner.reshape(-1,1,2).astype(np.float32), M_beard).astype(np.int32)
    #
    # mask_beard_inner = cv2.GaussianBlur(mask_beard_inner, (15,15), 0)
    # result = blend_images_using_mask(beard_tmp,result,mask_beard_inner)
    mask_beard = cv2.GaussianBlur(mask_beard, (15,15), 0)
    result = blend_images_using_mask(beard_tmp_cop,result,mask_beard)

    # alpha_0 = alpha
    # result[topy:bottomy, topx:bottomx] = cv2.addWeighted(result[topy:bottomy, topx:bottomx],\
    # alpha_0, frame[topy:bottomy, topx:bottomx], 1-alpha_0, 0.0)

    img_bgr_uint8_1 = normalize_channels(result[topy:bottomy, topx:bottomx], 3)
    img_bgr_1 = img_bgr_uint8_1.astype(np.float32) / 255.0
    img_bgr_1 = np.clip(img_bgr_1, 0, 1)
    img_bgr_uint8_2 = normalize_channels(frame[topy:bottomy, topx:bottomx], 3)
    img_bgr_2 = img_bgr_uint8_2.astype(np.float32) / 255.0
    img_bgr_2 = np.clip(img_bgr_2, 0, 1)

    result_new = linear_color_transfer(img_bgr_1, img_bgr_2)
    final_img = color_hist_match(result_new, img_bgr_2, 255).astype(dtype=np.float32)
    result[topy:bottomy, topx:bottomx] = blursharpen((final_img*255).astype(np.uint8), 1, 5, 0.5)

    #addWeighted to results after match color
    alpha_0 = alpha
    result[topy:bottomy, topx:bottomx] = cv2.addWeighted(result[topy:bottomy, topx:bottomx],\
    alpha_0, frame[topy:bottomy, topx:bottomx], 1-alpha_0, 0.0)
    # print("Max mask mouth:",np.unique(mask_mount))
    # m_topy, m_topx, m_bottomy,/ m_bottomx, center_mount = mask2box(mask_mount)
    mask_mount = cv2.dilate(mask_mount, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)), iterations = 1 )
    mask_face_2 = cv2.erode(mask_face, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)), iterations = 1 )
    mask_mount_error = cv2.bitwise_xor(mask_mount, mask_face_2)
    mask_mount = mask_mount - mask_mount_error
    m_topy, m_topx, m_bottomy, m_bottomx,center_mount = mask2box(mask_mount)
    if m_topy <5 or m_topx <5 or m_bottomy > video_h-5 or m_bottomx > video_w-5:
        return frame
    result = add_image_by_mask(result, frame, mask_mount)
    output_main = cv2.seamlessClone(result, frame, mask_mount, center_mount, cv2.NORMAL_CLONE)

    # cv2.polylines(output_main, [ganpts_out],True,color=(0,255,0),thickness=2)
    # cv2.polylines(output_main, [ganpts_beard_inner],True,color=(0,255,0),thickness=1)
    # cv2.polylines(output_main, [videopts_beard_inner],True,color=(255,0,0),thickness=2)
    # cv2.polylines(output_main, [beard_inner],True,color=(0,0,255),thickness=2)
    # cv2.polylines(output_main, [videopts_out],True,color=(0,0,255),thickness=2)
    # cv2.imwrite("results_finetune_beard.png",output_main)
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

def lipsync_one_frame(frame, mel_chunk, facemesh, MASK_KP_IDS,  wav2lip_model, gfpgan_model,mobile_net, resnet_net, device,to_tensor,model_Occ):
    """
    """
    # resize_factor = 2
    # # box = face_detect([frame], sfd_facedetector)[0]
    frame_cop =  frame.copy()
    # # frame = cv2.resize(frame, (frame.shape[1]//resize_factor, frame.shape[0]//resize_factor))
    # frame = cv2.resize(frame, (1280,720))
    box = get_face_by_RetinaFace(frame, mobile_net, resnet_net, device)
    if box==None:
        return frame_cop

    y1, y2, x1, x2 = box
    roi = frame[y1:y2,x1:x2, :]
    landmark,face_landmarks = get_fm_landmark(roi, facemesh)
    if landmark==None or face_landmarks == None:
        return frame
    videopts_out,videopts,videopts_beard,videopts_beard_inner,videopts_lips = get_keypoint_mouth(face_landmarks,box)
    # newval,_,_ = find_ratio_intersection(frame,box,videopts_out,to_tensor,model_Occ)
    # if newval < 0.83:
    #     return frame

    pred = gen_lipsync_img(frame, mel_chunk, wav2lip_model, box, device,gfpgan_model)

    #Adjust pred quality
    # alpha = 0.6
    # pred = cv2.addWeighted(pred,alpha,roi,1-alpha,0.0)

    restored_img = gfpgan_img(pred, gfpgan_model)
    # pred = my_cv2_resize(restored_img, (x2-x1), (y2-y1))
    # mask, center = get_fm_mask(roi, MASK_KP_IDS, landmark)
    # roi = cv2.seamlessClone(restored_img, roi, mask, center, cv2.NORMAL_CLONE)
    frame[y1:y2, x1:x2, :] = restored_img
    cv2.imwrite("Frame.png",frame)

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

def write_frame(images,encoder_video):
    image_draw = cv2.cvtColor(images,cv2.COLOR_RGB2BGR)
    imageout = Image.fromarray(np.uint8(image_draw))
    encoder_video.stdin.write(imageout.tobytes())

def main():
    vidpath = '/home/ubuntu/Duy_test_folder/Retinaface_Mediapipe/Dr_video/DrDisrespect-Falls-in-Love-with-Warzone-again-thanks-to-new-Game-Mode-30FPS.mp4'
    # vidpath = '/home/ubuntu/Duy_test_folder/Retinaface_Mediapipe/_wav2lip/test_0p25_resize.mp4'
    savepath = 'Wav2lip_7Sep_debug.mp4'
    wavpath = 'test_5p19.wav'
    device = 'cuda'
    wav2lip_modelpath = '/home/ubuntu/quyennv/DeepFake/_wav2lip/wav2lip_gan.pth'
    gfpgan_modelpath = '/home/ubuntu/Duy_test_folder/Retinaface_Mediapipe/GFPGAN/experiments/pretrained_models/GFPGANv1.3.pth'
    MASK_KP_IDS = [2,326, 423,425 ,411,416, 430, 431, 262, 428, 199, 208, 32, 211, 210,192, 187, 205 , 203, 97]

    # load all models
    # sfd_facedetector = face_detection.FaceAlignment(face_detection.LandmarksType._2D,
    #                                     flip_input=False, device=device)
    facemesh = init_face_mesh()
    wav2lip_model = load_model(wav2lip_modelpath, device)
    gfpgan_model = load_gfpgan_model(gfpgan_modelpath, device)
    mobile_net, resnet_net = loadmodelface()
    model_Occ,to_tensor = load_FaceOcc()
    with open("/home/ubuntu/Duy_test_folder/SP_Speechdetection/content/SpeakerDetection/Mel_chunks_29Aug_DrES.json", "r") as outfile:
        mel_chunks = json.load(outfile)
    # mel_chunks = get_mel_chunks(wavpath)
    # read video
    cap = cv2.VideoCapture(vidpath)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    encoder_video = ffmpeg_encoder(savepath, fps, width, height)
    List_speak_frame = load_speech_timestamp(fps)

    minute_start = 8
    second_start = 24
    minute_stop =8
    second_stop =35
    frame_start = int(minute_start*60*fps+second_start*fps)
    frame_stop = int(minute_stop*60*fps+second_stop*fps)
    frame_start = 15319
    # frame_stop = int(minute_stop*60*fps+second_stop*fps)
    total_output_frames = 16092 #min(total_frames, len(mel_chunks))
    count_frame = 0
    for i in tqdm(range(total_output_frames)):
        count_frame += 1
        ret, frame = cap.read()
        # if not ret:
        #     break
        mel_chunk = np.array(mel_chunks[str(i)])
        if count_frame <= frame_start:
            continue
        elif count_frame > frame_stop:
            break
        # elif count_frame == 10023:
        elif count_frame > frame_start and count_frame <= frame_stop:
            # ret, frame = cap.read()
            cv2.putText(frame, text='Fr:'+str(count_frame), org=(100, 40), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1.1, color=(0, 255, 0),thickness=2)
            if not ret:
                break
            # cv2.putText(frame, text='Fr:'+str(count_frame), org=(100, 40), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1.1, color=(0, 255, 0),thickness=2)
            if not count_frame in List_speak_frame:
                write_frame(frame,encoder_video)
            else:
                res = lipsync_one_frame(frame, mel_chunk, facemesh, MASK_KP_IDS,  wav2lip_model, gfpgan_model,mobile_net, resnet_net, device,to_tensor,model_Occ)
                write_frame(res,encoder_video)

        if count_frame == 15320:
            break
    # del all models
    del facemesh, wav2lip_model, gfpgan_model, mel_chunks
    # release
    cap.release()
    encoder_video.stdin.flush()
    encoder_video.stdin.close()

if __name__=='__main__':
    main()
