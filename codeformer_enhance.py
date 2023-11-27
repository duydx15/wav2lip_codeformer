# Import library
import sys
import json
import os
import audio
import numpy as np
import torch
import cv2
from CodeFormer.load_codeformer import process_img,load_codeformer_model,warp_face_codeformer
import segmentation_models_pytorch as smp
from color_transfer import color_transfer, color_hist_match,linear_color_transfer
from tqdm import tqdm
import math
import imutils
import subprocess
import mediapipe as mp
from pipeline_mobile_resnet_wav2lip import loadmodelface, detection_face_wav2lips

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

def init_face_mesh():
    mp_face_mesh = mp.solutions.face_mesh
    return mp_face_mesh.FaceMesh(min_detection_confidence=0.5)
    #static_image_mode=True, max_num_faces=1, refine_landmarks=True,

def get_keypoint_mouth(input_img,face_landmarks,bbox,kf_mouth=None,kf=None,kf_68=None):
    with open("/home/ubuntu/Duy_test_folder/Retinaface_Mediapipe/_wav2lip/config_lipsync_wav2lip.json",'r') as f_lips:
        list_keypoint = json.load(f_lips)
    streamer = "Dr"
    lips_pts = list_keypoint[streamer]["FACEMESH_lips_2_up"]
    lips_pts_small = list_keypoint[streamer]["FACEMESH_lips"]
    lips_pts_face = list_keypoint[streamer]["FACEMESH_bigmask"]
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

    point_face = np.array(listpoint, np.int32)
    point_face = point_face.reshape(-1,1,2)


    return point_mouth,point_face#,srcpts_beard,srcpts_beard_inner,srcpts_lips_small

def add_image_by_mask(img1, img2, mask_):
    mask_not = cv2.bitwise_not(mask_)
    img2_no_mask = cv2.bitwise_and(img2, img2, mask=mask_not)
    img1_mask_only = cv2.bitwise_and(img1, img1, mask=mask_)
    return cv2.add(img2_no_mask, img1_mask_only)

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



def main():

    device = 'cuda'
    facemesh = init_face_mesh()
     # Read image
    path_1 = "/home/ubuntu/Duy_test_folder/Retinaface_Mediapipe/Avatar_enhance/Cowboy_Doc_open2.png"
    image_1 = cv2.imread(path_1,flags=cv2.IMREAD_UNCHANGED)
    image_tmp_1 = image_1[:,:,:3]
    path = "/home/ubuntu/Duy_test_folder/Retinaface_Mediapipe/Avatar_enhance/Cowboy_Doc_open2_DFL.png"
    image_ori = cv2.imread(path,flags=cv2.IMREAD_UNCHANGED)
    image_tmp = image_ori[:,:,:3]

    # Load CodeFormer model
    mobile_net, resnet_net= loadmodelface()
    codeformer,face_helper,bg_upsampler = load_codeformer_model(upscale=2,detection_model='retinaface_resnet50')

    frame_h,frame_w = image_ori.shape[:2]
    box, five_points,padding_box,_ = get_box_by_RetinaFace(image_tmp, mobile_net, resnet_net, device)
    if box==None:
        print("Not detect")
        return image_ori

    x1, y1, x2, y2 = box
    bbox =box
    size_box_face = np.abs((box[0]-box[2])*(box[1]-box[3]))

    box = [y1,y2,x1,x2]
    roi = image_tmp[y1:y2,x1:x2, :]
    roi_2 = image_tmp[padding_box[0][1]:padding_box[1][1], padding_box[0][0]:padding_box[1][0],:]

    landmark,face_landmarks = get_fm_landmark(roi_2, facemesh)
    if landmark==None or face_landmarks == None:
        print("Not detect landmark")
        return image_ori
    padding_box = [padding_box[0][1],padding_box[1][1], padding_box[0][0],padding_box[1][0]]
    videopts_out,videopts= get_keypoint_mouth(image_tmp,face_landmarks,padding_box)

    # Backup and run codeformer
    restored_img,affine_matrix = warp_face_codeformer(image_tmp,five_points,face_size=512,border_mode='constant')
    restored_img = process_img(restored_img,codeformer,face_helper,bg_upsampler, device,fidelity_weight=1.0,\
    upscale =1,has_aligned=True, draw_box = False,face_upsample = False,only_center_face = False)
    # #### Warp back
    inverse_affine = cv2.invertAffineTransform(affine_matrix)
    mask_codeformer_warp = 255 * np.ones((512,512), np.uint8)
    mask_codeformer_warp = cv2.warpAffine(mask_codeformer_warp, inverse_affine, (frame_w, frame_h))
    restored_img = cv2.warpAffine(restored_img, inverse_affine, (frame_w, frame_h))
    frame = merge_color(videopts_out,videopts,image_tmp_1,box,restored_img,0.95)
    image_1[:,:,:3] = frame
    cv2.imwrite("/home/ubuntu/Duy_test_folder/Retinaface_Mediapipe/Avatar_enhance/Cowboy_Doc_open2_DFL_code.png",image_1)

if __name__=='__main__':
    main()
