import cv2
import mediapipe as mp
import os
from tqdm import tqdm
import numpy as np
from PIL import Image
import ffmpeg
import subprocess
from pipeline_mobile_resnet import loadmodelface, detection_face
from color_transfer import color_hist_match,linear_color_transfer
from common import normalize_channels
import torch
from torchvision import transforms as TF_s
import json
from collections import OrderedDict
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.1)
config=tf.compat.v1.ConfigProto(gpu_options=gpu_options)
session = tf.compat.v1.Session(config=config)

from GFPGAN.gfpgan.utils import GFPGANer
import segmentation_models_pytorch as smp
from yaw_pitch_roll import facePose
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
    intersection = torch.count_nonzero(torch.logical_and(mask1, mask2))
    iou = intersection / (mask1_area + mask2_area - intersection)
    return iou.numpy()

class KalmanArray(object):
    def __init__(self):
        self.kflist = []
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

    def getpoints(self, kps):
        orginmask = np.zeros((self.h,self.w),dtype=np.float32)
        orginmask = cv2.fillConvexPoly(orginmask, np.array(kps[:-18], np.int32), 1)
        kps_o = kps.copy()
        for i in range(len(kps)):
            intpoint = np.array([np.float32(kps[i][0]), np.float32(kps[i][1])], np.float32)
            tmp = self.kflist[i].getpoint(intpoint)
            kps[i] = (tmp[0][0], tmp[1][0])

        newmask = np.zeros((self.h,self.w),dtype=np.float32)
        newmask = cv2.fillConvexPoly(newmask, np.array(kps[:-18], np.int32), 1)
        val = binaryMaskIOU_(torch.from_numpy(orginmask), torch.from_numpy(newmask))
        if val < 0.9:
            del self.kflist[:]
            self.setpoints(kps_o,self.w, self.h)
            return kps_o

        return kps

def get_face_by_RetinaFace(facea, inimg, mobile_net, resnet_net, device, kf = None,kf_driving = None):
    h,w = inimg.shape[:2]
    count_loss_detect = 0
    padding_ratio = [0.3,0.1]
    detect_fail = False
    previous_model = 0
    crop_image = None
    for r_idx in range(len(padding_ratio)):
        l_coordinate, detected_face = detection_face(mobile_net,resnet_net, inimg, device,padding_ratio[r_idx])
        if not detected_face and r_idx == len(padding_ratio)-1:
            count_loss_detect = 1
            detect_fail = True
            return None, None, None,None,detect_fail,count_loss_detect,previous_model,None
        elif not detected_face and r_idx < len(padding_ratio)-1:
          previous_model =  1
          continue
        face_landmarks = None
        bbox = None

        crop_images_coors = None
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
            return None, None, None, crop_images_coors, detect_fail,count_loss_detect,previous_model,None
        elif not face_landmarks and r_idx < len(padding_ratio)-1 :
            continue

        bbox_w = bbox[2] - bbox[0]
        bbox_h = bbox[3] - bbox[1]

        listpointLocal = []
        for i in range(len(FACEMESH_bigmask)):
            idx = FACEMESH_bigmask[i]
            x = face_landmarks.landmark[idx].x
            y = face_landmarks.landmark[idx].y
            listpointLocal.append((x, y))

        listpoint = []

        for i in range(len(listpointLocal)):
            x = listpointLocal[i][0]
            y = listpointLocal[i][1]

            realx = x * bbox_w + bbox[0]
            realy = y * bbox_h + bbox[1]
            listpoint.append((realx, realy))

        # if kf is not None:
        #     if kf.noneArray():
        #         kf.setpoints(listpoint, w, h)

        #     else:
        #         listpoint = kf.getpoints(listpoint)

        srcpts = np.array(listpoint, np.int32)
        srcpts = srcpts.reshape(-1,1,2)

        listpoint2 = []
        for i in range(len(FACEMESH_lips_2)):
            idx = FACEMESH_lips_2[i]
            x = face_landmarks.landmark[idx].x
            y = face_landmarks.landmark[idx].y

            realx = x * bbox_w + bbox[0]
            realy = y * bbox_h + bbox[1]

            idx_out = FACEMESH_lips_3[i]
            x_out = face_landmarks.landmark[idx_out].x
            y_out = face_landmarks.landmark[idx_out].y

            realx_out = x_out * bbox_w + bbox[0]
            realy_out = y_out * bbox_h + bbox[1]

            listpoint2.append(((realx+realx_out)/2, (realy+realy_out)//2))
        # if kf_driving is not None:
        #     if kf_driving.noneArray():
        #         kf_driving.setpoints(listpoint2, w, h)
        #     else:
        #         listpoint2 = kf_driving.getpoints(listpoint2)
        srcpts2 = np.array(listpoint2, np.int32)

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
        return srcpts, srcpts2, crop_images_coors, detect_fail,count_loss_detect,previous_model,face_angle

def get_face(facea, inimg, kf=None, kfMount=None, iswide = False):
    listpoint = []
    h,w = inimg.shape[:2]
    results = facea.process(cv2.cvtColor(inimg, cv2.COLOR_BGR2RGB))
    if not results.multi_face_landmarks:
        return None, None

    face_landmarks = results.multi_face_landmarks[0]

    listpointLocal = []
    for i in range(len(FACEMESH_bigmask)):
        idx = FACEMESH_bigmask[i]
        x = face_landmarks.landmark[idx].x
        y = face_landmarks.landmark[idx].y
        listpointLocal.append((x, y))


    listpoint = []

    for i in range(len(listpointLocal)):
        x = listpointLocal[i][0]
        y = listpointLocal[i][1]

        realx = x * w
        realy = y * h
        listpoint.append((realx, realy))

    # if kf is not None:
    #     if kf.noneArray():
    #         kf.setpoints(listpoint, w, h)

    #     else:
    #         listpoint = kf.getpoints(listpoint)

    srcpts = np.array(listpoint, np.int32)
    srcpts = srcpts.reshape(-1,1,2)

    listpoint2 = []
    for i in range(len(FACEMESH_lips_2)):
        idx = FACEMESH_lips_2[i]
        x = face_landmarks.landmark[idx].x
        y = face_landmarks.landmark[idx].y

        realx = x * w
        realy = y * h

        idx_out = FACEMESH_lips_3[i]
        x_out = face_landmarks.landmark[idx_out].x
        y_out = face_landmarks.landmark[idx_out].y

        realx_out = x_out * w
        realy_out = y_out * h
        listpoint2.append(((realx+realx_out)//2, (realy+realy_out)//2))
    # if kfMount is not None:
    #     if kfMount.noneArray():
    #         kfMount.setpoints(listpoint2, w, h)
    #     else:
    #         listpoint2 = kfMount.getpoints(listpoint2)

    srcpts2 = np.array(listpoint2, np.int32)
    srcpts2 = srcpts2.reshape(-1,1,2)

    return srcpts, srcpts2

def getKeypointByMediapipe(videoimg,face_mesh_256, faceimg, kfVF, kfVM, kfFF, kfFM, mobile_net,resnet_net,device):
    videopts, videopts_out, video_crops_coors,detect_fail,count_loss_detect_Retina,previous_model,face_angle = get_face_by_RetinaFace(face_mesh_256, videoimg, mobile_net, resnet_net, device, kfVF, kfVM)

    if videopts is None:
        return None, None, None, None, None,video_crops_coors,detect_fail,None
    facepts, facepts_out = get_face(face_mesh_256, faceimg, kfFF, None)
    return videopts, videopts_out, facepts, facepts_out, video_crops_coors,detect_fail,face_angle

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
            .global_args("-nostats"),
            overwrite_output=True,
        ),
        stdin=subprocess.PIPE,
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
    center_ = (int((topx+bottomx)/2),int((bottomy+topy)/2))

    return topy, topx, bottomy, bottomx, center_

def fillhole(input_image):
    im_flood_fill = input_image.copy()
    h, w = input_image.shape[:2]
    masktmp = np.zeros((h + 2, w + 2), np.uint8)
    im_flood_fill = im_flood_fill.astype("uint8")
    cv2.floodFill(im_flood_fill, masktmp, (0, 0), 255)
    im_flood_fill_inv = cv2.bitwise_not(im_flood_fill)
    img_out = input_image | im_flood_fill_inv
    return img_out

def find_ratio_intersection(videoimg,crops_coors,videopts_out,FaceOcc_only = None,img_video_mask_mount=None):
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
    if FaceOcc_only is True:
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
        img_mouth_mask_occ = img_mouth_mask_occ*var_middle_occ*255
        img_mouth_mask_occ = fillhole(img_mouth_mask_occ)
        #cal ratio between 2 mouth mask Mediapipe and FaceOcc
        img_video_mask_mount = np.atleast_3d(img_video_mask_mount).astype(np.float) / 255.
        img_video_mask_mount[img_video_mask_mount != 1] = 0

        img_mouth_mask_occ = np.atleast_3d(img_mouth_mask_occ).astype(np.float) / 255.
        img_mouth_mask_occ[img_mouth_mask_occ != 1] = 0
        newval = len(img_mouth_mask_occ[img_mouth_mask_occ > 0])/len(img_video_mask_mount[img_video_mask_mount >0])

        return newval

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

    yaw, pitch, roll = facePose(posePoint[0], posePoint[1], posePoint[2], posePoint[3], posePoint[4])
    return (yaw, pitch, roll)

def add_image_by_mask(img1, img2, mask_):
    """
    paste masked area from img1 to img2.
    img1 and img2 must have same shape.
    """
    mask_not = cv2.bitwise_not(mask_)
    img2_no_mask = cv2.bitwise_and(img2, img2, mask=mask_not)
    img1_mask_only = cv2.bitwise_and(img1, img1, mask=mask_)
    return cv2.add(img2_no_mask, img1_mask_only)

if __name__ == "__main__":
    # # load facemesh streamer's mask
    # with open("/home/ubuntu/Duy_test_folder/Retinaface_Mediapipe/config_lipsync.json",'r') as f_lips:
    #     list_keypoint = json.load(f_lips)
    # streamer = "Dr"
    # FACEMESH_lips_2 = list_keypoint[streamer]["FACEMESH_lips_2"]
    # FACEMESH_lips_3 = list_keypoint[streamer]["FACEMESH_lips_3"]
    # FACEMESH_pose_estimation = list_keypoint[streamer]["FACEMESH_pose_estimation"]
    # FACEMESH_bigmask = list_keypoint[streamer]["FACEMESH_bigmask"]

    # facemesh streamer's mask
    FACEMESH_lips_2 = [185, 40, 39, 37, 0, 267, 269, 270, 409, 377, 152, 148]
    FACEMESH_lips_3 = [185, 40, 39, 37, 0, 267, 269, 270, 409, 377, 152, 148]
    FACEMESH_pose_estimation = [34,264,168,33, 263]
    FACEMESH_bigmask = [185, 40, 39, 37, 0, 267, 269, 270, 409, 377, 152, 148]

    # load smp model for segmentation
    ENCODER = 'resnet18'
    ENCODER_WEIGHTS = 'imagenet'
    CLASSES = 1
    ATTENTION = None
    ACTIVATION = None
    DEVICE = 'cuda:0'
    to_tensor = TF_s.ToTensor()
    model_Occ = smp.Unet(encoder_name=ENCODER,
                     encoder_weights=ENCODER_WEIGHTS,
                     classes=CLASSES,
                     activation=ACTIVATION)
    weights = torch.load('/home/ubuntu/Duy_test_folder/FaceExtraction/FaceOcc/FaceOcc/epoch_16_best.ckpt')
    new_weights = OrderedDict()
    for key in weights.keys():
        new_key = '.'.join(key.split('.')[1:])
        new_weights[new_key] = weights[key]
    model_Occ.load_state_dict(new_weights)
    model_Occ.to(DEVICE)
    model_Occ.eval()

    # load mobilenet, resnet
    mobile_net, resnet_net = loadmodelface()
    device = torch.device("cuda")

    # load facemesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh_256 = mp_face_mesh.FaceMesh(min_detection_confidence=0.2,
                                          refine_landmarks=True,
                                          max_num_faces=1,
                                          static_image_mode=True)


    FACE_PATH = '/home/ubuntu/quyennv/DeepFake/videos/fomm_8p22_gfpgan.mp4'
    FRAME_PATH = '/home/ubuntu/quyennv/DeepFake/videos/in_8p22.mp4'
    output_path = '/home/ubuntu/quyennv/DeepFake/videos/cmt.mp4'

    capFrame = cv2.VideoCapture(FRAME_PATH)
    capFace = cv2.VideoCapture(FACE_PATH)
    fps = capFrame.get(cv2.CAP_PROP_FPS)
    width_  = capFrame.get(cv2.CAP_PROP_FRAME_WIDTH)
    height_ = capFrame.get(cv2.CAP_PROP_FRAME_HEIGHT)
    encoder_video = ffmpeg_encoder(output_path, fps, int(width_), int(height_))
    totalF = int(capFrame.get(cv2.CAP_PROP_FRAME_COUNT))


    trackkpVideoFace = KalmanArray()
    trackkpVideoMount = KalmanArray()
    trackkpFaceFace = KalmanArray()
    trackkpFaceMount = KalmanArray()

    # get list of speak frames
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
    check_not_speech = 0
    speak_check = 0

    # dynamic variable
    count_frame = 0
    num_frame_loss = 0
    num_mouth_frame_loss = 0
    block_frames_count = 0
    original_frames = []
    drawed_frames = []
    select_block = []
    write_video = None

    # start and end of lipsync
    fps_per_second = 30
    num_frame_threshold = 1
    minute_start = 0
    second_start = 0
    minute_stop = 10
    second_stop = 10
    frame_start = int(minute_start*60*fps+second_start*fps)
    frame_stop = int(minute_stop*60*fps+second_stop*fps)
    total_f = frame_stop-frame_start
    pbar = tqdm(total=total_f)

    while capFrame.isOpened():
        okay1  , videoimg = capFrame.read()
        okay2 , faceimg = capFace.read()
        videoimg_copy = videoimg.copy()
        #cal frame for block_frames util reach num_frame_threshold
        count_frame = count_frame+1
        pbar.update(1)
        if count_frame <= frame_start:
            continue
        elif count_frame > frame_stop+1:
            break
        elif count_frame > frame_start and count_frame <= frame_stop+1:

            if not okay1 or not okay2:
                print('Cant read the video , Exit!')
                break

            # if not count_frame in List_speak_frame:
            #     check_not_speech +=1
            #     if len(drawed_frames)>0:
            #         for images in original_frames:
            #             image_draw = cv2.cvtColor(images,cv2.COLOR_RGB2BGR)
            #             imageout = Image.fromarray(np.uint8(image_draw))
            #             encoder_video.stdin.write(imageout.tobytes())
            #
            #         #reset dynamic val
            #         num_frame_loss = 0
            #         original_frames = []
            #         select_block = []
            #         drawed_frames = []
            #         block_frames_count = 0
            #         num_mouth_frame_loss = 0
            #         check_not_speech =0
            #         write_video = False
            #         speak_check = 0
            #
            #     image_draw = cv2.cvtColor(videoimg_copy,cv2.COLOR_RGB2BGR)
            #     imageout = Image.fromarray(np.uint8(image_draw))
            #     encoder_video.stdin.write(imageout.tobytes())
            #
            # else:
            block_frames_count += 1
            speak_check +=1

            video_h,video_w, = videoimg.shape[:2]

            videopts, videopts_out, facepts, facepts_out, crops_coors,detect_fail,face_angle = getKeypointByMediapipe(
                videoimg, face_mesh_256, faceimg, trackkpVideoFace, trackkpVideoMount, trackkpFaceFace, trackkpFaceMount, mobile_net,resnet_net,device)
            if videopts is None or facepts is None :
                if last_lipsynced_fr ==count_frame-1:
                    output_main = previous_output
                else:
                    output_main = videoimg_copy
                num_frame_loss +=1
                original_frames.append(videoimg_copy)
                drawed_frames.append(output_main)
            else:
                try:
                    #get mask of mediapipe
                    M_face, _ = cv2.findHomography(facepts, videopts)
                    face_img = cv2.warpPerspective(faceimg,M_face,(video_w, video_h))

                    facepts_out = cv2.perspectiveTransform(facepts_out.astype(np.float32), M_face).astype(np.int32)
                    M_mount, _ = cv2.findHomography(facepts_out, videopts_out)
                    face_img = cv2.warpPerspective(face_img,M_mount,(video_w, video_h), cv2.INTER_LINEAR)

                    img_video_mask_mount = np.zeros((video_h,video_w), np.uint8)
                    cv2.fillPoly(img_video_mask_mount, pts =[videopts_out], color=(255,255,255))
                    img_video_mask_mount_bk = np.zeros((video_h,video_w), np.uint8)
                    cv2.fillPoly(img_video_mask_mount_bk, pts =[videopts_out], color=(255,255,255))

                    img_video_mask_face = np.zeros((video_h,video_w), np.uint8)
                    cv2.fillPoly(img_video_mask_face, pts =[videopts], color=(255,255,255))
                    img_video_mask_face = cv2.bitwise_or(img_video_mask_face, img_video_mask_mount)
                    topy, topx, bottomy, bottomx, center_face = mask2box(img_video_mask_face)

                    result = add_image_by_mask(face_img, videoimg, img_video_mask_face)

                    alpha_0 = 0.95
                    result[topy:bottomy, topx:bottomx] = cv2.addWeighted(result[topy:bottomy, topx:bottomx],\
                    alpha_0, videoimg[topy:bottomy, topx:bottomx], 1-alpha_0, 0.0)

                    img_bgr_uint8_1 = normalize_channels(result[topy:bottomy, topx:bottomx], 3)
                    img_bgr_1 = img_bgr_uint8_1.astype(np.float32) / 255.0
                    img_bgr_1 = np.clip(img_bgr_1, 0, 1)
                    img_bgr_uint8_2 = normalize_channels(videoimg[topy:bottomy, topx:bottomx], 3)
                    img_bgr_2 = img_bgr_uint8_2.astype(np.float32) / 255.0
                    img_bgr_2 = np.clip(img_bgr_2, 0, 1)

                    result_new = linear_color_transfer(img_bgr_1, img_bgr_2)
                    final_img = color_hist_match(result_new, img_bgr_2, 255).astype(dtype=np.float32)

                    result[topy:bottomy, topx:bottomx] = blursharpen((final_img*255).astype(np.uint8), 1, 5, 0.5)
                    # addWeighted to results after match color
                    alpha_0 = 0.95
                    result[topy:bottomy, topx:bottomx] = cv2.addWeighted(result[topy:bottomy, topx:bottomx],\
                    alpha_0, videoimg[topy:bottomy, topx:bottomx], 1-alpha_0, 0.0)

                    img_video_mask_mount = cv2.dilate(img_video_mask_mount, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)), iterations = 1 )
                    img_video_mask_face = cv2.erode(img_video_mask_face, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)), iterations = 1 )
                    img_video_mask_mount_error = cv2.bitwise_xor(img_video_mask_mount, img_video_mask_face)
                    img_video_mask_mount = img_video_mask_mount - img_video_mask_mount_error

                    result = add_image_by_mask(result, videoimg, img_video_mask_mount)
                    m_topy, m_topx, m_bottomy, m_bottomx, center_mount = mask2box(img_video_mask_mount)
                    output_main = cv2.seamlessClone(result, videoimg, img_video_mask_mount, center_mount, cv2.NORMAL_CLONE)

                    # cv2.rectangle(output_main, (m_topx, m_topy), (m_bottomx,m_bottomy), (0,255,0), 4)
                    if m_topy <2 or m_topx <2 or m_bottomy > int(height_)-2 or m_bottomx > int(width_)-2 :
                        drawed_frames.append(videoimg_copy)
                        original_frames.append(videoimg_copy)
                        num_mouth_frame_loss += 1
                    else:
                        #Cal mask by FaceOcc
                        newval = find_ratio_intersection(videoimg,crops_coors,videopts,\
                        FaceOcc_only = False,img_video_mask_mount = img_video_mask_mount_bk)
                        # cv2.polylines(cropimage, [videopts_out],True,color=(0,0,255),thickness=3)
                        # print("Face angle ", face_angle)
                        if newval > 0.8902 or (newval > 0.83 and face_angle != 0):
                            # print("Mask good!")
                            drawed_frames.append(output_main)
                            original_frames.append(videoimg_copy)
                            last_lipsynced_fr = count_frame
                            previous_output = output_main.copy()
                        else:
                            num_mouth_frame_loss += 1
                            drawed_frames.append(videoimg_copy)
                            original_frames.append(videoimg_copy)
                except Exception:
                    print("Failed to try")
                    drawed_frames.append(videoimg_copy)
                    original_frames.append(videoimg_copy)

            if num_frame_loss > num_frame_threshold or num_mouth_frame_loss > num_frame_threshold:# or check_not_speech > num_frame_threshold:
                select_block = original_frames
                check_bl = "Normally"
                write_video = True
            elif block_frames_count >= fps_per_second:
                select_block = drawed_frames
                check_bl = "Lipsynced"
                write_video = True
            if write_video:
                for images in select_block:
                    image_draw = cv2.cvtColor(images,cv2.COLOR_RGB2BGR)
                    imageout = Image.fromarray(np.uint8(image_draw))
                    encoder_video.stdin.write(imageout.tobytes())

                #reset dynamic val
                num_frame_loss = 0
                original_frames = []
                select_block = []
                drawed_frames = []
                block_frames_count = 0
                num_mouth_frame_loss = 0
                check_not_speech =0
                write_video = False
                speak_check=0

    pbar.close()
    encoder_video.stdin.flush()
    encoder_video.stdin.close()
