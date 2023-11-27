import sys
sys.path.append('/home/ubuntu/quyennv/DeepFaceLab_Linux/DeepFaceLab')

from pathlib import Path
from core.leras import nn
from core.leras.device import Devices
from facelib import FaceType, LandmarksProcessor, XSegNet
import cv2
import os
import numpy as np
import ffmpeg
import subprocess
from tqdm import tqdm
from PIL import Image
from Retinaface_Mediapipe.common import  normalize_channels
from Retinaface_Mediapipe.pipeline_mobile_resnet import loadmodelface, detection_face
from skimage import measure
LOGURU_FFMPEG_LOGLEVELS = {
    "trace": "trace",
    "debug": "debug",
    "info": "info",
    "success": "info",
    "warning": "warning",
    "error": "error",
    "critical": "fatal",
}

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

def fillhole(input_image):
    # input_image = 255 - input_image
    labels_mask = measure.label(input_image)
    regions = measure.regionprops(labels_mask)
    regions.sort(key=lambda x: x.area, reverse=True)
    if len(regions) > 1:
        for rg in regions[1:]:
            labels_mask[rg.coords[:,0], rg.coords[:,1]] = 0
    labels_mask[labels_mask!=0] = 1
    input_image = labels_mask

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

def write_frame(images,encoder_video):
    image_draw = cv2.cvtColor(images,cv2.COLOR_RGB2BGR)
    imageout = Image.fromarray(np.uint8(image_draw))
    encoder_video.stdin.write(imageout.tobytes())

def my_cv2_resize(img, w, h):
    if img.size > w*h:
        return cv2.resize(img, (w, h), cv2.INTER_AREA)
    else:
        return cv2.resize(img, (w, h), cv2.INTER_CUBIC)

def init_XSeg(model_path, device='cpu'):
    """
    from pathlib import Path
    ******DeepFaceLab*******
    from core.leras import nn
    from facelib import XSegNet
    from core.leras.device import Devices
    """
    if device=='cpu':
        Devices.initialize_main_env()
        device_config = nn.DeviceConfig.CPU()
        nn.initialize(device_config)
    else:
        Devices.initialize_main_env()
        device_config = nn.DeviceConfig.GPUIndexes([0]) # change GPU index here
        nn.initialize(device_config)

    model_path = Path(model_path)
    xseg = XSegNet(name='XSeg',
                    load_weights=True,
                    weights_file_root=model_path,
                    data_format=nn.data_format,
                    raise_on_no_model_files=True)
    return xseg

def retinaface_check(frame):

    h,w = frame.shape[:2]
    count_loss_detect = 0
    padding_ratio = 0.3
    detect_fail = False
    previous_model = 0
    crop_image = None
    padding_ratio = 0.2
    detected_face, l_coordinate = detection_face(mobile_net,resnet_net, frame, device,padding_ratio)
    # block_check.append(detected_face)
    if not detection_face:
        return None,None
    crop_images_coors = None
    max_box = 0
    for i in range(len(l_coordinate)):
        topleft, bottomright = l_coordinate[i]
        if bottomright[1] < h/3:
            continue
        area_box = np.abs((topleft[0]-bottomright[0])*(topleft[1]-bottomright[1]))
        if area_box >max_box:
            max_box = area_box
            crop_images_coors = [topleft[1],bottomright[1], topleft[0],bottomright[0]]
            crop_image = frame[topleft[1]:bottomright[1], topleft[0]:bottomright[0],:]

    return crop_image,crop_images_coors



def get_XSeg_mask(img, xseg):
    """
    return mask segmented by XSeg, resized to the same with input img
    """
    xseg_res = xseg.get_resolution()
    h, w, c = img.shape
    img = my_cv2_resize(img, xseg_res, xseg_res)
    mask = xseg.extract(img)
    mask = my_cv2_resize(mask, w, h)
    return mask

def swap_back(videoimg,videopts_out,image_landmarks,input_size):
    img_size = videoimg.shape[1], videoimg.shape[0]
    xseg_res = input_size
    img_face_landmarks = np.array(image_landmarks)
    output_size = 512
    facetype = FaceType.WHOLE_FACE
    face_mask_output_mat = LandmarksProcessor.get_transform_mat (img_face_landmarks, output_size, face_type=facetype, scale= 1.0)
    # img_face_mask_a = LandmarksProcessor.get_image_hull_mask (videoimg.shape, img_landmarks)
    xseg_mat            = LandmarksProcessor.get_transform_mat (img_face_landmarks, xseg_res, face_type=facetype)
    dst_face_xseg_bgr   = cv2.warpAffine(videoimg, xseg_mat, (xseg_res,)*2, flags=cv2.INTER_CUBIC )
    return dst_face_xseg_bgr
    
def find_ratio_intersection_v2(videoimg,crops_coors,xseg_256_extract_func,videopts_out,image_landmarks,mask_mount=None):
    # print("import Xseg success")
    img_size = videoimg.shape[1], videoimg.shape[0]
    xseg_res = xseg_256_extract_func.get_resolution()
    img_face_landmarks = np.array(image_landmarks)
    output_size = 512
    facetype = FaceType.WHOLE_FACE
    face_mask_output_mat = LandmarksProcessor.get_transform_mat (img_face_landmarks, output_size, face_type=facetype, scale= 1.0)
    # img_face_mask_a = LandmarksProcessor.get_image_hull_mask (videoimg.shape, img_landmarks)
    xseg_mat            = LandmarksProcessor.get_transform_mat (img_face_landmarks, xseg_res, face_type=facetype)
    dst_face_xseg_bgr   = cv2.warpAffine(videoimg, xseg_mat, (xseg_res,)*2, flags=cv2.INTER_CUBIC )
    # cv2.imwrite("Xseg_debug.png",dst_face_xseg_bgr)
    dst_face_xseg_mask  = xseg_256_extract_func.extract(dst_face_xseg_bgr)
    # dst_face_xseg_mask  = fillhole(dst_face_xseg_mask)
    X_dst_face_mask_a_0 = cv2.resize (dst_face_xseg_mask, (output_size,output_size), interpolation=cv2.INTER_CUBIC)
    wrk_face_mask_a_0 = X_dst_face_mask_a_0
    wrk_face_mask_a_0[ wrk_face_mask_a_0 < (1.0/255.0) ] = 0.0
    img_face_mask_a = cv2.warpAffine( wrk_face_mask_a_0, face_mask_output_mat, img_size, np.zeros(videoimg.shape[0:2], dtype=np.float32), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_CUBIC )[...,None]
    img_face_mask_a = np.clip (img_face_mask_a, 0.0, 1.0)
    img_face_mask_a [ img_face_mask_a < (1.0/255.0) ] = 0.0
    img_face_mask_a [ img_face_mask_a >0.0 ] = 1.0

    img_face_occ = np.array(255*img_face_mask_a, dtype=np.uint8)
    var_middle_occ = np.copy(mask_mount[:,:])
    img_mouth_mask_occ = cv2.bitwise_and(img_face_occ,var_middle_occ)
    mount_mask_Occ = img_mouth_mask_occ.copy()

    #cal ratio between 2 mouth mask Mediapipe and FaceOcc
    mask_mount = np.atleast_3d(mask_mount).astype('float') / 255.
    mask_mount[mask_mount != 1] = 0

    img_mouth_mask_occ = np.atleast_3d(img_mouth_mask_occ).astype('float') / 255.
    img_mouth_mask_occ[img_mouth_mask_occ != 1] = 0
    newval = len(img_mouth_mask_occ[img_mouth_mask_occ > 0])/len(mask_mount[mask_mount >0])
    return newval,mount_mask_Occ,img_face_occ

if __name__=='__main__':
    model_path = '/home/ubuntu/quyennv/DeepFaceLab_Linux/workspace/model'
    xseg = init_XSeg(model_path, device='cuda')
    mobile_net, resnet_net = loadmodelface()
    device = "cuda"

    FRAME_PATH = '/home/ubuntu/Duy_test_folder/Retinaface_Mediapipe/Dr_video/DrDisrespect-Falls-in-Love-with-Warzone-again-thanks-to-new-Game-Mode-30FPS.mp4'
    capFrame = cv2.VideoCapture(FRAME_PATH)
    fps = capFrame.get(cv2.CAP_PROP_FPS)
    width_  = int(capFrame.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_ = int(capFrame.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # width_Face  = capFace.get(cv2.CAP_PROP_FRAME_WIDTH)
    # height_Face = capFace.get(cv2.CAP_PROP_FRAME_HEIGHT)
    output_path = '/home/ubuntu/quyennv/DeepFaceLab_Linux/DeepFaceLab/test_video/Xseg_Dr_debug.mp4'
    encoder_video = ffmpeg_encoder(output_path, fps, width_, height_)
    totalF = int(capFrame.get(cv2.CAP_PROP_FRAME_COUNT))
    minute_start = 0
    second_start = 0
    minute_stop =9
    second_stop =20
    frame_start = int(minute_start*60*fps+second_start*fps)
    frame_stop = int(minute_stop*60*fps+second_stop*fps)
    # frame_start = 9550
    # frame_stop = 9700
    total_f = frame_stop-frame_start
    pbar = tqdm(total=total_f)

    count_frame = 0
    while capFrame.isOpened():
        count_frame = count_frame+1
        okay1  , videoimg = capFrame.read()
        cv2.putText(videoimg, text='Fr:'+str(count_frame), org=(100, 40), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1.1, color=(0, 255, 0),thickness=2)
        videoimg_copy = videoimg.copy()

        if count_frame <= frame_start:
            continue
        elif count_frame > frame_stop:
            break
        elif count_frame > frame_start and count_frame <= frame_stop:
            pbar.update(1)
            crop_image,crop_images_coors = retinaface_check(videoimg)
            if crop_image is None or crop_images_coors is None:
                write_frame(videoimg_copy,encoder_video)
            else:
                mask = get_XSeg_mask(crop_image, xseg)
                mask = np.array(255*mask, dtype=np.uint8)
                mask[mask>0] = 255
                mask_frame = np.zeros((height_,width_), np.uint8)
                mask_frame[crop_images_coors[0]:crop_images_coors[1],crop_images_coors[2]:crop_images_coors[3]] = mask
                img_bgr_uint8_occ = normalize_channels(mask_frame, 3)
                img_bgr_occ = img_bgr_uint8_occ.astype(np.int32)*255
                img_bgr_occ[img_bgr_occ>0] =255
                img_bgr_occ[:,:,0] = 0
                img_bgr_occ[:,:,2] = 0
                alpha_0 = 0.4
                # print(mask_frame.shape, videoimg.shape)
                output_main = cv2.addWeighted(img_bgr_occ, alpha_0,videoimg, 1-alpha_0, 0, dtype = cv2.CV_32F)

                write_frame(output_main,encoder_video)

    # cv2.imwrite('/home/ubuntu/quyennv/media/mask.png', mask)
