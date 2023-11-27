import cv2
import os
from tqdm import tqdm
import numpy as np
import math
from PIL import Image
import ffmpeg
import subprocess
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
from GFPGAN.gfpgan.utils import GFPGANer
import warnings
warnings.filterwarnings("ignore")



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

def load_gfpgan_model(model_path):
    if not torch.cuda.is_available():  # CPU
        import warnings
        warnings.warn('The unoptimized RealESRGAN is slow on CPU. We do not use it. '
                    'If you really want to use it, please modify the corresponding codes.')
        bg_upsampler = None
    else:
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
            half=True)  # need to set False in CPU mode
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
    ENCODER = 'resnet18'
    ENCODER_WEIGHTS = 'imagenet'
    CLASSES = 1
    ATTENTION = None
    ACTIVATION = None
    DEVICE = 'cuda:0'

    FRAME_PATH = '/home/ubuntu/quyennv/DeepFake/videos/result_4p18_4p28.mp4'
    device = torch.device("cuda")

    return restorer

def gfpgan_img(img, restorer):
    restorer = load_gfpgan_model(model_path)
    cropped_faces, restored_faces,restored_img = restorer.enhance(
    img, has_aligned=False, only_center_face=False, paste_back=True)
    return restored_img


if __name__ == "__main__":
    model_path = "/home/ubuntu/Duy_test_folder/Retinaface_Mediapipe/GFPGAN/experiments/pretrained_models/GFPGANv1.3.pth"
    imgpath = '/home/ubuntu/quyennv/DeepFake/source_img/10.png'
    img = cv2.imread(imgpath)
    restorer = load_gfpgan_model(model_path)
    img = gfpgan_img(img, restorer)
    print(img)

    # font = cv2.FONT_HERSHEY_SIMPLEX
    #
    #
    # if not torch.cuda.is_available():  # CPU
    #     import warnings
    #     warnings.warn('The unoptimized RealESRGAN is slow on CPU. We do not use it. '
    #                 'If you really want to use it, please modify the corresponding codes.')
    #     bg_upsampler = None
    # else:
    #     from basicsr.archs.rrdbnet_arch import RRDBNet
    #     from realesrgan import RealESRGANer
    #     model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
    #     bg_upsampler = RealESRGANer(
    #         scale=2,
    #         model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
    #         model=model,
    #         tile=200,
    #         tile_pad=10,
    #         pre_pad=0,
    #         half=True)  # need to set False in CPU mode
    # model_path = "/home/ubuntu/Duy_test_folder/Retinaface_Mediapipe/GFPGAN/experiments/pretrained_models/GFPGANv1.3.pth"
    # arch = 'clean'
    # channel_multiplier = 2
    # model_name = 'GFPGANv1.3'
    # scale =1
    # restorer = GFPGANer(
    #                     model_path=model_path,
    #                     upscale=scale,
    #                     arch=arch,
    #                     channel_multiplier=channel_multiplier,
    #                     bg_upsampler=bg_upsampler)
    # ENCODER = 'resnet18'
    # ENCODER_WEIGHTS = 'imagenet'
    # CLASSES = 1
    # ATTENTION = None
    # ACTIVATION = None
    # DEVICE = 'cuda:0'
    #
    # FRAME_PATH = '/home/ubuntu/quyennv/DeepFake/videos/result_4p18_4p28.mp4'
    # device = torch.device("cuda")
    #
    # capFrame = cv2.VideoCapture(FRAME_PATH)
    # fps = capFrame.get(cv2.CAP_PROP_FPS)
    # width_  = capFrame.get(cv2.CAP_PROP_FRAME_WIDTH)
    # height_ = capFrame.get(cv2.CAP_PROP_FRAME_HEIGHT)
    # totalF = int(capFrame.get(cv2.CAP_PROP_FRAME_COUNT))
    #
    # # encoder_video = ffmpeg_encoder('Dr_video/3sourceheadMovCSV_8Aug_Dr_ES_newdriving_newlogic_gfpgan.mp4', fps, int(width_)*scale, int(height_)*scale)
    # encoder_video = ffmpeg_encoder('/home/ubuntu/quyennv/DeepFake/videos/result_4p18_4p28_gfpgan.mp4', fps, int(width_)*scale, int(height_)*scale)
    # #cropped_image_folder_path = '/content/gdrive/MyDrive/Deep_fake/video_test_mediapipe_ratio_0_dot_45'
    #
    # List_speak_frame = []
    # json_path = '/home/ubuntu/Duy_test_folder/SP_Speechdetection/content/SpeakerDetection/Speech_detected_Dr2206_ES_full_voiceonly.json'
    # with open(json_path,'r') as f:
    #     list_inital = json.load(f)
    #     List_speak_frame = []#list_inital['data'][:]
    #     for j in range(len(list_inital['data'])):
    #         start = int(float(list_inital['data'][j]['start'])*fps)
    #         if start >10:
    #             start = start-10
    #         stop = int(float(list_inital['data'][j]['end'])*fps) +10
    #         print(start, stop)
    #
    #         for i in range(start,stop+1):
    #             List_speak_frame.append(i)
    #
    # # start = int(float(0)*fps)
    # # if start >10:
    # #     start = start-10
    # # stop = int(float(100)*fps) +10
    # # print(start, stop)
    # # for i in range(start,stop+1):
    # #     List_speak_frame.append(i)
    #
    # List_speak_frame = np.unique(List_speak_frame)
    # print(List_speak_frame)
    # # print("Frame speech: ", len(List_speak_frame)," - ", 100*len(List_speak_frame)/totalF,"%")
    # minute_start = 0
    # second_start = 0
    # minute_stop = 10
    # second_stop = 0
    # frame_start = int(minute_start*60*fps+second_start*fps)
    # frame_stop = int(minute_stop*60*fps+second_stop*fps)
    # # frame_start = 1940
    # # frame_stop = 2100
    # # list_speak = np.array(range(7560,7800))
    # total_f = frame_stop-frame_start
    # pbar = tqdm(total=total_f)
    # count_frame =0
    # while capFrame.isOpened():
    #     count_frame = count_frame+1
    #     #cal frame for block_frames util reach num_frame_threshold
    #     # okay1  , faceimg = capFrame.read()
    #
    #     if count_frame < frame_start:
    #         continue
    #     elif count_frame > frame_stop+1:
    #         break
    #     elif count_frame >= frame_start and count_frame <= frame_stop+1:
    #         okay1  , faceimg = capFrame.read()
    #         if not okay1:
    #             print('Cant read the video , Exit!')
    #             break
    #         # if count_frame in List_speak_frame:
    #         cropped_faces, restored_faces,restored_img = restorer.enhance(
    #         faceimg, has_aligned=False, only_center_face=False, paste_back=True)
    #         faceimg = restored_img
    #         faceimg = cv2.cvtColor(faceimg,cv2.COLOR_BGR2RGB)
    #         encoder_video.stdin.write(faceimg.tobytes())
    #         # else:
    #         #     faceimg = cv2.resize(faceimg,(int(width_)*scale, int(height_)*scale))
    #         #     faceimg = cv2.cvtColor(faceimg,cv2.COLOR_BGR2RGB)
    #         #     encoder_video.stdin.write(faceimg.tobytes())
    #     pbar.update(1)
    #
    # pbar.close()
    # encoder_video.stdin.flush()
    # encoder_video.stdin.close()
    # print("=================\n","DONE!\n","=================")
