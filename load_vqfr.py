import sys
sys.path.append('/home/ubuntu/Duy_test_folder/Retinaface_Mediapipe/VQFR_')

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
# from filterpy.kalman import KalmanFilter
# from filterpy.common import Q_discrete_white_noise
from scipy import spatial
import skimage
import json
from collections import OrderedDict
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# import tensorflow as tf

from vqfr.demo_util import VQFR_Demo


def load_vqfr_model():
    DEVICE = 'cuda:0'
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
    model_path = "/home/ubuntu/Duy_test_folder/Retinaface_Mediapipe/VQFR_/experiments/pretrained_models/VQFR_v1-33a1fac5.pth"
    # arch = 'clean'
    channel_multiplier = 2
    scale =1
    arch = 'original'
    model_name = 'VQFR_v1-33a1fac5'
    restorer = VQFR_Demo(\
                        model_path=model_path,
                        upscale=scale,
                        arch=arch,
                        # channel_multiplier=channel_multiplier,
                        bg_upsampler=bg_upsampler)
    return restorer
