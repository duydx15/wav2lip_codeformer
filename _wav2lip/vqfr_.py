from basicsr.archs.rrdbnet_arch import RRDBNet
import torch
from vqfr.demo_util import VQFR_Demo

def load_vqfr_model(model_path, device):
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
    channel_multiplier = 2
    arch = 'original'
    model_name = 'VQFR_v1-33a1fac5'
    scale =1
    restorer = VQFR_Demo(
                        model_path=model_path,
                        upscale=scale,
                        arch=arch,
                        # channel_multiplier=channel_multiplier,
                        bg_upsampler=bg_upsampler)

    return restorer


if __name__=='__main__':
    pass
