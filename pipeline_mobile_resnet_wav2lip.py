import os
import sys
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from data import cfg_mnet, cfg_re50
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from models.retinaface import RetinaFace
from utils.box_utils import decode, decode_landm
from utils.timer import Timer
import imutils
import time

import mediapipe as mp
import pickle

from tqdm import tqdm
from PIL import Image
import ffmpeg
import subprocess
sys.path.append(os.path.dirname(__file__))
LOGURU_FFMPEG_LOGLEVELS = {
	"trace": "trace",
	"debug": "debug",
	"info": "info",
	"success": "info",
	"warning": "warning",
	"error": "error",
	"critical": "fatal",
}

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh




parser = argparse.ArgumentParser(description='Retinaface')
parser.add_argument('--path_video', help='Path to input video')
parser.add_argument('-m', '--trained_model', default='./weights/mobilenet0.25_Final.pth',
					type=str, help='Trained state_dict file path to open')
parser.add_argument('--output_path', default='result_pipeline.mp4', help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--network', default='mobile0.25', help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--save_folder', default='eval/', type=str, help='Dir to save results')
parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
parser.add_argument('--dataset', default='FDDB', type=str, choices=['FDDB'], help='dataset')
parser.add_argument('--confidence_threshold', default=0.9, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=5000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
parser.add_argument('-s', '--save_image', action="store_true", default=False, help='show detection results')
parser.add_argument('--vis_thres', default=0.5, type=float, help='visualization_threshold')
parser.add_argument('--input_video', default="../../workspace/model", type=str, help='file text list input video')
parser.add_argument('--input_audio', default="../../workspace/model", type=str, help='file text list input audio')
parser.add_argument('--output_video', default="../../workspace/model", type=str, help='file text list output video')
parser.add_argument('--dfl_model', default="../../workspace/model", type=str, help='path model DFLab')
parser.add_argument('--influencer', default="Kaja", type=str, help='Name model')
args = parser.parse_args()


def check_keys(model, pretrained_state_dict):
	ckpt_keys = set(pretrained_state_dict.keys())
	model_keys = set(model.state_dict().keys())
	used_pretrained_keys = model_keys & ckpt_keys
	unused_pretrained_keys = ckpt_keys - model_keys
	missing_keys = model_keys - ckpt_keys
	print('Missing keys:{}'.format(len(missing_keys)))
	print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
	print('Used keys:{}'.format(len(used_pretrained_keys)))
	assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
	return True


def remove_prefix(state_dict, prefix):
	''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
	print('remove prefix \'{}\''.format(prefix))
	f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
	return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
	print('Loading pretrained model from {}'.format(pretrained_path))
	if load_to_cpu:
		pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
	else:
		device = torch.cuda.current_device()
		pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
	if "state_dict" in pretrained_dict.keys():
		pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
	else:
		pretrained_dict = remove_prefix(pretrained_dict, 'module.')
	check_keys(model, pretrained_dict)
	model.load_state_dict(pretrained_dict, strict=False)
	return model


def predict_retinaface(model, cfg, img, scale, im_height, im_width, device):
	loc, conf, landms = model(img)  # forward pass
	priorbox = PriorBox(cfg, image_size=(im_height, im_width))
	priors = priorbox.forward()
	priors = priors.to(device)
	# print(device)
	prior_data = priors.data
	boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
	boxes = boxes * scale
	boxes = boxes.cpu().numpy()
	scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
	# print("Conf:",len(scores))
	landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
	scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
							img.shape[3], img.shape[2], img.shape[3], img.shape[2],
							img.shape[3], img.shape[2]])
	scale1 = scale1.to(device)
	landms = landms * scale1
	landms = landms.cpu().numpy()

	# ignore low scores
	inds = np.where(scores > args.confidence_threshold)[0]
	boxes = boxes[inds]
	landms = landms[inds]
	scores = scores[inds]

	# keep top-K before NMS
	# order = scores.argsort()[::-1][:args.top_k]
	order = scores.argsort()[::-1]
	boxes = boxes[order]
	landms = landms[order]
	scores = scores[order]
	# print("score",len(scores))
	# print("landms_brefore",landms)
	# do NMS
	dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
	keep = py_cpu_nms(dets, args.nms_threshold)

	dets = dets[keep, :]
	landms = landms[keep]
	# print("dest before",dets)
	# print("landms",landms)
	dets = np.concatenate((dets, landms), axis=1)
	return dets,landms


def detection_face_wav2lips(mobile_net,resnet_net, img, device,padding_ratio = None):
	H,W,_ = img.shape
	padding_size_ratio = padding_ratio
	detected_face = False
	# print("Shape" ,img.shape)
	img = imutils.resize(img, width = 640)
	H_resize, W_resize, _ = img.shape
	img_draw = img.copy()
	img = np.float32(img)
	im_height, im_width, _ = img.shape
	# print(img.shape)
	#Resize, normalize and preprocess
	scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
	img -= (104, 117, 123)
	img = img.transpose(2, 0, 1)
	img = torch.from_numpy(img).unsqueeze(0)
	img = img.to(device)
	scale = scale.to(device)

	dets,landms = predict_retinaface(mobile_net, cfg_mnet, img, scale, im_height, im_width, device)
	if dets.shape[0] == 0:
		dets,landms  = predict_retinaface(resnet_net, cfg_re50, img, scale, im_height, im_width, device)
	l_coordinate = []

	for k in range(dets.shape[0]):
		detected_face = True
		xmin = int(dets[k, 0])
		ymin = int(dets[k, 1])
		xmax = int(dets[k, 2])
		ymax = int(dets[k, 3])
		bbox = ((xmin, ymin , xmax, ymax))
		topleft = (int(bbox[0]), int(bbox[1]))
		bottomright = (int(bbox[2]), int(bbox[3]))

		#Expand all box
		# padding_X = int((bottomright[0] - topleft[0]) * padding_size_ratio)
		# padding_Y = int((bottomright[1] - topleft[1]) * padding_size_ratio)
		# padding_topleft = (max(0, topleft[0] - padding_X), max(0, topleft[1]- padding_Y))
		# padding_bottomright = (min(W, bottomright[0] + padding_X), min(H, bottomright[1] + padding_Y))

		# #Expand bottom only_center_face
		padding_Y = int((bottomright[1] - topleft[1]) * padding_size_ratio)
		padding_topleft = (max(0, topleft[0]), max(0, topleft[1]))
		padding_bottomright = (min(W, bottomright[0]), min(H, bottomright[1] + padding_Y))
		coordinate = (padding_topleft, padding_bottomright)
		l_coordinate.append(coordinate)

	truth_face_coordinate = []
	five_points = []
	highest_ycenter_bottomright = 0
	index_truth_face = -1
	for index, coordinate in enumerate(l_coordinate):
		scale_ratio = W/W_resize
		scale_topleft = (int(coordinate[0][0] * scale_ratio), int(coordinate[0][1] * scale_ratio))
		scale_bottomright = (min(int(coordinate[1][0] * scale_ratio), W), min(int(coordinate[1][1] * scale_ratio),H))
		y_center = (scale_topleft[1] + scale_bottomright[1])/2
		if y_center > highest_ycenter_bottomright:
			highest_ycenter_bottomright = y_center
			# index_truth_face = index
			truth_face_coordinate = [(scale_topleft, scale_bottomright)]
			five_points = landms[index][:]*scale_ratio
	# print("five_points",five_points)

	return truth_face_coordinate, detected_face,five_points


def facial_landmark_detection(face_mesh, image_in):
	image = image_in.copy()
	image.flags.writeable = False
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	results = face_mesh.process(image)

	# Draw the face mesh annotations on the image.
	image.flags.writeable = True
	image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
	detected = False
	if results.multi_face_landmarks:
		for face_landmarks in results.multi_face_landmarks:
			detected = True
			mp_drawing.draw_landmarks(
				image=image,
				landmark_list=face_landmarks,
				connections=mp_face_mesh.FACEMESH_TESSELATION,
				landmark_drawing_spec=None,
				connection_drawing_spec=mp_drawing_styles
				.get_default_face_mesh_tesselation_style())
			mp_drawing.draw_landmarks(
				image=image,
				landmark_list=face_landmarks,
				connections=mp_face_mesh.FACEMESH_CONTOURS,
				landmark_drawing_spec=None,
				connection_drawing_spec=mp_drawing_styles
				.get_default_face_mesh_contours_style())
			mp_drawing.draw_landmarks(
				image=image,
				landmark_list=face_landmarks,
				connections=mp_face_mesh.FACEMESH_IRISES,
				landmark_drawing_spec=None,
				connection_drawing_spec=mp_drawing_styles
				.get_default_face_mesh_iris_connections_style())
	return image, detected

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

def loadmodelface():
	torch.set_grad_enabled(False)
	# face_mesh = mp_face_mesh.FaceMesh(
	# 						max_num_faces=1,
	# 						refine_landmarks=True,
	# 						min_detection_confidence=0.5,
	# 						min_tracking_confidence=0.5)
	device = torch.device("cuda")
	mobile_net = RetinaFace(cfg=cfg_mnet, phase = 'test')
	mobile_net = load_model(mobile_net, "/home/ubuntu/Documents/wav2lip_codeformer/weights/mobilenet0.25_Final.pth", args.cpu)
	mobile_net.eval()
	print('Finished loading model!')
	cudnn.benchmark = True

	mobile_net = mobile_net.to(device)

	resnet_net = RetinaFace(cfg=cfg_re50, phase = 'test')
	resnet_net = load_model(resnet_net, "/home/ubuntu/Documents/wav2lip_codeformer/weights/Resnet50_Final.pth", args.cpu)
	resnet_net.eval()
	resnet_net = resnet_net.to(device)

	return mobile_net, resnet_net


if __name__ == '__main__':
	torch.set_grad_enabled(False)
	face_mesh = mp_face_mesh.FaceMesh(
							max_num_faces=1,
							refine_landmarks=True,
							min_detection_confidence=0.5,
							min_tracking_confidence=0.5)

	mobile_net = RetinaFace(cfg=cfg_mnet, phase = 'test')
	mobile_net = load_model(mobile_net, "/home/ubuntu/Documents/wav2lip_codeformer/weights/mobilenet0.25_Final.pth", args.cpu)
	mobile_net.eval()
	print('Finished loading model!')
	cudnn.benchmark = True
	device = torch.device("cuda")
	mobile_net = mobile_net.to(device)

	resnet_net = RetinaFace(cfg=cfg_re50, phase = 'test')
	resnet_net = load_model(resnet_net, "/home/ubuntu/Documents/wav2lip_codeformer/weights/Resnet50_Final.pth", args.cpu)
	resnet_net.eval()
	resnet_net = resnet_net.to(device)


	# save file
	if not os.path.exists(args.save_folder):
		os.makedirs(args.save_folder)
	cap = cv2.VideoCapture(args.path_video)
	# cap = cv2.VideoCapture(0)
	frame_width = int(cap.get(3))
	frame_height = int(cap.get(4))
	fps = cap.get(cv2.CAP_PROP_FPS)

	size = (frame_width, frame_height)

	length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


	encoder = ffmpeg_encoder('vid256.mp4', fps, frame_width, frame_height)


	# file = open("Log_MobileNet.txt",'w')
	count_frame = 0
	count_detected = 0
	count_landmark = 0
	time_s = time.time()

	pbar = tqdm(total=length)

	while cap.isOpened():
		success, image = cap.read()
		if not success:
			print("Ignoring empty camera frame.")
			break
		l_coordinate, detected_face = detection_face(mobile_net,resnet_net, image, device)
		o_frame = image.copy()
		if not detected_face:
			continue
		else:
			# for i in range(len(l_coordinate[:1])):
			coordinate = l_coordinate[0]
			topleft, bottomright = coordinate
			# print(coordinate)

			# height, width, _ = o_frame.shape
			# expand_ratio = 0.3
			# mostleft = topleft[0]
			# mostright = bottomright[0]
			# boxW = int((mostright - mostleft)*expand_ratio)
			# mostleft = max(mostleft - boxW, 0)
			# mostright = min(mostright + boxW, width)
			# top = topleft[1]
			# bottom = bottomright[1]
			# boxH = int((bottom - top)*expand_ratio)
			# top = max(top - boxH, 0)
			# bottom = min(bottom + boxH, height)

			crop_image = image[topleft[1]:bottomright[1], topleft[0]:bottomright[0],:]
			# result = insightface_facial_landmark(insight_app, crop_image)

			result, detected_keypoint = facial_landmark_detection(face_mesh, crop_image)
			image[topleft[1]:bottomright[1],topleft[0]:bottomright[0],:] = result
			image = cv2.rectangle(image, topleft, bottomright, (255, 0, 0), 1)

			# dim = (256, 256)
			# resized = cv2.resize(o_frame[topleft[1]:bottomright[1], topleft[0]:bottomright[0],:], dim, interpolation = cv2.INTER_AREA)
			# resized = cv2.resize(o_frame[top:bottom, mostleft:mostright,:], dim, interpolation = cv2.INTER_AREA)
			# cv2.imshow('Tesster', resized)
			frame = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
			encoder.stdin.write(frame.tobytes())



		count_frame += 1
		pbar.update(1)
		# if cv2.waitKey(5) & 0xFF == 27:
		# 	break
	cap.release()
	pbar.close()
	encoder.stdin.flush()
	encoder.stdin.close()
