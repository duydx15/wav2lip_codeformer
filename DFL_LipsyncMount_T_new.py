import cv2
import mediapipe as mp
import os
from tqdm import tqdm
import numpy as np
import math
from PIL import Image
import ffmpeg
import subprocess
from pipeline_mobile_resnet import loadmodelface, detection_face
from color_transfer import color_transfer, color_transfer_mix, color_transfer_sot, color_transfer_mkl, color_transfer_idt, color_hist_match, reinhard_color_transfer, linear_color_transfer
from common import random_crop, normalize_channels, cut_odd_image, overlay_alpha_image

import torch

from scipy import spatial
# from estimate_sharpness import estimate_sharpness
LOGURU_FFMPEG_LOGLEVELS = {
	"trace": "trace",
	"debug": "debug",
	"info": "info",
	"success": "info",
	"warning": "warning",
	"error": "error",
	"critical": "fatal",
}


import warnings
warnings.filterwarnings("ignore")

class KalmanTracking(object):
	# init kalman filter object
	def __init__(self, point):
		deltatime = 0.03 # 30fps
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
		 [0, 0, 0, 1]], np.float32) * deltatime
		self.kalman.measurementNoiseCov = np.array([[1, 0],
													[0, 1]], np.float32) * 1.5

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
		self.w = 0
		self.h = 0
		self.kpold = None

	def noneArray(self):
		return len(self.kflist) == 0

	def bigMask(self):
		return len(self.kflist) == len(FACEMESH_bigmask)

	def setpoints(self, points, w=1920, h=1080):
		for value in points:
			intpoint = np.array([np.float32(value[0]), np.float32(value[1])], np.float32)
			self.kflist.append(KalmanTracking(intpoint))

		self.w = w
		self.h = h
		self.kpold = points

		# print('setpoints ', self.bigmask)
	def getpoints_new(self, kps):
		orginmask = np.zeros((self.h,self.w),dtype=np.float32)
		orginmask = cv2.fillConvexPoly(orginmask, np.array(self.kpold[:-18], np.int32), 1) if self.bigMask() else cv2.fillConvexPoly(orginmask, np.array(self.kpold, np.int32), 1)

		newmask = np.zeros((self.h,self.w),dtype=np.float32)
		newmask = cv2.fillConvexPoly(newmask, np.array(kps[:-18], np.int32), 1) if self.bigMask() else cv2.fillConvexPoly(newmask, np.array(kps, np.int32), 1)
		# cv2.imwrite('orginmask.jpg' , orginmask*255)
		val = binaryMaskIOU_(torch.from_numpy(orginmask), torch.from_numpy(newmask))
		# print('binaryMaskIOU_ ', val)

		if val < 0.9:
			del self.kflist[:]
			self.setpoints(kps,self.w, self.h)
			return kps

		kps_o = kps.copy()
		for i in range(len(kps)):
			intpoint = np.array([np.float32(kps[i][0]), np.float32(kps[i][1])], np.float32)
			tmp = self.kflist[i].getpoint(intpoint)
			kps[i] = (tmp[0][0], tmp[1][0])
		# kflist_backup = self.kflist.copy()

		# distance = 1
		# if not self.bigMask():
		#
		# 	# orgindata = np.array(kps_o, np.int32).reshape(-1)
		# 	# newdata = np.array(kps, np.int32).reshape(-1)
		# 	# distance = spatial.distance.cosine(orgindata, newdata)
		#
			# print('binaryMaskIOU_ ', val)
		# if not self.bigMask() and val >= 0.95:
		# 	for i in range(len(kps)):
		# 		intpoint = np.array([np.float32(kps[i][0]), np.float32(kps[i][1])], np.float32)
		# 		tmp = self.kflist[i].getpoint(intpoint)
		# 		kps[i] = (tmp[0][0], tmp[1][0])
		# 	# del self.kflist[:]
		# 	# self.kflist = kflist_backup
		# 	return self.kpold
		# 	# return kps
		#
		# self.kpold = kps
		return kps

	def getpoints(self, kps):
		orginmask = np.zeros((self.h,self.w),dtype=np.float32)
		orginmask = cv2.fillConvexPoly(orginmask, np.array(kps[:-18], np.int32), 1) if self.bigMask() else cv2.fillConvexPoly(orginmask, np.array(kps, np.int32), 1)
		kps_o = kps.copy()

		for i in range(len(kps)):
			intpoint = np.array([np.float32(kps[i][0]), np.float32(kps[i][1])], np.float32)
			tmp = self.kflist[i].getpoint(intpoint)
			kps[i] = (tmp[0][0], tmp[1][0])

		newmask = np.zeros((self.h,self.w),dtype=np.float32)
		newmask = cv2.fillConvexPoly(newmask, np.array(kps[:-18], np.int32), 1) if self.bigMask() else cv2.fillConvexPoly(newmask, np.array(kps, np.int32), 1)

		val = binaryMaskIOU_(torch.from_numpy(orginmask), torch.from_numpy(newmask))

		if val < 0.9:
			del self.kflist[:]
			self.setpoints(kps_o,self.w, self.h)
			return kps_o

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
	# 	distance = distance + 0.1
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

debuglist = [8, 298, 301, 368, 347, 329, 437, 168, 217, 100, 118, 139, 71, 68, #sunglass
	164, 391, 436, 432, 430, 273, 409, 11, 185, 210, 212, 216, 165,  #mustache
	13, 311, 308, 402, 14, 178, 78, 81, #lips Inner
	0, 269, 291, 375, 405, 17, 181, 146, 61, 39, #lips Outer
	4, 429, 327, 2, 98, 209, #noise
	151, 200, 175, 65, 69, 194, 140, 205, 214, 135, 215,
	177, 137, 34, 295, 299, 418, 369,
	425, 434, 364, 435, 401, 366, 264, 43]


FACEMESH_mustache = [164, 391, 436, 432, 430, 273, 409, 11, 185, 210, 212, 216, 165]

# FACEMESH_lips_2 = [326, 426, 427, 434, 430, 431, 262, 428, 199, 208, 32, 211, 210, 214, 207, 206, 97]
FACEMESH_lips_2 = [326, 423,425 ,411, 416, 430, 431, 262, 428, 199, 208, 32, 211, 210,192, 187, 205 , 203, 97]
# FACEMESH_lips_3 = [326, 426, 411, 416, 394, 395, 369, 396, 175, 171, 140, 170, 169, 192, 187, 203, 97]
# FACEMESH_lips_3 = [326, 426, 411, 416, 394, 424, 418, 421, 200, 201, 194, 204, 169, 192, 187, 203, 97]

FACEMESH_bigmask = [197, 419, 399, 437, 355, 371, 266, 425, 411, 416,
					394, 395,369, 396, 175, 171, 140, 170, 169,
					192, 187, 205, 36, 142, 126, 217, 174, 196,
					164, 391, 436, 432, 430, 273, 409, 11, 185, 210, 212, 216, 165]

# FACEMESH_bigmask = [197, 419, 399, 437, 355, 371, 266, 425, 411, 416,
# 					394, 395,369, 396, 175, 171, 140, 170, 169,
# 					192, 187, 205, 36, 142, 126, 217, 174, 196,
# 					185, 40, 39, 37, 0, 267, 269, 270, 409,
# 					186, 92, 165, 167, 164, 393, 391, 322, 410]

FACEMESH_pose_estimation = [34,264,168,33, 263]
# 61 191-95 80-88 81-178 82-87 13-14 312-317 311-402 310-318 415-324 291
# 206 97 326 426 2, 164, 94, 19
landmark_points_68 = [162,234,93,58,172,136,149,148,152,377,378,365,397,288,323,454,389,71,63,105,66,107,336,
				  296,334,293,301,168,197,5,4,75,97,2,326,305,33,160,158,133,153,144,362,385,387,263,373,
				  380,61,39,37,0,267,269,291,405,314,17,84,181,78,82,13,312,308,317,14,87,
				  206, 97, 326, 426, 2, 164, 94, 19, 61, 291]


# def pointInRect(point, rect):
# 	x1, y1, x2, y2 = rect
# 	wbox = abs(x2-x1)
# 	xo = (x1+x2)/2
# 	yo = (y1+y2)/2
# 	x, y = point
# 	dist1 = math.hypot(x-xo, y-yo)
#
# 	aaa = dist1/wbox if wbox>0 else 1
# 	# print(dist1, ' ', wbox, ' ',aaa)
# 	# print('cur: ', point, '\told: ',  (xo,yo))
# 	# print('oldbox: ', rect)
# 	if (x1 < x and x < x2):
# 		if (y1 < y and y < y2):
# 			if aaa <= 0.06:
# 				return True
# 	return False

# def facemeshTrackByBox(multi_face_landmarks,  w, h):
# 	# print(bbox)
# 	for faceIdx, face_landmarks in enumerate(multi_face_landmarks):
# 		listpoint = []
# 		for i in range(len(FACEMESH_lips_1)):
# 			idx = FACEMESH_lips_1[i]
# 			x = face_landmarks.landmark[idx].x
# 			y = face_landmarks.landmark[idx].y
#
# 			realx = x * w
# 			realy = y * h
# 			listpoint.append((realx, realy))
#
# 		video_leftmost_x = min(x for x, y in listpoint)
# 		video_bottom_y = min(y for x, y in listpoint)
# 		video_rightmost_x = max(x for x, y in listpoint)
# 		video_top_y = max(y for x, y in listpoint)
#
# 		# x = (video_leftmost_x+video_rightmost_x)/2
# 		y = (video_bottom_y+video_top_y)/2
# 		# point = (x,y)
# 		# print(point, ' ', h, w)
# 		if y < h/2:
# 			continue
# 		# if pointInRect(point, bbox):
# 		return faceIdx
# 	return -1

# def facemeshTrackByBoxCrop(multi_face_landmarks,  w, h):
# 	for faceIdx, face_landmarks in enumerate(multi_face_landmarks):
# 		listpoint = []
# 		for i in range(len(FACEMESH_lips_1)):
# 			idx = FACEMESH_lips_1[i]
# 			x = face_landmarks.landmark[idx].x
# 			y = face_landmarks.landmark[idx].y
#
# 			realx = x * w
# 			realy = y * h
# 			listpoint.append((realx, realy))
#
# 		video_leftmost_x = min(x for x, y in listpoint)
# 		video_bottom_y = min(y for x, y in listpoint)
# 		video_rightmost_x = max(x for x, y in listpoint)
# 		video_top_y = max(y for x, y in listpoint)
#
# 		y = (video_bottom_y+video_top_y)/2
# 		# point = (x,y)
# 		# print(point, ' ', h, w)
# 		if y < h/2:
# 			continue
# 		# if pointInRect(point, bbox):
# 		return faceIdx
# 	return -1

def get_face_by_RetinaFace(facea, inimg, mobile_net, resnet_net, device, kf = None, kf_driving = None):

	h,w = inimg.shape[:2]

	l_coordinate, detected_face = detection_face(mobile_net,resnet_net, inimg, device)
	if not detected_face:
		return None, None

	face_landmarks = None
	bbox = None
	# print('get_face_by_RetinaFace ', l_coordinate)
	for i in range(len(l_coordinate)):
		topleft, bottomright = l_coordinate[i]
		if bottomright[1] < h/3:
			continue

		crop_image = inimg[topleft[1]:bottomright[1], topleft[0]:bottomright[0],:]
		# result = insightface_facial_landmark(insight_app, crop_image)
		# print('l_coordinate[i] ', l_coordinate[i])
		# result, detected_keypoint = facial_landmark_detection(face_mesh, crop_image)

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

	if not face_landmarks:
		return None, None
	# print('bbox ', bbox)
	bbox_w = bbox[2] - bbox[0]
	bbox_h = bbox[3] - bbox[1]

	listpointLocal = []
	for i in range(len(face_landmarks.landmark)):#FACEMESH_bigmask)):
		idx =i# FACEMESH_bigmask[i]
		x = face_landmarks.landmark[idx].x
		y = face_landmarks.landmark[idx].y

		# realx = x * bbox_w + bbox[0]
		# realy = y * bbox_h + bbox[1]
		listpointLocal.append((x, y))

	listpoint = []

	for i in range(len(listpointLocal)):
		# idx = FACEMESH_bigmask[i]
		x = listpointLocal[i][0]
		y = listpointLocal[i][1]

		realx = x * bbox_w + bbox[0]
		realy = y * bbox_h + bbox[1]
		listpoint.append((realx, realy))

	if kf is not None:
		if kf.noneArray():
			kf.setpoints(listpoint, w, h)
		else:
			listpoint = kf.getpoints(listpoint)

	# mostRight = max(listpoint[:-18], key=lambda x: (x[0], -x[1]))[0]
	# mostLeft = min(listpoint[:-18], key=lambda x: (x[0], -x[1]))[0]
	srcpts = np.array(listpoint, np.int32)
	srcpts = srcpts.reshape(-1,1,2)

	listpoint2 = []
	for i in range(len(FACEMESH_lips_2)):
		idx = FACEMESH_lips_2[i]
		z = face_landmarks.landmark[idx].z
		x = face_landmarks.landmark[idx].x
		y = face_landmarks.landmark[idx].y

		# if i == 12 or i == 13 or i == 14:
		# 	print(x , ' ' ,face_landmarks.landmark[idx].z )
		# 	# x -= (face_landmarks.landmark[idx].z * x)
		# 	x -= z/2
		# if i == 3 or i == 4 or i == 2:
		# 	print(x , ' ' ,face_landmarks.landmark[idx].z )
		# 	x += z/2

		realx = x * bbox_w + bbox[0]
		realy = y * bbox_h + bbox[1]
		# print(realx, '  ', realy)
		# idx_out = FACEMESH_lips_3[i]
		# x_out = face_landmarks.landmark[idx_out].x
		# y_out = face_landmarks.landmark[idx_out].y
		# z_out = face_landmarks.landmark[idx_out].z
		#
		# if i == 12 or i == 13 or i == 14:
		# 	# print(x_out , ' ' ,face_landmarks.landmark[idx].z )
		# 	x_out -= z_out/2
		# if i == 3 or i == 4 or i == 2:
		# 	# print(x_out , ' ' ,face_landmarks.landmark[idx].z )
		# 	x_out += z_out/2
		#
		# realx_out = x_out * bbox_w + bbox[0]
		# realy_out = y_out * bbox_h + bbox[1]
		# # if i ==  12 or i == 13 or i == 14:
		# # 	print(face_landmarks.landmark[idx].z )
		#
		# newx = (realx+realx_out)/2
		# if newx < mostLeft:
		# 	newx = mostLeft
		# if newx > mostRight:
		# 	newx = mostRight
		listpoint2.append((realx, realy))
	# print('-------------------')
	# print('listpoint2 ', listpoint2)
	if kf_driving is not None:
		if kf_driving.noneArray():
			kf_driving.setpoints(listpoint2, w, h)
		else:
			listpoint2 = kf_driving.getpoints(listpoint2)
	# print('new listpoint2 ', listpoint2)
	srcpts2 = np.array(listpoint2, np.int32)
	# srcpts2 = srcpts2.reshape(-1,1,2)

	return srcpts, srcpts2


def get_face(facea, inimg, kf=None,  kfMount=None, iswide = False):
	listpoint = []
	h,w = inimg.shape[:2]
	# print(inimg.shape[:2])
	results = facea.process(cv2.cvtColor(inimg, cv2.COLOR_BGR2RGB))
	if not results.multi_face_landmarks:
		return None, None

	# face_landmarks = None

	# if kf:
	# 	tmp = facemeshTrackByBox(results.multi_face_landmarks, w, h)
	# 	if tmp<0:
	# 		kf[:]
	# 		return None, None
	# 	else:
	# 		face_landmarks = results.multi_face_landmarks[tmp]
	# else:
	face_landmarks = results.multi_face_landmarks[0]

	listpointLocal = []
	for i in range(len(FACEMESH_bigmask)):
		idx = FACEMESH_bigmask[i]
		x = face_landmarks.landmark[idx].x
		y = face_landmarks.landmark[idx].y

		# realx = x * bbox_w + bbox[0]
		# realy = y * bbox_h + bbox[1]
		listpointLocal.append((x, y))

	# print('befor ', listpointLocal)
	# listpointLocal = kf(np.array(listpointLocal).reshape(-1)).reshape(-1,2)
	# print('after ', listpointLocal)

	listpoint = []

	for i in range(len(listpointLocal)):
		# idx = FACEMESH_bigmask[i]
		x = listpointLocal[i][0]
		y = listpointLocal[i][1]

		realx = x * w
		realy = y * h
		listpoint.append((realx, realy))

	if kf is not None:
		if kf.noneArray():
			kf.setpoints(listpoint, w, h)
		else:
			listpoint = kf.getpoints(listpoint)

	srcpts = np.array(listpoint, np.int32)
	srcpts = srcpts.reshape(-1,1,2)

	listpoint2 = []
	for i in range(len(FACEMESH_lips_2)):
		idx = FACEMESH_lips_2[i]
		x = face_landmarks.landmark[idx].x
		y = face_landmarks.landmark[idx].y

		realx = x * w
		realy = y * h
		listpoint2.append((realx,realy))
		# idx_out = FACEMESH_lips_3[i]
		# x_out = face_landmarks.landmark[idx_out].x
		# y_out = face_landmarks.landmark[idx_out].y
		#
		#
		# realx_out = x_out * w
		# realy_out = y_out * h
		# listpoint2.append(((realx+realx_out)//2, (realy+realy_out)//2))

	if kfMount is not None:
		if kfMount.noneArray():
			kfMount.setpoints(listpoint2, w, h)
		else:
			listpoint2 = kfMount.getpoints(listpoint2)

	srcpts2 = np.array(listpoint2, np.int32)
	# srcpts = np.concatenate((srcpts, srcpts2), axis=0)
	srcpts2 = srcpts2.reshape(-1,1,2)


	return srcpts, srcpts2

def getKeypointByMediapipe(face_mesh_wide, videoimg,face_mesh_256, faceimg, kfVF, kfVM, kfFF, kfFM, mobile_net,resnet_net,device):
	# videopts, videopts_out = get_face(face_mesh_wide, videoimg, kf)
	# if videopts is None:
		# return None, None, None, None
	videopts, videopts_out = get_face_by_RetinaFace(face_mesh_wide, videoimg, mobile_net, resnet_net, device, kfVF, kfVM)
	if videopts is None:
		return None, None, None, None
	# facepts, facepts_out = get_face(face_mesh_256, faceimg, kfFF, None)
	return videopts, videopts_out#, facepts, facepts_out


# def is_on_line(x1, y1, x2, y2, x3):
# 	slope = (y2 - y1) / (x2 - x1)
# 	return slope * (x3 - x1) + y1

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
	# (topy, topx) = (np.min(y), np.min(x))
	# (bottomy, bottomx) = (np.max(y), np.max(x))
	center_ = (int((topx+bottomx)/2),int((bottomy+topy)/2))

	return topy, topx, bottomy, bottomx, center_

def get_concat_h(im1, im2):
	dst = Image.new('RGB', (im1.width + im2.width, im1.height))
	dst.paste(im1, (0, 0))
	dst.paste(im2, (im1.width, 0))
	# print(dst.size)
	return dst

def get_concat_v(im1, im2):
	dst = Image.new('RGB', (im1.width, im1.height + im2.height))
	dst.paste(im1, (0, 0))
	dst.paste(im2, (0, im1.height))
	return dst

def applyAffineTransform(src, srcTri, dstTri, size):
	# Given a pair of triangles, find the affine transform.
	warpMat = cv2.getAffineTransform(np.float32(srcTri), np.float32(dstTri))

	# Apply the Affine Transform just found to the src image
	dst = cv2.warpAffine(src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

	return dst

def warpTriangle(img1, img2, t1, t2):
	# Find bounding rectangle for each triangle
	r1 = cv2.boundingRect(np.float32(t1))
	r2 = cv2.boundingRect(np.float32(t2))

	# Offset points by left top corner of the respective rectangles
	t1Rect = []
	t2Rect = []
	t2RectInt = []

	for i in range(0, 3):
		# print(t1[i][0] - r1[0])
		# print(i, ' ' , t1[i][0][1] )
		# print(r1[1])
		t1Rect.append(((t1[i][0][0] - r1[0]), (t1[i][0][1] - r1[1])))
		t2Rect.append(((t2[i][0][0] - r2[0]), (t2[i][0][1] - r2[1])))
		t2RectInt.append(((t2[i][0][0] - r2[0]), (t2[i][0][1] - r2[1])))

	# Get mask by filling triangle
	mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
	cv2.fillConvexPoly(mask, np.int32(t2RectInt), (1.0, 1.0, 1.0), 16, 0)

	# Apply warpImage to small rectangular patches
	img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
	size = (r2[2], r2[3])
	img2Rect = applyAffineTransform(img1Rect, t1Rect, t2Rect, size)
	img2Rect = img2Rect * mask

	# Copy triangular region of the rectangular patch to the output image
	img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] * ((1., 1., 1.) - mask)
	img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] + img2Rect

def write_frame(encoder_, frame_, isCv2Img = True):
	imageout = Image.fromarray(cv2.cvtColor(frame_,cv2.COLOR_RGB2BGR)) if isCv2Img else frame_
	encoder_.stdin.write(imageout.tobytes())

def add_image_by_mask(img1, img2, mask_):
	mask_not = cv2.bitwise_not(mask_)
	img2_no_mask = cv2.bitwise_and(img2, img2, mask=mask_not)
	img1_mask_only = cv2.bitwise_and(img1, img1, mask=mask_)
	return cv2.add(img2_no_mask, img1_mask_only)

font = cv2.FONT_HERSHEY_SIMPLEX

if __name__ == "__main__":

	FRAME_PATH = '/home/ubuntu/Duy_test_folder/Retinaface_Mediapipe/Dr_video/Doc_lipsync_Apr6/video/Test.mp4'
	FACE_PATH = '/home/ubuntu/Duy_test_folder/Retinaface_Mediapipe/Dr_video/Doc_lipsync_Apr6/video/Test.mp4'
	# FACE_PATH = 'headMovCSV_06June22.mp4'

	mobile_net, resnet_net = loadmodelface()
	device = torch.device("cuda")

	mp_face_mesh = mp.solutions.face_mesh
	face_mesh_256 = mp_face_mesh.FaceMesh(min_detection_confidence=0.2, min_tracking_confidence=0.2,max_num_faces=1)
	face_mesh_wide = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.2)

	facecout = 0
	# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))

	capFrame = cv2.VideoCapture(FRAME_PATH)
	capFace = cv2.VideoCapture(FACE_PATH)

	fps = int(capFrame.get(cv2.CAP_PROP_FPS))
	width_  = capFrame.get(cv2.CAP_PROP_FRAME_WIDTH)
	height_ = capFrame.get(cv2.CAP_PROP_FRAME_HEIGHT)
	encoder = ffmpeg_encoder('/home/ubuntu/Duy_test_folder/Retinaface_Mediapipe/Dr_video/Doc_lipsync_Apr6/video/crop_driving.mp4', fps, int(width_), int(height_))
	# encoder = ffmpeg_encoder('vidout256.mp4', fps, 256, 256)

	totalF = int(capFace.get(cv2.CAP_PROP_FRAME_COUNT))
	pbar = tqdm(total=totalF)

	trackkpVideoFace = None# KalmanArray()
	trackkpVideoMount = None# KalmanArray()
	trackkpFaceFace = None# KalmanArray()
	trackkpFaceMount = None# KalmanArray()

	#dynamic variable
	count_frame = -1
	Previous_model_count =0
	total_frame_loss = 0

	num_frame_loss = 0
	num_frame_loss_retina = 0
	original_frames = []
	drawed_frames = []
	save_loss_frame = []

	#const variable
	num_frame_threshold = fps + 1
	minute_start = 0
	second_start = 0
	minute_stop = 1
	second_stop = 40
	frame_start = int(minute_start * 60 * fps + second_start * fps)
	frame_stop = int(minute_stop * 60 * fps + second_stop * fps)

	print('frame_start ', frame_start, '\tframe_stop ', frame_stop, '\tfps ', fps)

	while capFrame.isOpened() or capFace.isOpened():
		okay1, videoimg = capFrame.read()
		okay2, faceimg = capFace.read()

		if not okay1 or not okay2:

			break

		count_frame += 1
		pbar.update(1)
		if count_frame < frame_start:
			continue
		elif count_frame > frame_stop:
			break



		videoimg_copy = videoimg.copy()
		original_frames.append(videoimg_copy)
		video_h,video_w, = videoimg.shape[:2]

		videopts, videopts_out = getKeypointByMediapipe(face_mesh_wide, videoimg, face_mesh_256, faceimg, trackkpVideoFace, trackkpVideoMount, trackkpFaceFace, trackkpFaceMount, mobile_net,resnet_net,device)


		# try:
		# if okay2:

			# M_face, _ = cv2.findHomography(facepts, videopts)
			# face_img = cv2.warpPerspective(faceimg,M_face,(video_w, video_h), cv2.INTER_LINEAR)

			# facepts_out = cv2.perspectiveTransform(facepts_out.astype(np.float32), M_face).astype(np.int32)
			# M_mount, _ = cv2.findHomography(facepts_out, videopts_out)
			# face_img = cv2.warpPerspective(face_img,M_mount,(video_w, video_h), cv2.INTER_LINEAR)

			# img_video_mask_mount = np.zeros((video_h,video_w), np.uint8)
			# cv2.fillPoly(img_video_mask_mount, pts =[videopts_out], color=(255,255,255))
			# # img_video_mask_mount = cv2.erode(img_video_mask_mount, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)), iterations = 1 )
			# # wrk_face_mask_a_0 = img_video_mask_mount - wrk_face_mask_a_0
			# _, _, _, _, center_mount = mask2box(img_video_mask_mount)


			# img_video_mask_face = np.zeros((video_h,video_w), np.uint8)
			# # cv2.fillPoly(img_video_mask_face, pts =[videopts[:-18]], color=(255,255,255))
			# cv2.fillPoly(img_video_mask_face, pts =[videopts[:-13]], color=(255,255,255))
			# img_video_mask_face = cv2.bitwise_or(img_video_mask_face, img_video_mask_mount)
			# topy, topx, bottomy, bottomx, center_face = mask2box(img_video_mask_face)

			# result = add_image_by_mask(face_img, videoimg, img_video_mask_face)

			# alpha_0 = 0.95
			# result[topy:bottomy, topx:bottomx] = cv2.addWeighted(result[topy:bottomy, topx:bottomx], alpha_0, videoimg[topy:bottomy, topx:bottomx], 1-alpha_0, 0.0)
			# videoimg = cv2.seamlessClone(face_img, videoimg, img_video_mask_mount, center_mount, cv2.MIXED_CLONE)

			# print('estimate_sharpness ', estimate_sharpness(cv2.bitwise_and(videoimg, videoimg, mask=img_video_mask_face)))

			# img_bgr_uint8_1 = normalize_channels(result[topy:bottomy, topx:bottomx], 3)
			# img_bgr_1 = img_bgr_uint8_1.astype(np.float32) / 255.0
			# img_bgr_1 = np.clip(img_bgr_1, 0, 1)
			# img_bgr_uint8_2 = normalize_channels(videoimg[topy:bottomy, topx:bottomx], 3)
			# img_bgr_2 = img_bgr_uint8_2.astype(np.float32) / 255.0
			# img_bgr_2 = np.clip(img_bgr_2, 0, 1)

			# new_binary = img_video_mask_face[topy:bottomy, topx:bottomx][..., np.newaxis]
			# img_face_mask_a = np.clip(new_binary, 0.0, 1.0)
			# img_face_mask_a[ img_face_mask_a < (1.0/255.0) ] = 0.0 # get rid of noise
			#
			#
			# result_new = color_transfer_mkl(img_bgr_1,img_bgr_2)
			# result_new = color_transfer_idt(img_bgr_1,img_bgr_2)
			# result_new = reinhard_color_transfer (img_bgr_1, img_bgr_2, target_mask=img_face_mask_a, source_mask=img_face_mask_a)

			# result_new = linear_color_transfer(img_bgr_1, img_bgr_2)
			# final_img = color_hist_match(result_new, img_bgr_2, 255).astype(dtype=np.float32)


			# result[topy:bottomy, topx:bottomx] = blursharpen((final_img*255).astype(np.uint8), 1, 5, 1)

			# # alpha_0 = 0.95
			# # result[topy:bottomy, topx:bottomx] = cv2.addWeighted(result[topy:bottomy, topx:bottomx], alpha_0, videoimg[topy:bottomy, topx:bottomx], 1-alpha_0, 0.0)


			# result = add_image_by_mask(result, videoimg, img_video_mask_mount)

			# img_video_mask_mount = cv2.dilate(img_video_mask_mount, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)), iterations = 1 )

			# # video_tmp = add_image_by_mask(result, videoimg, img_video_mask_mount)
			# # videoimg = cv2.addWeighted(result, 0.5, videoimg, 0.5, 0, videoimg)
			# result = cv2.seamlessClone(result, videoimg, img_video_mask_face, center_face, cv2.NORMAL_CLONE)
			# output = cv2.seamlessClone(result, videoimg, img_video_mask_mount, center_mount, cv2.NORMAL_CLONE)
			# videoimg = cv2.polylines(videoimg, [videopts_out], True, (0, 0, 255), 1)
			# videoimg = cv2.polylines(videoimg, [videopts[:-18]], True, (0, 255, 0), 1)
		# print(videopts)
		# break
		for idx in range(len(videopts)):
			videoimg = cv2.circle(videoimg, videopts[idx][0], 3, (0, 255, 0), -1)
		write_frame(encoder, videoimg)
			# for i in range(len(videopts[:-18])):
			# 	videoimg = cv2.putText(videoimg, str(i), videopts[i], font,
            #        1, (0, 0, 255), 2, cv2.LINE_AA)
			# drawed_frames.append(output)

		# except Exception:
		# 	write_frame(encoder, videoimg)
		# 	# use orgin frame
		# 	# drawed_frames.append(videoimg_copy)
		# 	continue

	pbar.close()
	encoder.stdin.flush()
	encoder.stdin.close()
