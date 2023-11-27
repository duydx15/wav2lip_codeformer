import imutils
import cv2
import numpy as np
# import pafy
import os
import uuid
import time
import json
import requests
import datetime
from PIL import Image
import ffmpeg
import subprocess
from tqdm import tqdm
LOGURU_FFMPEG_LOGLEVELS = {
	"trace": "trace",
	"debug": "debug",
	"info": "info",
	"success": "info",
	"warning": "warning",
	"error": "error",
	"critical": "fatal",
}

#ffmpeg -y -f image2 -framerate 60 -i /dev/shm/out/%5d.png -vcodec libx264 /dev/shm/debug.mp4
FRAMES_TO_PERSIST = 100
MIN_SIZE_FOR_MOVEMENT = 1000
MOVEMENT_DETECTED_PERSISTENCE = 100


frames = ffmpeg.input(
	"pipe:0",
	format="rawvideo",
	pix_fmt="rgb24",
	vsync="1",
	s=f"640x360",
	# s=f"2560x1440",
	r=30,
)

encoder = subprocess.Popen(
	ffmpeg.compile(
		ffmpeg.output(
			frames,
			'debug.mp4',
			pix_fmt="yuv420p",
			vcodec="libx264",
			acodec="copy",
			r=30,
			crf=17,
			vsync="1",
			# map_metadata=1,
			# metadata="comment=Upscaled with Video2X",
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
)

def get_concat_v(im1, im2):
	dst = Image.new('RGB', (im1.width, im1.height + im2.height))
	dst.paste(im1, (0, 0))
	dst.paste(im2, (0, im1.height))
	return dst

font = cv2.FONT_HERSHEY_SIMPLEX

urlstream = "/home/thotd/Documents/streamer_interactive/Retinaface_Mediapipe/in.mp4"

def runcapture(inputlink = ''):
	capture = cv2.VideoCapture(inputlink)

	totalF = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
	pbar = tqdm(total=totalF)

	first_frame = None
	next_frame = None

	delay_counter = 0
	movement_persistent_counter = 0

	totaltime = 0
	frame_count = 0

	oldtime = time.time()
	while True:
		grabbed, frame = capture.read()
		if not grabbed:
			print("CAPTURE ERROR")
			# capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
			# continue
			capture.release()
			time.sleep(2)
			break

		frame = imutils.resize(frame, width = 640)
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		gray = cv2.GaussianBlur(gray, (21, 21), 0)

		if first_frame is None:
			first_frame = gray

		delay_counter += 1
		if delay_counter > FRAMES_TO_PERSIST:
			delay_counter = 0
			first_frame = next_frame
		next_frame = gray

		frame_delta = cv2.absdiff(first_frame, next_frame)
		thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]

		thresh = cv2.dilate(thresh, None, iterations = 2)
		# cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

		# cv2.imshow("frame", thresh)
		alpha = 0.5  # Transparency factor.

		# Following line overlays transparent rectangle over the image
		img2 = cv2.merge((thresh,thresh,thresh))
		image_new = cv2.addWeighted(img2, alpha, frame, 1 - alpha, 0)

		img = cv2.cvtColor(image_new, cv2.COLOR_BGR2RGB)
		im_pil = Image.fromarray(img)
		# imgFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		# im_pil2 = Image.fromarray(imgFrame)
		# im_out = get_concat_v(im_pil, im_pil2)


		encoder.stdin.write(im_pil.tobytes())

		pbar.update(1)


		# for c in cnts:
		# 	if cv2.contourArea(c) > MIN_SIZE_FOR_MOVEMENT:
		# 		transient_movement_flag = True
		#
		# 		(x, y, w, h) = cv2.boundingRect(c)
		# 		cv2.rectangle(gateonly, (x, y), (x + w, y + h), (0, 255, 0), 2)
		#
		# if transient_movement_flag == True:
		# 	movement_persistent_flag = True
		# 	movement_persistent_counter = MOVEMENT_DETECTED_PERSISTENCE

		# print(gray.shape)
		# if movement_persistent_counter > 0:
			# movement_persistent_counter -= 1
			# movement_persistent_counter = 0

			# difftime = time.time() - oldtime
			# if difftime >= 0.5:
			# 	delay_counter = 0
			# 	first_frame = next_frame
			# 	try:
			#
			# 	except Exception as e:
			# 		print(e)
			# 	oldtime = time.time()
			# cv2.putText(frame, str("Movement Detected"), (10,35), font, 0.75, (255,255,255), 2, cv2.LINE_AA)

		# ch = cv2.waitKey(1)
		# if ch & 0xFF == ord('q'):
		# 	break
	pbar.close()

runcapture(urlstream)

encoder.stdin.flush()
encoder.stdin.close()
# cv2.destroyAllWindows()
