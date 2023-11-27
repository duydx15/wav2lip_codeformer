"""
"""
import cv2
import numpy as np
import sys
import os
sys.path.append('/home/anlab/ANLAB/Wav2Lip')
import face_detection
import ffmpeg
import subprocess
from tqdm import tqdm
from PIL import Image
import time
import mediapipe as mp


def init_sfd(device):
    """
    """
    sfd_facedetector = face_detection.FaceAlignment(face_detection.LandmarksType._2D,
                                    flip_input=False, device=device)
    return sfd_facedetector

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

def main_sfd_img():
    """
    """
    device = 'cpu'
    imgpath = '/home/anlab/ANLAB/DeepFake/debug/debug_wav2lip/tmp/dst/00043.png'

    sfd = init_sfd(device)
    img = cv2.imread(imgpath)
    results = face_detect([img], sfd)
    y1, y2, x1, x2 = results[0]
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imshow('Img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    del sfd

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

def init_face_mesh():
    mp_face_mesh = mp.solutions.face_mesh
    return mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

def get_fm_landmark(img, facemesh):
    h, w = img.shape[:2]
    results = facemesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if not results.multi_face_landmarks:
        return None
    else:
        points = results.multi_face_landmarks[0].landmark
        points = [[point.x * w, point.y * h] for point in points]
    return points

def main_sfd():
    """
    """
    device = 'cuda'
    vidpath = 'DrDisrespect-Falls-in-Love-with-Warzone-again-thanks-to-new-Game-Mode-30FPS.mp4'
    savepath = 'Sfd_facemesh.mp4'
    MASK_KP_IDS = [2,326, 423,425 ,411,416, 430, 431, 262, 428, 199, 208, 32, 211, 210,192, 187, 205 , 203, 97]

    sfd = init_sfd(device)
    facemesh = init_face_mesh()

    cap = cv2.VideoCapture(vidpath)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    encoder_video = ffmpeg_encoder(savepath, fps, width, height)

    face_count = 0
    start_time = time.time()
    pbar = tqdm(total=total_frames)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        box = face_detect([frame], sfd)[0]
        if box==None:
            imageout = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            imageout = Image.fromarray(np.uint8(imageout))
            encoder_video.stdin.write(imageout.tobytes())
        else:
            y1, y2, x1, x2 = box
            roi = frame[y1:y2, x1:x2, :]
            landmark = get_fm_landmark(roi, facemesh)
            if landmark==None:
                imageout = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                imageout = Image.fromarray(np.uint8(imageout))
                encoder_video.stdin.write(imageout.tobytes())
            else:
                points = [landmark[id] for id in MASK_KP_IDS]
                points = np.array(points, dtype=np.int32)
                cv2.polylines(roi, [points], 1, (0, 255, 0), 2)
                frame[y1:y2, x1:x2, :] = roi
                face_count += 1
                now = time.time()
                cv2.putText(frame, f'Avg Time: {(now-start_time)/face_count:.2f}s', (60, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
                cv2.putText(frame, f'Detected Faces: {face_count}', (60, 60), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
                y1, y2, x1, x2 = box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                imageout = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                imageout = Image.fromarray(np.uint8(imageout))
                encoder_video.stdin.write(imageout.tobytes())
        pbar.update(1)

    pbar.close()
    encoder_video.stdin.flush()
    encoder_video.stdin.close()
    cap.release()
    del sfd, facemesh

if __name__=='__main__':
    main_sfd()
