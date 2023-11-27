# import face_alignment
import skimage.io
import numpy
from argparse import ArgumentParser
from skimage import img_as_ubyte
from skimage.transform import resize
from tqdm import tqdm
import os
import imageio
import numpy as np
import math
import mediapipe as mp
import cv2
from config import config
from multiprocessing import Pool, cpu_count
from pipeline_mobile_resnet import loadmodelface, detection_face
import torch

def _normalized_to_pixel_coordinates(
    normalized_x: float, normalized_y: float, image_width: int,
    image_height: int):
  """Converts normalized value pair to pixel coordinates."""

  def is_valid_normalized_value(value: float) -> bool:
    return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                      math.isclose(1, value))

  if not (is_valid_normalized_value(normalized_x) and
          is_valid_normalized_value(normalized_y)):
    return None
  x_px = min(math.floor(normalized_x * image_width), image_width - 1)
  y_px = min(math.floor(normalized_y * image_height), image_height - 1)
  return x_px, y_px

def extract_bbox(frame, face_detection_model):
    results = face_detection_model.process(frame)

    if not results.detections:
        return []
    annotated_image = frame.copy()
    image_rows, image_cols, _ = annotated_image.shape
    colordotred = (0, 0, 255)
    bboxes = []
    for detection in results.detections:
        location = detection.location_data
        relative_bounding_box = location.relative_bounding_box

        rect_start_point = _normalized_to_pixel_coordinates(
            relative_bounding_box.xmin, relative_bounding_box.ymin, image_cols,
            image_rows)
        rect_end_point = _normalized_to_pixel_coordinates(
            relative_bounding_box.xmin + relative_bounding_box.width,
            relative_bounding_box.ymin + +relative_bounding_box.height, image_cols,
            image_rows)
        if rect_start_point is None or rect_end_point is None:
            # return []
            continue
        cv2.rectangle(annotated_image, rect_start_point, rect_end_point,
            colordotred, thickness=2)
        bboxes.append([rect_start_point[0],rect_start_point[1],\
                        rect_end_point[0],rect_end_point[1]])
    return np.array(bboxes)

def bb_intersection_over_union(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def join(tube_bbox, bbox):
    xA = min(tube_bbox[0], bbox[0])
    yA = min(tube_bbox[1], bbox[1])
    xB = max(tube_bbox[2], bbox[2])
    yB = max(tube_bbox[3], bbox[3])
    return (xA, yA, xB, yB)


def compute_bbox(start, end, fps, tube_bbox, frame_shape,\
    inp, image_shape, increase_area=0.1, bname='', count=0, path_save_output = None):
    left, top, right, bot = tube_bbox
    width = right - left
    height = bot - top

    #Computing aspect preserving bbox
    width_increase = max(increase_area, ((1 + 2 * increase_area) * height - width) / (2 * width))
    height_increase = max(increase_area, ((1 + 2 * increase_area) * width - height) / (2 * height))

    left = int(left - width_increase * width)
    top = int(top - height_increase * height)
    right = int(right + width_increase * width)
    bot = int(bot + height_increase * height)

    top, bot, left, right = max(0, top), min(bot, frame_shape[0]),\
                            max(0, left), min(right, frame_shape[1])
    h, w = bot - top, right - left

    start = start / fps
    end = end / fps
    time = end - start

    scale = f'{image_shape[0]}:{image_shape[1]}'

    filename = path_save_output + "/" + bname + '_' + str(count) + '.mp4'

    return f'ffmpeg -y -nostdin -hide_banner -loglevel error -i {inp} -ss {start} -t {time} -filter:v "crop={w}:{h}:{left}:{top}, scale={scale}" -vcodec libx264 {filename}'


def compute_bbox_trajectories(trajectories, fps, frame_shape, path_input_video,\
                                                        count, path_save_output):
    bname = os.path.basename(os.path.splitext(path_input_video)[0])
    commands = []
    for i, (bbox, tube_bbox, start, end) in enumerate(trajectories):
        if (end - start) > config["min_frames"]:
            count = count + 1
            command = compute_bbox(start, end, fps, tube_bbox, frame_shape,\
                                inp=path_input_video, image_shape=config["image_shape"],\
                                increase_area=config["increase"], bname=bname,\
                                count=count, path_save_output = path_save_output)

            # print(command)
            os.system(command)
            commands.append(command)
    return commands



def process_video(inputdata):
    path_video_input, path_save_output, position_ = inputdata
    mp_face_detection = mp.solutions.face_detection
    # global face_detection_model
    face_detection_model = mp_face_detection.FaceDetection(min_detection_confidence=0.5, model_selection=1)

    video = imageio.get_reader(path_video_input)
    count = 0

    trajectories = []
    previous_frame = None
    fps = video.get_meta_data()['fps']
    duration = video.get_meta_data()['duration']
    totalF = fps * duration
    pbar = tqdm(total=totalF, position = position_)
    commands = []
    try:
        # for i, frame in tqdm(enumerate(video)):
        for i, frame in enumerate(video):
            frame_shape = frame.shape
            bboxes =  extract_bbox(frame, face_detection_model)
            # print(bboxes)
            ## For each trajectory check the criterion
            not_valid_trajectories = []
            valid_trajectories = []

            for trajectory in trajectories:
                tube_bbox = trajectory[0]
                intersection = 0
                for bbox in bboxes:
                    intersection = max(intersection, bb_intersection_over_union(tube_bbox, bbox))
                if intersection > config["iou_with_initial"]:
                    valid_trajectories.append(trajectory)
                else:
                    not_valid_trajectories.append(trajectory)
            count = len(commands)
            commands += compute_bbox_trajectories(not_valid_trajectories, fps,\
                                    frame_shape, path_video_input, count, path_save_output)

            trajectories = valid_trajectories

            ## Assign bbox to trajectories, create new trajectories
            for bbox in bboxes:
                intersection = 0
                current_trajectory = None
                for trajectory in trajectories:
                    tube_bbox = trajectory[0]
                    current_intersection = bb_intersection_over_union(tube_bbox, bbox)
                    if intersection < current_intersection\
                    and current_intersection > config["iou_with_initial"]:
                        intersection = bb_intersection_over_union(tube_bbox, bbox)
                        current_trajectory = trajectory

                ## Create new trajectory
                if current_trajectory is None:
                    trajectories.append([bbox, bbox, i, i])
                else:
                    current_trajectory[3] = i
                    current_trajectory[1] = join(current_trajectory[1], bbox)

            pbar.update(1)


    except IndexError as e:
        raise (e)

    count = len(commands)
    commands += compute_bbox_trajectories(trajectories, fps,\
                        frame_shape, path_video_input, count, path_save_output)
    pbar.close()
    # return commands

def bbox_by_RetinaFace(frame_input, mobile_net, resnet_net, device):
    bboxes = []
    l_coordinate, detected_face = detection_face(mobile_net,resnet_net, frame_input, device)
    if detected_face:
        topleft, bottomright = l_coordinate[0]
        bboxes.append([topleft[0], topleft[1], bottomright[0], bottomright[1]])

    return np.array(bboxes)

def process_video_by_RetinaFace(path_video_input, path_save_output, mobile_net, resnet_net):

    mp_face_detection = mp.solutions.face_detection
    # global face_detection_model
    face_detection_model = mp_face_detection.FaceDetection(min_detection_confidence=0.5, model_selection=1)

    video = imageio.get_reader(path_video_input)
    count = 0
    device = torch.device("cuda")

    trajectories = []
    previous_frame = None
    fps = video.get_meta_data()['fps']
    duration = video.get_meta_data()['duration']
    totalF = fps * duration
    pbar = tqdm(total=totalF)
    commands = []
    try:
        # for i, frame in tqdm(enumerate(video)):
        for i, frame in enumerate(video):
            frame_shape = frame.shape
            # bboxes =  extract_bbox(frame, face_detection_model)
            bboxes =  bbox_by_RetinaFace(frame, mobile_net, resnet_net, device)
            # print(bboxes)
            ## For each trajectory check the criterion
            not_valid_trajectories = []
            valid_trajectories = []

            for trajectory in trajectories:
                tube_bbox = trajectory[0]
                intersection = 0
                for bbox in bboxes:
                    intersection = max(intersection, bb_intersection_over_union(tube_bbox, bbox))
                if intersection > config["iou_with_initial"]:
                    valid_trajectories.append(trajectory)
                else:
                    not_valid_trajectories.append(trajectory)
            count = len(commands)
            commands += compute_bbox_trajectories(not_valid_trajectories, fps,\
                                    frame_shape, path_video_input, count, path_save_output)

            trajectories = valid_trajectories

            ## Assign bbox to trajectories, create new trajectories
            for bbox in bboxes:
                intersection = 0
                current_trajectory = None
                for trajectory in trajectories:
                    tube_bbox = trajectory[0]
                    current_intersection = bb_intersection_over_union(tube_bbox, bbox)
                    if intersection < current_intersection\
                    and current_intersection > config["iou_with_initial"]:
                        intersection = bb_intersection_over_union(tube_bbox, bbox)
                        current_trajectory = trajectory

                ## Create new trajectory
                if current_trajectory is None:
                    trajectories.append([bbox, bbox, i, i])
                else:
                    current_trajectory[3] = i
                    current_trajectory[1] = join(current_trajectory[1], bbox)

            pbar.update(1)


    except IndexError as e:
        raise (e)

    count = len(commands)
    commands += compute_bbox_trajectories(trajectories, fps,\
                        frame_shape, path_video_input, count, path_save_output)
    pbar.close()

def process_crop_face(path_video_input, path_save_output, num_process):
    if os.path.isfile(path_video_input):
        process_video((path_video_input, path_save_output, 0))
    else:
        video_paths = []
        for video_name in os.listdir(path_video_input):
            path_video = os.path.join(path_video_input, video_name)
            video_paths.append((path_video, path_save_output, len(video_paths)))

        pool = Pool(num_process)
        # for _ in tqdm(pool.imap_unordered(process_video, [file for file in video_paths]), total=len(video_paths)):
        for _ in pool.imap_unordered(process_video, [file for file in video_paths]):
        	pass
        pool.close()
        pool.join()


    # commands = process_video(path_video_input, path_save_output)
    # for command in commands:
    #     print(command)

if __name__ == '__main__':
    mobile_net, resnet_net = loadmodelface()
    process_video_by_RetinaFace('/home/thotd/Downloads/DrDisrespectEggs.mp4', '/home/thotd/Documents/streamer_interactive/TeethDetector/visualize', mobile_net, resnet_net)
