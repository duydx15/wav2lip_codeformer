"""
"""
# from retinaface import RetinaFace
import os
import numpy as np
import math
import cv2


class AffineTransform():
    def __init__(self, mat=None):
        self.mat = mat

    def __str__(self):
        return str(self.mat)

    @staticmethod
    def get_rot_mat(center, angle, scale):
        center = float(center[0]), float(center[1]) # to have valid type of center in getRotationMatrix2D
        rot_mat = cv2.getRotationMatrix2D(center, angle, scale)
        return rot_mat

    @staticmethod
    def get_mat_from_3pts(src, dst):
        mat = cv2.getAffineTransform(src, dst)
        return mat

    def transform_points(self, points):
        if not isinstance(points, np.ndarray):
            points = np.float32(points)

        _ones = np.ones((points.shape[0], 1), dtype=np.float32)
        points = np.concatenate((points, _ones), axis=1)
        return np.matmul(points, np.transpose(self.mat))

    def transform_img(self, img, H, W):
        return cv2.warpAffine(img, self.mat, (W, H))

    def invert(self):
        inv_mat = cv2.invertAffineTransform(self.mat)
        return AffineTransform(inv_mat)

def get_angle(point1, point2):
    """
    return the angle between vector Ox and vector (point1, point2)
    -180 < angle <= 180
    ---------------------
    arguments:
    - point1, point2: nparray
    """
    w, h = point2 - point1
    if w < 0 and h >= 0:
        return 180+ math.atan(h/w) * 180/math.pi
    elif w < 0 and h < 0:
        return -180 + math.atan(h/w) * 180/math.pi
    return math.atan(h/w) * 180/math.pi

def get_point(org_point, angle, distance):
    """
    """
    angle = math.radians(angle)
    h = distance * math.sin(angle)
    w = distance * math.cos(angle)
    return (org_point[0] + w, org_point[1] + h)

# def get_face_bboxes_and_landmarks_by_retina(img, threshold, view=False):
    # """
    # bbox: [l, t, r, b]
    # landmarks: right_eye, left_eye, nose, mouth_right, mouth_left
    # """
    # resp = RetinaFace.detect_faces(img)
    # if isinstance(resp, tuple):
    #     return None
    #
    # bboxes = []
    # landmarks = []
    # for face in resp.keys():
    #     info = resp.get(face)
    #     if info.get('score') > threshold:
    #         bbox = info.get('facial_area')
    #         bboxes.append(bbox)
    #
    #         lmrks = info.get('landmarks')
    #         right_eye = lmrks.get('right_eye')
    #         left_eye = lmrks.get('left_eye')
    #         nose = lmrks.get('nose')
    #         mouth_right = lmrks.get('mouth_right')
    #         mouth_left = lmrks.get('mouth_left')
    #
    #         # swap right and left eye if needed
    #         if right_eye[0] > left_eye[0]:
    #             tmp = right_eye
    #             right_eye = left_eye
    #             left_eye = tmp
    #
    #         landmarks.append((right_eye, left_eye, nose, mouth_right, mouth_left))
    # if view:
    #     img_copy = img.copy()
    #
    #     for l, t, r, b in bboxes:
    #         cv2.rectangle(img_copy, (l, t), (r, b), (0, 255, 0), 3)
    #
    #     for lmrks in landmarks:
    #         for point in lmrks:
    #             point = (int(point[0]), int(point[1]))
    #             cv2.circle(img_copy, point, 2, (255, 0, 0), -1)
    #
    #     cv2.imshow('Img', img_copy)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    # return bboxes, landmarks

def _show_point(img, point):
    point = (int(point[0]), int(point[1]))
    cv2.circle(img, point, 3, (0, 0, 255), -1)

def add_image_by_mask(img1, img2, mask_):
    mask_not = cv2.bitwise_not(mask_)
    img2_no_mask = cv2.bitwise_and(img2, img2, mask=mask_not)
    img1_mask_only = cv2.bitwise_and(img1, img1, mask=mask_)
    return cv2.add(img2_no_mask, img1_mask_only)

def merge_wav2lip(face, img, tf_mat):
    """
    """
    H, W = img.shape[:2]
    h, w = face.shape[:2]
    face = tf_mat.invert().transform_img(face, H, W)

    pts = np.float32([
        [0, 0],
        [0, h],
        [w, h],
        [w, 0]
    ])

    global_pts = tf_mat.invert().transform_points(pts)
    global_pts = np.int32(global_pts)

    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, [global_pts], 255, cv2.LINE_8)

    merged = add_image_by_mask(face, img, mask)
    return merged

def crop_aligned_face(img, bbox, lmrks, target_ratio, scale, view=False):
    """
    return cropped, aligned face and the corresponding affine transform matrix
    """
    lmrks = np.float32(lmrks)

    # define the size to crop
    l, t, r, b = bbox
    face_w = (r - l) * scale
    face_h = (b - t) * scale

    if face_w / target_ratio[0] > face_h / target_ratio[1]:
        crop_W = face_w
        crop_H = face_w / target_ratio[0] * target_ratio[1]
    else:
        crop_H = face_h
        crop_W = face_h / target_ratio[1] * target_ratio[0]

    crop_H, crop_W = int(crop_H), int(crop_W)

    # get eyes and mouths
    right_eye, left_eye, mouth_right, mouth_left = lmrks[0], lmrks[1], lmrks[3], lmrks[4]

    # align so that middle eye and middle mouth are on a vertical line
    middle_eye = (right_eye + left_eye) / 2
    middle_mouth = (mouth_right + mouth_left) / 2
    angle = get_angle(middle_eye, middle_mouth)

    # center
    center = get_point(middle_mouth, angle+180, crop_H/4)

    middle_bot = get_point(center, angle, crop_H/2)
    middle_left = get_point(center, angle+90, crop_W/2)

    global_points = np.float32([
        center,
        middle_bot,
        middle_left
    ])

    local_points = np.float32([
        [crop_W//2, crop_H//2],
        [crop_W//2, crop_H],
        [0, crop_H//2]
    ])

    mat = AffineTransform.get_mat_from_3pts(global_points, local_points)
    transformation = AffineTransform(mat)

    cropped_img = transformation.transform_img(img, crop_H, crop_W)

    if view:
        img_copy = img.copy()
        middle_bot = np.float32(middle_bot)
        middle_left = np.float32(middle_left)
        center = np.float32(center)

        middle_top = center*2 - middle_bot
        middle_right = center*2 - middle_left

        left_top = middle_left + middle_top - center
        right_top = middle_right + middle_top - center
        right_bot = middle_right + middle_bot - center
        left_bot = middle_left + middle_bot - center

        pts = np.int32([left_top, right_top, right_bot, left_bot])
        cv2.polylines(img_copy, [pts], 1, (0, 255, 0), 3)
        cv2.imshow('Img', img_copy)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return cropped_img, transformation


def crop_dfl_image(img, tf_mat, w, h, scale=1.0):
    """
    w, h is the size of cropped face
    """
    dfl_size = int(max(w, h) * scale)
    pts = np.float32([
        [w/2, h/2],
        [w/2 - dfl_size/2, h/2],
        [w/2, h/2 + dfl_size/2]
    ])

    global_pts = tf_mat.invert().transform_points(pts)
    global_pts = np.float32(global_pts)

    new_pts = np.float32([
        [dfl_size/2, dfl_size/2],
        [0, dfl_size/2],
        [dfl_size/2, dfl_size]
    ])
    new_mat = AffineTransform.get_mat_from_3pts(global_pts, new_pts)
    new_mat = AffineTransform(new_mat)
    dfl_img = new_mat.transform_img(img, dfl_size, dfl_size)

    return dfl_img, new_mat



# def main():
#     # imgpath = '/home/anlab/Pictures/daddario.jpg'
#     # imgpath = '/home/anlab/Pictures/putin.jpg'
#     imgpath = '/home/anlab/Pictures/90.png'
#     img = cv2.imread(imgpath)
#
#     res = get_face_bboxes_and_landmarks_by_retina(img, 0.9, False)
#     if res is not None:
#         bboxes, landmarks = res
#
#     for bbox, lmrks in zip(bboxes, landmarks):
#         crop_aligned_face(img, bbox, lmrks, (3, 4), 0.8, True)
#
# if __name__=='__main__':
#     main()
