# Import modules
import sys, cv2, time
import numpy as np
from debug_LipsyncMount import get_face
import mediapipe as mp
import math
from scipy.spatial import Delaunay
# detect facial landmarks in image
# def getLandmarks(faceDetector, landmarkDetector, im, FACE_DOWNSAMPLE_RATIO = 1):
#   points = []
#   imSmall = cv2.resize(im,None,
# 					   fx=1.0/FACE_DOWNSAMPLE_RATIO,
# 					   fy=1.0/FACE_DOWNSAMPLE_RATIO,
# 					   interpolation = cv2.INTER_LINEAR)
#
#   faceRects = faceDetector(imSmall, 0)
#
#   if len(faceRects) > 0:
# 	  maxArea = 0
# 	  maxRect = None
# 	  # TODO: test on images with multiple faces
# 	  for face in faceRects:
# 		  if face.area() > maxArea:
#
# 			  maxArea = face.area()
# 			  maxRect = [face.left(), face.top(), face.right(), face.bottom()]
#
# 	scaledRect = dlib.rectangle(int(rect.left()*FACE_DOWNSAMPLE_RATIO),
# 							 int(rect.top()*FACE_DOWNSAMPLE_RATIO),
# 							 int(rect.right()*FACE_DOWNSAMPLE_RATIO),
# 							 int(rect.bottom()*FACE_DOWNSAMPLE_RATIO))
#
# 	landmarks = landmarkDetector(im, scaledRect)
# 	points = dlibLandmarksToPoints(landmarks)
#   return points

# convert Dlib shape detector object to list of tuples
def dlibLandmarksToPoints(shape):
	points = []
	for p in shape.parts():
		pt = (p.x, p.y)
		points.append(pt)
	return points

# Compute similarity transform given two sets of two points.
# OpenCV requires 3 pairs of corresponding points.
# We are faking the third one.
def similarityTransform(inPoints, outPoints):
	s60 = math.sin(60*math.pi/180)
	c60 = math.cos(60*math.pi/180)

	inPts = np.copy(inPoints).tolist()
	outPts = np.copy(outPoints).tolist()

	# The third point is calculated so that the three points make an equilateral triangle
	xin = c60*(inPts[0][0] - inPts[1][0]) - s60*(inPts[0][1] - inPts[1][1]) + inPts[1][0]
	yin = s60*(inPts[0][0] - inPts[1][0]) + c60*(inPts[0][1] - inPts[1][1]) + inPts[1][1]

	inPts.append([np.int(xin), np.int(yin)])
	xout = c60*(outPts[0][0] - outPts[1][0]) - s60*(outPts[0][1] - outPts[1][1]) + outPts[1][0]
	yout = s60*(outPts[0][0] - outPts[1][0]) + c60*(outPts[0][1] - outPts[1][1]) + outPts[1][1]

	outPts.append([np.int(xout), np.int(yout)])

	# Now we can use estimateAffine2D for calculating the similarity transform.
	tform = cv2.estimateAffine2D(np.array([inPts]), np.array([outPts]), False)
	return tform

def normalizeImagesAndLandmarks(outSize,imIn,pointsIn):
	h, w = outSize

	# Corners of the eye in the input image
	if len(pointsIn) == 68:
		eyecornerSrc = [pointsIn[36],pointsIn[45]]
	elif len(pointsIn) == 5:
		eyecornerSrc = [pointsIn[2],pointsIn[0]]

	# Corners of the eye i  normalized image
	eyecornerDst = [(np.int(0.3*w),np.int(h/3)),(np.int(0.7*w),np.int(h/3))]

	# Calculate similarity transform
	tform = similarityTransform(eyecornerSrc, eyecornerDst)
	imOut = np.zeros(imIn.shape, dtype=imIn.dtype)

	# Apply similarity transform to input image
	imOut = cv2.warpAffine(imIn, tform[0], (w,h))

	# reshape pointsIn from numLandmarks x 2 to  numLandmarks x 1 x 2
	points2 = np.reshape(pointsIn,(pointsIn.shape[0],1,pointsIn.shape[1]))

	# Apply similarity transform to landmarks
	pointsOut = cv2.transform(points2,tform[0])

	# reshape pointsOut to numLandmarks x 2
	pointsOut = np.reshape(pointsOut,(pointsIn.shape[0],pointsIn.shape[1]))

	return imOut, pointsOut


# Check if a point is inside a rectangle
def rectContains(rect, point):
	if point[0] < rect[0]:
		return False
	elif point[1] < rect[1]:
		return False
	elif point[0] > rect[2]:
		return False
	elif point[1] > rect[3]:
		return False
	return True

# Calculate Delaunay triangles for set of points
# Returns the vector of indices of 3 points for each triangle
def calculateDelaunayTriangles(rect, points):
	# Create an instance of Subdiv2D
	subdiv = cv2.Subdiv2D(rect)
	# print(subdiv)

	# print(points[0])

	# Insert points into subdiv
	for p in points:
		print(p[0][0], p[0][1])
		subdiv.insert((p[0][0], p[0][1]))

	# Get Delaunay triangulation
	triangleList = subdiv.getTriangleList()

	# Find the indices of triangles in the points array
	delaunayTri = []

	for t in triangleList:
		# The triangle returned by getTriangleList is
		# a list of 6 coordinates of the 3 points in
		# x1, y1, x2, y2, x3, y3 format.
		# Store triangle as a list of three points
		pt = []
		pt.append((t[0], t[1]))
		pt.append((t[2], t[3]))
		pt.append((t[4], t[5]))

		pt1 = (t[0], t[1])
		pt2 = (t[2], t[3])
		pt3 = (t[4], t[5])

		if rectContains(rect, pt1) and rectContains(rect, pt2) and rectContains(rect, pt3):
			# Variable to store a triangle as indices from list of points
			ind = []
			# Find the index of each vertex in the points list
			for j in range(0, 3):
				for k in range(0, len(points)):
					if(abs(pt[j][0] - points[k][0]) < 1.0 and abs(pt[j][1] - points[k][1]) < 1.0):
						ind.append(k)
			# Store triangulation as a list of indices
			if len(ind) == 3:
				delaunayTri.append((ind[0], ind[1], ind[2]))
	return delaunayTri

# Apply affine transform calculated using srcTri and dstTri to src and
# output an image of size.
def applyAffineTransform(src, srcTri, dstTri, size):
	# Given a pair of triangles, find the affine transform.
	warpMat = cv2.getAffineTransform(np.float32(srcTri), np.float32(dstTri))

	# Apply the Affine Transform just found to the src image
	dst = cv2.warpAffine(src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

	return dst

def warpPerspectiveTri(img1, img2, t1, t2, img2_new_face):
	rect1 = cv2.boundingRect(t1)
	(x, y, w, h) = rect1
	cropped_triangle = img1[y: y + h, x: x + w]
	cropped_tr1_mask = np.zeros((h, w), np.uint8)

	print(t1[1][0])
	points = np.array([[t1[0][0][0] - x, t1[0][0][1] - y],
					   [t1[1][0][0] - x, t1[1][0][1] - y],
					   [t1[2][0][0] - x, t1[2][0][1] - y]], np.int32)

	cv2.fillConvexPoly(cropped_tr1_mask, points, 255)

	# Lines space
	# cv2.line(lines_space_mask, tr1_pt1, tr1_pt2, 255)
	# cv2.line(lines_space_mask, tr1_pt2, tr1_pt3, 255)
	# cv2.line(lines_space_mask, tr1_pt1, tr1_pt3, 255)
	# lines_space = cv2.bitwise_and(img1, img1, mask=lines_space_mask)



	rect2 = cv2.boundingRect(t2)
	(x, y, w, h) = rect2

	cropped_tr2_mask = np.zeros((h, w), np.uint8)

	points2 = np.array([[t2[0][0][0] - x, t2[0][0][1] - y],
						[t2[1][0][0] - x, t2[1][0][1] - y],
						[t2[2][0][0] - x, t2[2][0][1] - y]], np.int32)

	cv2.fillConvexPoly(cropped_tr2_mask, points2, 255)

	# Warp triangles
	points = np.float32(points)
	points2 = np.float32(points2)
	M = cv2.getAffineTransform(points, points2)
	warped_triangle = cv2.warpAffine(cropped_triangle, M, (w, h))
	warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=cropped_tr2_mask)

	# Reconstructing destination face
	img2_new_face_rect_area = img2_new_face[y: y + h, x: x + w]
	img2_new_face_rect_area_gray = cv2.cvtColor(img2_new_face_rect_area, cv2.COLOR_BGR2GRAY)
	_, mask_triangles_designed = cv2.threshold(img2_new_face_rect_area_gray, 1, 255, cv2.THRESH_BINARY_INV)
	warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=mask_triangles_designed)

	img2_new_face_rect_area = cv2.add(img2_new_face_rect_area, warped_triangle)
	img2_new_face[y: y + h, x: x + w] = img2_new_face_rect_area

	return img2_new_face

	# cv2.imwrite("img2_new_face.png",img2_new_face)



# Warps and alpha blends triangular regions from img1 and img2 to img
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

if __name__ == "__main__":
	mp_face_mesh = mp.solutions.face_mesh
	face_mesh_wide = mp_face_mesh.FaceMesh(min_detection_confidence=0.5)
	face_mesh_small = mp_face_mesh.FaceMesh(min_detection_confidence=0.5)

	videoimg = cv2.imread('/home/thotd/Documents/streamer_interactive/Retinaface_Mediapipe/vlcsnap-2022-04-07-09h50m31s404.jpg')
	faceimg = cv2.imread('/home/thotd/Documents/DeepFake/data/00069.png')
	video_h,video_w, channels = videoimg.shape

	videopts, videopts_out = get_face(face_mesh_wide, videoimg, None)
	facepts, facepts_out = get_face(face_mesh_small, faceimg, None)

	# faceimg = cv2.polylines(faceimg, [facepts_out], True, (0, 0, 255), 1)
	# cv2.imwrite("faceimg.png",faceimg)
	# rect = (0, 0, video_w, video_h)
	# print(len(videopts_out))
	points1 = []
	for value in videopts_out:
		points1.append([value[0][0], value[0][1]])
	points1 = np.array(points1)
	tri = Delaunay(points1)

	img2_new_face = np.zeros((video_h, video_w, channels), np.uint8)
	# warpPerspectiveTri(videoimg, faceimg, tri, videopts_out, facepts_out)


	M, _ = cv2.findHomography(facepts, videopts)
	faceimg = cv2.warpPerspective(faceimg,M,(video_w, video_h))

	facepts_out = cv2.perspectiveTransform(facepts_out.astype(np.float32), M).astype(np.int32)

	tris1 = []
	tris2 = []
	for i in range(0, len(tri.simplices)):
		tri1 = []
		tri2 = []
		for j in range(0, 3):
			tri1.append(videopts_out[tri.simplices[i][j]])
			tri2.append(facepts_out[tri.simplices[i][j]])

		tris1.append(tri1)
		tris2.append(tri2)
	tris1 = np.array(tris1, np.float32)
	tris2 = np.array(tris2, np.float32)

	# videoimgWarped1 = videoimg.copy()
	videoimgWarped = videoimg.copy()
	for i in range(0, len(tris1)):
		warpTriangle(faceimg, videoimgWarped, tris1[i], tris2[i])
	# # cout= 0
	# # kernel = np.ones((5,5),np.uint8)
	# for valtri in tris1:
	# 	mask_blur = np.zeros((video_h,video_w), np.uint8)
	# 	cv2.fillPoly(mask_blur, pts =[np.array(valtri, np.int32)], color=(255,255,255,0))
	# 	center_x = int((valtri[0][0][0] + valtri[1][0][0] + valtri[2][0][0])/3)
	# 	center_y = int((valtri[0][0][1] + valtri[1][0][1] + valtri[2][0][1])/3)
	# 	videoimgWarped1 = cv2.seamlessClone(videoimgWarped, videoimgWarped1, mask_blur, (center_x, center_y), cv2.NORMAL_CLONE)
	#
	# cv2.imwrite("videoimgWarped1.png",videoimgWarped1)

	# videoimgWarped = videoimg.copy()
	# for i in range(0, len(tris1)):
	# 	img2_new_face = warpPerspectiveTri(faceimg, videoimgWarped, tris1[i], tris2[i], img2_new_face)
	# cv2.imwrite("img2_new_face.png",img2_new_face)
	# for i in range(0, len(tris1)):
	# 	warpTriangle(faceimg, videoimgWarped, tris1[i], tris2[i])

		# mask_blur = np.zeros((video_h,video_w), np.uint8)
		# cv2.fillPoly(mask_blur, pts =[np.array(tris1[i], np.int32)], color=(255,255,255,0))
		# valtri = tris2[i]
		# center_x = int((valtri[0][0][0] + valtri[1][0][0] + valtri[2][0][0])/3)
		# center_y = int((valtri[0][0][1] + valtri[1][0][1] + valtri[2][0][1])/3)
		# videoimgWarped = cv2.circle(videoimgWarped, (center_x, center_y), radius=1, color=(0, 0, 255), thickness=-1)
		# videoimgWarped = cv2.seamlessClone(faceimg, videoimg, mask_blur, (center_x, center_y), cv2.NORMAL_CLONE)


	# cv2.imwrite("videoimgWarped.png",videoimgWarped)

	# videopts_up = [videopts_out[0], videopts_out[1], videopts_out[2], videopts_out[3], videopts_out[17],
	# 				videopts_out[13], videopts_out[14], videopts_out[15], videopts_out[16]]
	# # print(videopts_out)
	# videopts_up = np.array(videopts_up, np.int32)
	# videopts_up = videopts_up.reshape(-1,1,2)
	#
	# videopts_down = [videopts_out[3], videopts_out[4], videopts_out[5], videopts_out[6], videopts_out[7],
	# 				videopts_out[8], videopts_out[9], videopts_out[10], videopts_out[11], videopts_out[12],
	# 				videopts_out[13], videopts_out[17]]
	# videopts_down = np.array(videopts_down, np.int32)
	# videopts_down = videopts_down.reshape(-1,1,2)

	mask_blur = np.zeros((video_h,video_w), np.uint8)
	cv2.fillPoly(mask_blur, pts =[videopts_out], color=(255,255,255,0))
	(y, x) = np.where(mask_blur >0)
	(topy, topx) = (np.min(y), np.min(x))
	(bottomy, bottomx) = (np.max(y), np.max(x))
	center = (int((topx+bottomx)/2),int((bottomy+topy)/2))
	# cv2.imwrite("mask_up.jpg",mask_blur)

	# videoimgWarped = cv2.seamlessClone(img2_new_face, videoimgWarped, mask_blur, center, cv2.NORMAL_CLONE)
	# videoimg = cv2.inpaint(videoimg, mask_blur, 1, flags=cv2.INPAINT_TELEA)
	# cv2.imwrite("inpaint.jpg",videoimg)
	output = cv2.seamlessClone(faceimg, videoimg, mask_blur, center, cv2.NORMAL_CLONE)
	cv2.imwrite("aaa.jpg",output)
	# mask_blur = np.zeros((video_h,video_w), np.uint8)
	# cv2.fillPoly(mask_blur, pts =[videopts_down], color=(255,255,255,0))
	# (y, x) = np.where(mask_blur >0)
	# (topy, topx) = (np.min(y), np.min(x))
	# (bottomy, bottomx) = (np.max(y), np.max(x))
	# center = (int((topx+bottomx)/2),int((bottomy+topy)/2))
	# cv2.imwrite("mask_down.jpg",mask_blur)
	#
	#
	# output = cv2.seamlessClone(videoimgWarped, output, mask_blur, center, cv2.MIXED_CLONE)
	# cv2.imwrite("videoframe1.jpg",output)
