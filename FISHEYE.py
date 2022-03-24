# #Opens the Video file and segment it
# import cv2
# cap = cv2.VideoCapture('D:/IDE Pycharm LP2/Program/Projects/Fisheye/video/video_editat.mp4')
# i = 0
# while (cap.isOpened()):
#     ret, frame = cap.read()
#     if ret == False:
#         break
#     cv2.imwrite('img' + str(i) + '.jpg', frame)
#     i += 1
#
# cap.release()
# cv2.destroyAllWindows()

################################

import pickle

import cv2
import numpy as np
from cv2.fisheye import undistortImage
from defisheye import Defisheye, plt

dtype = 'linear'
format = 'fullframe'
fov = 180
pfov = 120

img = "D:/IDE Pycharm LP2/Program/Projects/Fisheye/img614.jpg"
imgout = f"D:/IDE Pycharm LP2/Program/Projects/Fisheye/image_out/newimag{dtype}{format}{pfov}_{fov}.jpg"

obj = Defisheye(img, dtype=dtype, format=format, fov=fov, pfov=pfov)
obj.convert(imgout)

########################################

import cv2

image = cv2.imread('D:/IDE Pycharm LP2/Program/Projects/Fisheye/image_out/newimaglinearfullframe120_180.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imshow('Original image',image)
cv2.imshow('Gray image', gray)

cv2.waitKey(0)
cv2.destroyAllWindows()
