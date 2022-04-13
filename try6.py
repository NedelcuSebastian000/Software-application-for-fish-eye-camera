from __future__ import (
    division, absolute_import, print_function, unicode_literals)
import cv2
import numpy as np
import os
import glob
from matplotlib import pyplot as plt
from PIL import Image
import rawpy
from skimage.io import imread, imshow
from skimage import img_as_ubyte
from matplotlib.patches import Rectangle


# Defining the dimensions of checkerboard
CHECKERBOARD = (6, 9)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Creating vector to store vectors of 3D points for each checkerboard image
objpoints = []
# Creating vector to store vectors of 2D points for each checkerboard image
imgpoints = []

# Defining the world coordinates for 3D points
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
prev_img_shape = None

# Extracting path of individual image stored in a given directory
dim=(600,800)
images = glob.glob('B:/OTHERS/PROIECT/again2/calib*.jpg')
for fname in images:
    img = cv2.imread(fname)
    img=cv2.resize(img,dim,interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    # If desired number of corners are found in the image then ret = true
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD,
                                             cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

    """
    If desired number of corner are detected,
    we refine the pixel coordinates and display 
    them on the images of checker board
    """
    if ret == True:
        objpoints.append(objp)
        # refining pixel coordinates for given 2d points.
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)

    cv2.imshow('img', img)
    cv2.waitKey(1)

cv2.destroyAllWindows()

h, w = img.shape[:2]

"""
Performing camera calibration by 
passing the value of known 3D points (objpoints)
and corresponding pixel coordinates of the 
detected corners (imgpoints)
"""
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print("Camera matrix : \n")
print(mtx)
print("dist : \n")
print(dist)
print("rvecs : \n")
print(rvecs)
print("tvecs : \n")
print(tvecs)



# Refining the camera matrix using parameters obtained by calibration
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))


imgDIST = cv2.imread('B:/OTHERS/PROIECT/again2/fisheye1.jpg')
imgDIST = cv2.resize(imgDIST,dim,interpolation=cv2.INTER_AREA)

# Method 1 to undistort the image
dst = cv2.undistort(imgDIST, mtx, dist, None, newcameramtx)

# Method 2 to undistort the image
# mapx,mapy=cv2.initUndistortRectifyMap(mtx,dist,None,newcameramtx,(w,h),5)
#
# dst = cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)
dim2=(600,600)
x=90
y=180
h1=400
w1=410
crop_dst = dst[y:y+h1,x:x+w1]
crop_dst = cv2.resize(crop_dst,dim2,interpolation=cv2.INTER_AREA)

# Displaying the undistorted image
cv2.imshow("undistorted image",crop_dst)
cv2.waitKey(0)
print("done")


plt.style.use('seaborn')

cor = cv2.fastNlMeansDenoisingColored(crop_dst, None, 11, 6, 7, 21)

row, col = 1, 2
fig, axs = plt.subplots(row, col, figsize=(15, 10))
fig.tight_layout()
axs[0].imshow(cv2.cvtColor(crop_dst, cv2.COLOR_BGR2RGB))
axs[0].set_title('Before')
axs[1].imshow(cv2.cvtColor(cor, cv2.COLOR_BGR2RGB))
axs[1].set_title('Denoising')
plt.show()


image4 = Image.open('B:/OTHERS/PROIECT/POZEFISH/1.jpg')
#a=input()
#b=input()
a=400
b=400
new_image = image4.resize((int(a), int(b)))
new_image.save('B:/OTHERS/PROIECT/POZEFISH/after.jpg')

print(image4.size)
print(new_image.size)


#auto white balance

def show(final):
    print('display')
    cv2.imshow('Temple', final)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Insert any filename with path
img = cv2.imread('B:/OTHERS/PROIECT/POZEFISH/333.jpg')

def white_balance_loops(img):
    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    for x in range(result.shape[0]):
        for y in range(result.shape[1]):
            l, a, b = result[x, y, :]
            # fix for CV correction
            l *= 100 / 255.0
            result[x, y, 1] = a - ((avg_a - 128) * (l / 100.0) * 1.1)
            result[x, y, 2] = b - ((avg_b - 128) * (l / 100.0) * 1.1)
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    return result

final = np.hstack((cor, white_balance_loops(cor)))
show(final)
#cv2.imwrite('result.jpg', final)

img9 = cv2.imread('B:/OTHERS/PROIECT/again2/fisheye1.jpg')

layer = img9.copy()

for i in range(4):
    plt.subplot(2, 2, i + 1)

    # using pyrDown() function
    layer = cv2.pyrDown(layer)

    plt.imshow(layer)
    cv2.imshow("str(i)", layer)
    cv2.waitKey(0)


cv2.destroyAllWindows()

