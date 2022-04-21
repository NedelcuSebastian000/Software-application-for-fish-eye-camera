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

def debayer(infile,output):
    with rawpy.imread(infile) as raw:
        rawimg = raw.raw_image
        rawimg = cv2.normalize(raw.raw_image, dst=None, alpha=0, beta=65535, norm_type=cv2.NORM_MINMAX)
        rawimg = cv2.cvtColor(rawimg, cv2.COLOR_BayerRGGB2RGB)
        rawimg = cv2.resize(rawimg, (0, 0), fx=0.3, fy=0.3)
        img8 = rawimg / 255
        img8 = img8.astype(np.uint8)
        cv2.imwrite('B:/OTHERS/PROIECT/OUTPUTFISHEYE/{}.jpg'.format(output), img8)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def defisheye(img_calib,img_fisheye,output) :
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
    dim = (600, 800)
    images = glob.glob(img_calib)
    for fname in images:
        img = cv2.imread(fname)
        img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
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


    cv2.destroyAllWindows()

    h, w = img.shape[:2]

    """
    Performing camera calibration by 
    passing the value of known 3D points (objpoints)
    and corresponding pixel coordinates of the 
    detected corners (imgpoints)
    """
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    #
    # print("Camera matrix : \n")
    # print(mtx)
    # print("dist : \n")
    # print(dist)
    # print("rvecs : \n")
    # print(rvecs)
    # print("tvecs : \n")
    # print(tvecs)

    # Refining the camera matrix using parameters obtained by calibration
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    imgDIST = cv2.imread(img_fisheye)
    imgDIST = cv2.resize(imgDIST, dim, interpolation=cv2.INTER_AREA)

    # Undistort the image
    dst = cv2.undistort(imgDIST, mtx, dist, None, newcameramtx)

    dim2 = (600, 600)
    x = 90
    y = 180
    h1 = 400
    w1 = 410
    crop_dst = dst[y:y + h1, x:x + w1]
    crop_dst = cv2.resize(crop_dst, dim2, interpolation=cv2.INTER_AREA)

    # Displaying the undistorted image
    cv2.imwrite('B:/OTHERS/PROIECT/OUTPUTFISHEYE/{}.jpg'.format(output),crop_dst)

def denoise(image,output) :
    img = cv2.imread(image)
    plt.style.use('seaborn')

    cor = cv2.fastNlMeansDenoisingColored(img, None, 11, 6, 7, 21)

    cv2.imwrite('B:/OTHERS/PROIECT/OUTPUTFISHEYE/{}.jpg'.format(output),cor)

def resize(image,width,height,output) :
    img = Image.open(image)
    new_image = img.resize((int(width), int(height)))
    new_image.save('B:/OTHERS/PROIECT/OUTPUTFISHEYE/{}.jpg'.format(str(output)))

def white_balance(image,output):
    img = cv2.imread(image)
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
    cv2.imwrite('B:/OTHERS/PROIECT/OUTPUTFISHEYE/{}.jpg'.format(output),result)

def scale(image, output):
    img = cv2.imread(image)
    layer = img.copy()
    x=0
    for i in range(4):
        plt.subplot(2, 2, i + 1)

        # using pyrDown() function
        layer = cv2.pyrDown(layer)
        x=x+1
        cv2.imwrite(output.format(i),layer)

    cv2.destroyAllWindows()

def greyscale(input,output):
    image = cv2.imread(input)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(output, gray)

def autoExposureAlgorithm(image, clip_hist_percent=25):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate grayscale histogram
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_size = len(hist)

    # Calculate cumulative distribution from the histogram
    accumulator = []
    accumulator.append(float(hist[0]))
    for index in range(1, hist_size):
        accumulator.append(accumulator[index - 1] + float(hist[index]))

    # Locate points to clip
    maximum = accumulator[-1]
    clip_hist_percent *= (maximum / 100.0)
    clip_hist_percent /= 2.0

    # Locate left cut
    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1

    # Locate right cut
    maximum_gray = hist_size - 1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1

    # Calculate alpha and beta values
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha

    '''
    # Calculate new histogram with desired range and show histogram 
    new_hist = cv2.calcHist([gray],[0],None,[256],[minimum_gray,maximum_gray])
    plt.plot(hist)
    plt.plot(new_hist)
    plt.xlim([0,256])
    plt.show()
    '''

    auto_result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return (auto_result, alpha, beta)

def autoExposureCor(input,output):
    image = cv2.imread(input)
    auto_result, alpha, beta = autoExposureAlgorithm(image)
    cv2.imwrite(output, auto_result)
    cv2.waitKey()



debayer('B:/OTHERS/PROIECT/Raw/malph1.dng','debayered')
defisheye('B:/OTHERS/PROIECT/again2/calib*.jpg','B:/OTHERS/PROIECT/again2/fisheye1.jpg','defisheyed')
denoise('B:/OTHERS/PROIECT/OUTPUTFISHEYE/defisheyed.jpg','denoised')
resize('B:/OTHERS/PROIECT/OUTPUTFISHEYE/denoised.jpg',1000,1000,'resized')
white_balance('B:/OTHERS/PROIECT/OUTPUTFISHEYE/resized.jpg','whitebalanced')
autoExposureCor('B:/OTHERS/PROIECT/OUTPUTFISHEYE/whitebalanced.jpg','B:/OTHERS/PROIECT/OUTPUTFISHEYE/exp.jpg')
greyscale('B:/OTHERS/PROIECT/OUTPUTFISHEYE/exp.jpg','B:/OTHERS/PROIECT/OUTPUTFISHEYE/gray.jpg')
scale('B:/OTHERS/PROIECT/OUTPUTFISHEYE/gray.jpg','B:/OTHERS/PROIECT/OUTPUTFISHEYE/{}.jpg')