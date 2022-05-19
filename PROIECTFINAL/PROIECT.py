from __future__ import (
    division, absolute_import, print_function, unicode_literals)
import cv2
import numpy as np
import os
import glob
from matplotlib import pyplot as plt
from PIL import Image
from skimage.io import imread, imshow
from skimage import img_as_ubyte
from matplotlib.patches import Rectangle

def debayer(infile):
    im = Image.open(infile)
    rgb_im = im.convert('RGB')
    output=infile[0:-4]+".jpg"
    rgb_im.save(output)
    return output

def autodefisheye(img_calib,img_fisheye) :
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

    # print("Camera matrix : \n")
    # print(mtx)
    # print("dist : \n")
    # print(dist)
    # print("rvecs : \n")
    # print(rvecs)
    # print("tvecs : \n")
    # print(tvecs)
    # print(w)
    # print(h)

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
    output = img_fisheye[0:-4] + ".jpg"
    cv2.imwrite(output,crop_dst)

def defisheye(img_fisheye,mtx,dist):
    # Refining the camera matrix using parameters obtained by calibration
    # w=600
    # h=800
    h, w = img_fisheye.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    dim = (600, 800)
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
    output = img_fisheye[0:-4]+".jpg"
    cv2.imwrite(output, crop_dst)
    return output

def denoise(image) :
    img = cv2.imread(image)
    plt.style.use('seaborn')

    cor = cv2.fastNlMeansDenoisingColored(img, None, 11, 6, 7, 21)
    output = image[0:-4]+".jpg"
    cv2.imwrite(output,cor)
    return output

def resize(image,width,height) :
    img = Image.open(image)
    new_image = img.resize((int(width), int(height)))
    output = image[0:-4]+".jpg"
    new_image.save(output)

def white_balance(image):
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
    output = image[0:-4]+".jpg"
    cv2.imwrite(output,result)
    return output

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

def greyscale(input):
    image = cv2.imread(input)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    output = input[0:-4]+".jpg"
    cv2.imwrite(output, gray)
    return output

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

def autoExposureCor(input):
    image = cv2.imread(input)
    auto_result, alpha, beta = autoExposureAlgorithm(image)
    output = input[0:-4]+".jpg"
    cv2.imwrite(output, auto_result)
    cv2.waitKey()
    return output

def breakvideo(input):
    cap = cv2.VideoCapture(input)
    i = 0
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        output = "VIDEOSPART/" + input[0:-4]
        cv2.imwrite(output + str(i) + '.jpg', frame)
        i += 1
    return i

    cap.release()
    cv2.destroyAllWindows()

def interface():
    print("Your input is:\nA video (type 1)\nA raw photo (type 2)")
    inputfile = input()
    if inputfile == "2":
        print("Raw photo path: ")
        rawphoto = input()
        print("Should the image be black and white(1) or colored(2): ")
        color = input()
        print("Undistort the fisheye effect?Y/N")
        fisheye = input()
        if fisheye.lower() == "y":
            print("Do you have the camera parameters?Y/N")
            param = input()
            if param.lower() == "n":
                print("Type the path to the checkerboard images: ")
                checkerboard = input()
            elif param.lower() == "y":
                print("Type the intrinsic camera parameters.")
                print("Intrinsic camera matrix: \n")
                cammtx = input()
                print("Lens distortion coefficients: ")
                distcoef = input()
        print("White Balance?Y/N")
        whitebal = input()
        print("Denoising?Y/N")
        deno = input()
        print("Autoexposure Correction?Y/N")
        autoexp = input()
        print("Resize the image?Y/N")
        res = input()
        if res.lower() == "y":
            print("Height: ")
            height = input()
            print("Width: ")
            width = input()

        debayer(rawphoto)
        debphoto = debayer(rawphoto)
        if color == "1":
            greyscale(debphoto)
            colphoto = greyscale(debphoto)
        elif color == "2":
            colphoto = debphoto

        if fisheye.lower() == "y":
            if param.lower() == "n":
                autodefisheye(checkerboard, colphoto)
                imgdefish = colphoto[0:-4] + ".jpg"
            elif param.lower() == "n":
                defisheye(colphoto, cammtx, distcoef)
                imgdefish = colphoto[0:-4] + ".jpg"
        else:
            imgdefish = colphoto

        if whitebal.lower() == "y":
            white_balance(imgdefish)
            whitebalimg = white_balance(imgdefish)
        else:
            whitebalimg = imgdefish

        if deno.lower() == "y":
            denoise(whitebalimg)
            imgdenoise = denoise(whitebalimg)
        else:
            imgdenoise = whitebalimg

        if autoexp.lower() == "y":
            autoExposureCor(imgdenoise)
            imgcor = autoExposureCor(imgdenoise)
        else:
            imgcor = imgdenoise

        if res.lower() == "y":
            resize(imgcor, width, height)
    elif inputfile == "1":
        print("Video path: ")
        video = input()
        breakvideo(video)
        x=breakvideo(video)
        print(x)

        print("Should the images be black and white(1) or colored(2): ")
        color = input()
        print("Undistort the fisheye effect?Y/N")
        fisheye = input()
        if fisheye.lower() == "y":
            print("Do you have the camera parameters?Y/N")
            param = input()
            if param.lower() == "n":
                print("Type the path to the checkerboard images: ")
                checkerboard = input()
            elif param.lower() == "y":
                print("Type the intrinsic camera parameters.")
                print("Intrinsic camera matrix: \n")
                cammtx = input()
                print("Lens distortion coefficients: ")
                distcoef = input()
        print("White Balance?Y/N")
        whitebal = input()
        print("Denoising?Y/N")
        deno = input()
        print("Autoexposure Correction?Y/N")
        autoexp = input()
        print("Resize the images?Y/N")
        res = input()
        if res.lower() == "y":
            print("Height: ")
            height = input()
            print("Width: ")
            width = input()

        for i in range (x):
            photo= "VIDEOSPART/" + video[0:-4]+str(i)+ ".jpg"

            if color == "1":
                greyscale(photo)
                colphoto = greyscale(photo)

            elif color == "2":
                colphoto = photo

            if fisheye.lower() == "y":
                if param.lower() == "n":
                    autodefisheye(checkerboard, colphoto)
                    imgdefish = colphoto[0:-4] + ".jpg"
                elif param.lower() == "n":
                    defisheye(colphoto, cammtx, distcoef)
                    imgdefish = colphoto[0:-4] + ".jpg"
            else:
                imgdefish = colphoto

            if whitebal.lower() == "y":
                white_balance(imgdefish)
                whitebalimg = white_balance(imgdefish)
            else:
                whitebalimg = imgdefish

            if deno.lower() == "y":
                denoise(whitebalimg)
                imgdenoise = denoise(whitebalimg)
            else:
                imgdenoise = whitebalimg

            if autoexp.lower() == "y":
                autoExposureCor(imgdenoise)
                imgcor = autoExposureCor(imgdenoise)
            else:
                imgcor = imgdenoise

            if res.lower() == "y":
                resize(imgcor, width, height)



interface()
#B:/OTHERS/PROIECT/FOLDERVIDEO/videofish.mp4
#B:/OTHERS/PROIECT/FOLDER/FISHEYE.raw
#B:/OTHERS/PROIECT/AGAIN2/calib*.jpg