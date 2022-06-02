from __future__ import (
    division, absolute_import, print_function, unicode_literals)
import cv2
import numpy as np
import glob
import math
from matplotlib import pyplot as plt
from PIL import Image
import rawpy
import imageio

def debayer(path):
    with rawpy.imread(path) as raw:
        rgb = raw.postprocess(use_auto_wb=True)
    output = path[0:9] + "_DEBAYERED.jpg"
    imageio.imsave(output, rgb)
    res = cv2.imread(output)
    scale_percent = 30  # percent of original size
    width = int(res.shape[1] * scale_percent / 100)
    height = int(res.shape[0] * scale_percent / 100)
    dim = (width, height)
    res = cv2.resize(res, dim, interpolation=cv2.INTER_AREA)
    cv2.imwrite(output, res)
    return output

def calibrate(img_calib):
    CHECKERBOARD = (6, 9)
    subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
    calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC \
                        + cv2.fisheye.CALIB_CHECK_COND + cv2.fisheye.CALIB_FIX_SKEW
    objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    _img_shape = None
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.
    images = glob.glob(img_calib)
    for fname in images:
        img = cv2.imread(fname)
        if _img_shape == None:
            _img_shape = img.shape[:2]
        else:
            assert _img_shape == img.shape[:2], "All images must share the same size."
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD,
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            cv2.cornerSubPix(gray, corners, (3, 3), (-1, -1), subpix_criteria)
            imgpoints.append(corners)
    N_OK = len(objpoints)
    K = np.zeros((3, 3))
    D = np.zeros((4, 1))
    rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
    tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
    rms, _, _, _, _ = \
        cv2.fisheye.calibrate(
            objpoints,
            imgpoints,
            gray.shape[::-1],
            K,
            D,
            rvecs,
            tvecs,
            calibration_flags,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
        )
    # print("Found " + str(N_OK) + " valid images for calibration")
    # print("DIM=" + str(_img_shape[::-1]))
    # print("K=np.array(" + str(K.tolist()) + ")")
    # print("D=np.array(" + str(D.tolist()) + ")")
    DIM=_img_shape[::-1]
    return K, D, DIM

def undistort(img_path,K,D,DIM):
    img = cv2.imread(img_path)
    h,w = img.shape[:2]
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    # cv2.imshow("undistorted", undistorted_img)
    x = 125
    y = 25
    h1 = 825
    w1 = 1000
    undistorted_img = undistorted_img[y:y + h1, x:x + w1]
    output = img_path[0:9] + "_UNDISTORTED.jpg"
    cv2.imwrite(output, undistorted_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return output

def denoise(image) :
    img = cv2.imread(image)
    plt.style.use('seaborn')

    cor = cv2.fastNlMeansDenoisingColored(img, None, 1, 1, 1, 1)
    output = image[0:9]+"_DENOISED.jpg"
    cv2.imwrite(output,cor)
    # cv2.imshow('denoise',cor)
    return output

def denoise2(image,luminance,photo_render,search_window,block_size) :
    """luminance-    Parameter regulating filter strength. Big h value perfectly removes noise but also removes image details,
                 smaller h value preserves details but also preserves some noise
    photo_render  float The same as h but for color components.
                 For most images value equals 10 will be enough to remove colored noise and do not distort colors
    search_window Size in pixels of the window that is used to compute weighted average for given pixel. Should be odd.
                 Affect performance linearly: greater search_window - greater denoising time. Recommended value 21 pixels
    block_size    Size in pixels of the template patch that is used to compute weights. Should be odd.
                 Recommended value 7 pixels
    """
    img = cv2.imread(image)
    plt.style.use('seaborn')

    cor = cv2.fastNlMeansDenoisingColored(img, None, luminance,photo_render , search_window, block_size)
    output = image[0:9]+"_DENOISED.jpg"
    cv2.imwrite(output,cor)
    return output

# def color_correction(image) :
#     img = cv2.imread(image)
#     plt.style.use('seaborn')
#
#     cor = cv2.ColorCorrectionModel(img)
#     output = image[0:-4]+"1.jpg"
#     cv2.imwrite(output,cor)
#     return output

def Enhance(img,contrast):
    """"""
    img2=cv2.imread(img)
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8))
    img_hsv = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(img_hsv)

    V_C = clahe.apply(V)

    def loop(R):
        height_R = R.shape[0]
        width_R = R.shape[1]
        for i in np.arange(height_R):
            for j in np.arange(width_R):
                a = R.item(i, j)
                b = math.ceil(a * float(contrast))
                if b > 255:
                    b = 255
                R.itemset((i, j), b)
        return R

    HSV = cv2.merge((H, S, V_C))
    output = cv2.cvtColor(HSV, cv2.COLOR_HSV2BGR)
    B, G, R = cv2.split(output)
    B = loop(B)
    G = loop(G)
    R = loop(R)
    OUTPUT1 = cv2.merge((B, G, R))
    # cv2.imshow('color enh',OUTPUT1)
    output = img[0:9] + "_ENHANCED.jpg"
    cv2.imwrite(output,OUTPUT1)
    return output

def resize(image,width,height) :
    img = Image.open(image)
    new_image = img.resize((int(width), int(height)))
    output = image[0:9]+"_FINAL_RESIZED.jpg"
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
    output = image[0:9]+"_WHITE_BALANCED.jpg"
    cv2.imwrite(output,result)
    # cv2.imshow('white balance',result)
    return output

def scale(image):
    img = cv2.imread(image)
    layer = img.copy()
    x=0
    output=image[0:-4]+"{}.jpg"
    for i in range(4):
        plt.subplot(2, 2, i + 1)

        # using pyrDown() function
        layer = cv2.pyrDown(layer)
        x=x+1

        cv2.imwrite(output.format(i),layer)

    cv2.destroyAllWindows()

def greyscale(img):
    imgg=cv2.imread(img)
    R, G, B = imgg[:, :, 0], imgg[:, :, 1], imgg[:, :, 2]
    imgGray = 0.2989 * R + 0.5870 * G + 0.1140 * B
    output = img[0:9] + "_GRAY.jpg"
    cv2.imwrite(output,imgGray)
    return output

def autoExposureAlgorithm(image, clip_hist_percent):
    """ clip_hist_percent represents the percentage """
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

def autoExposureCor(input,percent):
    image = cv2.imread(input)
    auto_result, alpha, beta = autoExposureAlgorithm(image,int(percent))
    output = input[0:9]+"_FINAL.jpg"
    cv2.imwrite(output, auto_result)
    # cv2.imshow('exposure',auto_result)
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
        print("Color enhancement?Y/N")
        corr=input()
        if corr.lower() == "y":
            print("Contrast Level(0-100)(Default = 0) :")
            corpr=input()
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
                K = input()
                K = np.array(K)
                K = np.array(str(K.tolist()))
                print("Lens distortion coefficients: ")
                D = input()
                D = np.array(D)
                D = np.array(str(D.tolist()))
                print("Photo dimensions: ")
                print("Width: ")
                width = input()
                print("Height: ")
                height = input()
                DIM=(int(width),int(height))
        print("White Balance?Y/N")
        whitebal = input()
        print("Denoising?Y/N")
        deno = input()
        if deno.lower() == "y":
            print("Luminance(0-100)(Default = 1) :")
            luminance=input()
            print("Photo render distance(0-100)(Default = 1) : ")
            photo_render = input()
            print("Search window(0-100)(Default = 1) : ")
            search_window=input()
            print("Block size(0-100)(Default = 1) : ")
            block_size=input()
        print("Autoexposure Correction?Y/N")
        autoexp = input()
        if autoexp.lower() == "y":
            print("Percentage(0-100)(Default = 0) : ")
            percentage = input()
        print("Resize the image?Y/N")
        res = input()
        if res.lower() == "y":
            print("Height: ")
            height = input()
            print("Width: ")
            width = input()
        print("Pyramid scale?Y/N")
        pir = input()


        debayer(rawphoto)
        debphoto = debayer(rawphoto)
        if corr.lower() == "y":
            Enhance(debphoto,corpr)
            corphoto=Enhance(debphoto,corpr)
        else:
            corphoto = debphoto
        if color == "1":
            greyscale(corphoto)
            colphoto = greyscale(corphoto)
        elif color == "2":
            colphoto = corphoto

        if fisheye.lower() == "y":
            if param.lower() == "n":
                calibrate(checkerboard)
                K, D, DIM = calibrate(checkerboard)
                undistort(colphoto, K, D, DIM)
                imgdefish = undistort(colphoto, K, D, DIM)
            elif param.lower() == "y":
                undistort(colphoto,K,D,DIM)
                imgdefish = undistort(colphoto, K, D, DIM)
        else:
            imgdefish = colphoto

        if whitebal.lower() == "y":
            white_balance(imgdefish)
            whitebalimg = white_balance(imgdefish)
        else:
            whitebalimg = imgdefish

        if deno.lower() == "y":
            denoise2(whitebalimg,int(luminance),int(photo_render),int(search_window),int(block_size))
            imgdenoise = denoise2(whitebalimg,int(luminance),int(photo_render),int(search_window),int(block_size))
        else:
            imgdenoise = whitebalimg

        if autoexp.lower() == "y":
            autoExposureCor(imgdenoise,percentage)
            imgcor = autoExposureCor(imgdenoise,percentage)
        else:
            imgcor = imgdenoise

        if res.lower() == "y":
            resize(imgcor, width, height)
        if pir.lower() == "y":
            scale(imgcor)



    elif inputfile == "1":
        print("Video path: ")
        video = input()
        breakvideo(video)
        x=breakvideo(video)
        print(x)

        print("Color enhancement?Y/N")
        corr = input()
        if corr.lower() == "y":
            print("Contrast Level(0-100)(Default = 0) :")
            corpr = input()
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
                K = input()
                K = np.array(K)
                K = np.array(str(K.tolist()))
                print("Lens distortion coefficients: ")
                D = input()
                D = np.array(D)
                D = np.array(str(D.tolist()))
                print("Photo dimensions: ")
                print("Width: ")
                width = input()
                print("Height: ")
                height = input()
                DIM = (int(width), int(height))
        print("White Balance?Y/N")
        whitebal = input()
        print("Denoising?Y/N")
        deno = input()
        if deno.lower() == "y":
            print("Luminance(0-100)(Default = 1) :")
            luminance = input()
            print("Photo render distance(0-100)(Default = 1) : ")
            photo_render = input()
            print("Search window(0-100)(Default = 1) : ")
            search_window = input()
            print("Block size(0-100)(Default = 1) : ")
            block_size = input()
        print("Autoexposure Correction?Y/N")
        autoexp = input()
        if autoexp.lower() == "y":
            print("Percentage(0-100)(Default = 0) : ")
            percentage = input()
        print("Resize the image?Y/N")
        res = input()
        if res.lower() == "y":
            print("Height: ")
            height = input()
            print("Width: ")
            width = input()

        for i in range (x):
            photo= "VIDEOSPART/" + video[0:-4]+str(i)+ ".jpg"

            if corr.lower() == "y":
                Enhance(photo, corpr)
                corphoto = Enhance(photo, corpr)
            else:
                corphoto = photo
            if color == "1":
                greyscale(corphoto)
                colphoto = greyscale(corphoto)
            elif color == "2":
                colphoto = corphoto

            if fisheye.lower() == "y":
                if param.lower() == "n":
                    if checkerboard[-4:] != ".jpg":
                        images = glob.glob(checkerboard)
                        for img in images:
                            debayer(img)
                    checkerboard = checkerboard[0:-4] + ".jpg"
                    calibrate(checkerboard)
                    K, D, DIM = calibrate(checkerboard)
                    undistort(colphoto, K, D, DIM)
                    imgdefish = colphoto[0:-4] + ".jpg"
                elif param.lower() == "y":
                    undistort(colphoto, K, D, DIM)
                    imgdefish = colphoto[0:-4] + ".jpg"
            else:
                imgdefish = colphoto

            if whitebal.lower() == "y":
                white_balance(imgdefish)
                whitebalimg = white_balance(imgdefish)
            else:
                whitebalimg = imgdefish

            if deno.lower() == "y":
                denoise2(whitebalimg, int(luminance), int(photo_render), int(search_window), int(block_size))
                imgdenoise = denoise2(whitebalimg, int(luminance), int(photo_render), int(search_window),
                                      int(block_size))
            else:
                imgdenoise = whitebalimg

            if autoexp.lower() == "y":
                autoExposureCor(imgdenoise, percentage)
                imgcor = autoExposureCor(imgdenoise, percentage)
            else:
                imgcor = imgdenoise

            if res.lower() == "y":
                resize(imgcor, width, height)


interface()
