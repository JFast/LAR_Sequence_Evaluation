import cv2
import numpy as np
import DisplayVideo as display
import Preprocessing as prepro

print("OpenCV version: " + cv2.__version__)

# define array for calibration images
frame_array = []

for sequence_index in range(1, 21):
    # define sequence folder path
    sequence_path = r"E:/PATH/TO/SEQUENCE/" + str(sequence_index) + ".FILE_TYPE"

    # load sequence
    sequence = display.loadVideo(sequence_path)

    # extract first frame from each sequence
    frame_read = False
    while (sequence.isOpened()) and (frame_read is False):
        ret, frame = sequence.read()
        if not ret:
            print("NO frame could be read!")
            # stop loop if no frame could be loaded (end of sequence)
            break
        else:
            # add first frame to array
            frame_array.append(frame)
            frame_read = True

# perform camera calibration with frames in frame_array
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html
# https://docs.opencv.org/master/d4/d94/tutorial_camera_calibration.html

# inter-circle distance in mm
distance_between_circles = 2.0

# prepare object point coordinates, like (0,0,0), (1,0,0), (2,0,0), ..., (6,5,0)
obj_p = np.zeros((4*11, 3), np.float32)

# for i in range(44):
#     obj_p[i] = (, 2*i, 0)

# circle_coords = []
# for i in range(11):
#     for j in range(4):
#         circle_coords[i, j] = [(2*j+divmod(i, 2)[1])*distance_between_circles, i*distance_between_circles]

obj_p[0] = (0, 0, 0)
obj_p[1] = (4.0, 0, 0)
obj_p[2] = (8.0, 0, 0)
obj_p[3] = (12.0, 0, 0)

obj_p[4] = (2.0, 2.0, 0)
obj_p[5] = (6.0, 2.0, 0)
obj_p[6] = (10.0, 2.0, 0)
obj_p[7] = (14.0, 2.0, 0)

obj_p[8] = (0, 4.0, 0)
obj_p[9] = (4.0, 4.0, 0)
obj_p[10] = (8.0, 4.0, 0)
obj_p[11] = (12.0, 4.0, 0)

obj_p[12] = (2.0, 6.0, 0)
obj_p[13] = (6.0, 6.0, 0)
obj_p[14] = (10.0, 6.0, 0)
obj_p[15] = (14.0, 6.0, 0)

obj_p[16] = (0, 8.0, 0)
obj_p[17] = (4.0, 8.0, 0)
obj_p[18] = (8.0, 8.0, 0)
obj_p[19] = (12.0, 8.0, 0)

obj_p[20] = (2.0, 10.0, 0)
obj_p[21] = (6.0, 10.0, 0)
obj_p[22] = (10.0, 10.0, 0)
obj_p[23] = (14.0, 10.0, 0)

obj_p[24] = (0, 12.0, 0)
obj_p[25] = (4.0, 12.0, 0)
obj_p[26] = (8.0, 12.0, 0)
obj_p[27] = (12.0, 12.0, 0)

obj_p[28] = (2.0, 14.0, 0)
obj_p[29] = (6.0, 14.0, 0)
obj_p[30] = (10.0, 14.0, 0)
obj_p[31] = (14.0, 14.0, 0)

obj_p[32] = (0, 16.0, 0)
obj_p[33] = (4.0, 16.0, 0)
obj_p[34] = (8.0, 16.0, 0)
obj_p[35] = (12.0, 16.0, 0)

obj_p[36] = (2.0, 18.0, 0)
obj_p[37] = (6.0, 18.0, 0)
obj_p[38] = (10.0, 18.0, 0)
obj_p[39] = (14.0, 18.0, 0)

obj_p[40] = (0, 20.0, 0)
obj_p[41] = (4.0, 20.0, 0)
obj_p[42] = (8.0, 20.0, 0)
obj_p[43] = (12.0, 20.0, 0)

# arrays to store object points and image points from all frames
object_points = []  # 3D points in world coordinates
image_points = []   # 2D points in calibration pattern coordinates

print("len(frame_array): " + str(len(frame_array)))

# tune blob detector
params = cv2.SimpleBlobDetector_Params()
params.minThreshold = 1
params.maxThreshold = 255

params.thresholdStep = 1
params.minDistBetweenBlobs = 3

detector = cv2.SimpleBlobDetector_create(params)

saving_path = r"E:/OUTPUT/PATH/"

# array for frames with found circles
frame_array_circles = []

# for all frames in frame_array
for i in range(len(frame_array)):
    img = frame_array[i]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = cv2.equalizeHist(gray)

    clahe = cv2.createCLAHE(1.0)
    gray = clahe.apply(gray)

    # find circles in asymmetrical circle grid
    ret, circles = cv2.findCirclesGrid(gray, (4, 11), cv2.CALIB_CB_ASYMMETRIC_GRID, detector, None)

    detections = detector.detect(gray)

    blank = np.zeros((1, 1))

    blobs = cv2.drawKeypoints(gray, detections, blank, (0, 0, 255))

    cv2.imshow('blobs', blobs)
    cv2.waitKey(0)

    # if found, add object points, image points
    if ret:
        print("Pattern found!")
        object_points.append(obj_p)
        image_points.append(circles)

        # draw and display circles
        img_circles_temp = img.copy()
        img_circles_temp = cv2.drawChessboardCorners(img_circles_temp, (4, 11), circles, ret)
        frame_array_circles.append(img_circles_temp)
        cv2.imshow('frame', img_circles_temp)
        cv2.waitKey(0)

        # save raw frame with detected pattern
        cv2.imwrite(saving_path + "Raw_Imgs_Pattern/" + "raw_image_pattern_" + str(i+1) + ".png", img_circles_temp)

cv2.destroyAllWindows()

# perform calibration
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html

# calibration termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)

calib_error, cam_mtx, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, gray.shape[::-1],
                                                                       None, None, None, None, None, criteria)
print(calib_error)
print(cam_mtx)
print(dist_coeffs)

# save calibration results to XML file

s = cv2.FileStorage(saving_path + "Calibration.xml", cv2.FileStorage_WRITE)

s.write('CAMERA_MATRIX', cam_mtx)
s.write('DISTORTION_COEFFICIENTS', dist_coeffs)
s.write('REPROJECTION_ERROR', calib_error)
s.release()

print("XML file successfully written!")

# save raw/undistorted/difference frames

index = 1
for frame in frame_array:
    # save raw frame
    cv2.imwrite(saving_path + "Raw_Imgs/" + "raw_image_" + str(index) + ".png", frame)

    # undistort frame
    undist = cv2.undistort(frame, cam_mtx, dist_coeffs, None, None)
    # save undistorted frame
    cv2.imwrite(saving_path + "Undist_Imgs/" + "undist_image_" + str(index) + ".png", undist)

    # compute difference image
    diff_img_temp = cv2.subtract(frame, undist)
    diff_img = cv2.bitwise_not(diff_img_temp)
    # convert into grayscale
    diff_img = cv2.cvtColor(diff_img, cv2.COLOR_BGR2GRAY)
    # apply contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0)
    diff_img = clahe.apply(diff_img)
    # save difference image
    cv2.imwrite(saving_path + "Difference_Imgs/" +"difference_image_" + str(index) + ".png", diff_img)

    index += 1

# read test grid image
grid_path = r"E:/GRID/PATH/Calibration_Test_Grid.png"
grid_img = cv2.imread(grid_path)

# undistort frame
undist = cv2.undistort(grid_img, cam_mtx, dist_coeffs, None, None)
# save undistorted frame
cv2.imwrite(saving_path + "Undist_Imgs/" + "undist_GRID.png", undist)

# compute difference image
diff_img_temp = cv2.subtract(grid_img, undist)
diff_img = cv2.bitwise_not(diff_img_temp)
# convert into grayscale
diff_img = cv2.cvtColor(diff_img, cv2.COLOR_BGR2GRAY)
# apply contrast enhancement
clahe = cv2.createCLAHE(clipLimit=2.0)
diff_img = clahe.apply(diff_img)
# save difference image
cv2.imwrite(saving_path + "Difference_Imgs/" + "difference_image_GRID.png", diff_img)

# alpha = 0.5
# beta = 1.0 - alpha
# blended = cv2.addWeighted(frame_array[5], alpha, diff_img_temp, beta, 0.0)

# cv2.imshow('frame 6 (blended)', blended)

cv2.waitKey(0)

cv2.destroyAllWindows()
