import cv2
import numpy as np

def convertToVChannel(frame):
    """
    - yields V channel of input frame in HSV color space
    :param frame: input frame
    :return: V channel of transformed frame
    """
    # return V channel of frame in HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    return v


def enhancementContrast(frame, factor):
    """
    - applies contrast enhancement by multiplication by factor
    - followed by clamping to range [0, 255]
    :param frame: frame to be contrast enhanced
    :param factor: contrast enhancement factor
    :return: contrast enhanced frame
    """
    # multiply pixel intensities by desired factor
    frame = np.array(frame * factor)
    # clamping
    frame[frame > 255] = 255
    # data type conversion
    frame = np.array(frame).astype('uint8')
    return frame


def applyCLAHE(frame, clipLimitIn):
    """
    - applies CLAHE contrast enhancement
    :param frame: frame to be CLAHE contrast enhanced
    :param clipLimitIn: clipping limit
    :return: CLAHE contrast enhanced frame
    """
    clahe = cv2.createCLAHE(clipLimit=clipLimitIn)
    frame = clahe.apply(frame)
    # clamping
    frame[frame > 255] = 255
    # data type conversion
    frame = np.array(frame).astype('uint8')
    return frame

def equalizeHistogram(frame):
    """
    - applies simple histogram equalization
    :param frame: frame to be processed
    :return: processed frame
    """
    frame_equalized = cv2.equalizeHist(frame)
    # clamping
    frame_equalized[frame_equalized > 255] = 255
    # data type conversion
    frame_equalized = np.array(frame_equalized).astype('uint8')
    return frame_equalized


def convertImageToArray(frame):
    """
    - transforms frame to numpy array
    :param image: frame
    :return: frame as array
    """
    return np.array(frame)


def bilateralFiltering(frame, filterSize, color, space):
    """
    - applies edge-preserving bilateral filter
    :param frame: input frame
    :param filterSize: filter size
    :param color: standard deviation of range filter
    :param space: standard deviation of domain filter
    :return: filtered frame
    """
    return cv2.bilateralFilter(frame, filterSize, color, space)


def convertToGray(frame):
    """
    - converts BGR color frame to grayscale frame
    :param frame: BGR color frame
    :return: grayscale frame
    """
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def getMean(shape_x, shape_y, glottisContour, frame):
    """
    - computes average value of intensity values at glottal area for a given frame
    :param shape_x: frame size in horizontal direction
    :param shape_y: frame size in vertical direction
    :param glottisContour: contour points of glottis
    :param frame: input frame
    :return: average value of intensity values at glottal area
    """
    mask = np.zeros((shape_y, shape_x)).astype('uint8')
    mask = cv2.drawContours(mask, [glottisContour], 0, 255, -1)
    mean = np.mean(frame[mask == 255])
    return mean


def getPreForBackgroundSubtraction(frame):
    """
    - performs frame pre-processing for droplet impact detection (removes white rows at top of frames)
    - executes frame cropping and average value filtering (noise reduction)
    :param frame: input frame
    :return: cropped and filtered frame
    """
    # remove artifacts from upper boundary of frames
    frame = frame[5:, :, :]
    # apply Gaussian blurring step
    frame_mean = cv2.GaussianBlur(frame, (7, 7), 7)
    return frame_mean
