import cv2

def loadVideo(path):
    """
    - loads video file using OpenCV
    :param path: path to video file
    :return: OpenCV video object
    """
    video = cv2.VideoCapture(path)
    return video


def displayFrame(name, frame, wait):
    """
    - shows image using OpenCV
    :param name: window name
    :param frame: image to be shown
    :param wait: showing time (0 -> until key is pressed)
    :return: void
    """
    cv2.imshow(name, frame)
    cv2.waitKey(wait)


def destroyWindows():
    """
    - closes all OpenCV windows
    :return: void
    """
    cv2.destroyAllWindows()


def displayReferencePoint(frame, x, y, color, saving_path):
    """
    - shows and saves frame with computed reference point as circular overlay
    :param frame: input frame
    :param x: x coordinate of reference point
    :param y: y coordinate of reference point
    :param color: circle color
    :return: void
    """
    frame = cv2.circle(frame, (x, y), 2, color, -1)
    cv2.imwrite(saving_path, frame)
    frame_large = cv2.resize(frame, (int(2.0 * frame.shape[1]), int(2.0 * frame.shape[0])),
                             interpolation=cv2.INTER_LINEAR)
    displayFrame("Reference Point", frame_large, 0)


def displayWatershedSegmentation(frame, watershed, color):
    """
    - shows segmentation result (area borders) of watershed method
    :param frame: image
    :param watershed: marker image (border with value -1)
    :param color: color
    :return: void
    """
    frame_watershed = frame.copy()
    frame_watershed[watershed == -1] = color
    frame_watershed_large = cv2.resize(frame, (int(2.0 * frame_watershed.shape[1]), int(2.0 * frame_watershed.shape[0])),
                             interpolation=cv2.INTER_LINEAR)
    displayFrame("Watershed Segmentation Result", frame_watershed_large, 0)


def drawGlottisContour(frame, contour, color):
    """
    - shows computed glottis contour
    :param frame: input frame
    :param contour: contour points
    :param color: color
    :return: input frame
    """
    frame = frame.copy()
    cv2.drawContours(frame, [contour], 0, color, 1)
    # frame_large = cv2.resize(frame, (int(2.0 * frame.shape[1]), int(2.0 * frame.shape[0])), interpolation=cv2.INTER_LINEAR)
    displayFrame("Glottis Contour", frame, 0)
    return frame
