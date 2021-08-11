import cv2
import numpy as np
import math

def getConvexHull(contour):
    """
    - computes convex hull of (glottis) contour
    :param contour: contour points
    :return: points of convex hull
    """
    hull = cv2.convexHull(contour)
    return hull


def getExtremePointsContour(contour):
    """
    - computes extreme points on contour (left, right, top, bottom)
    :param contour: contour points
    :return: extreme points on contour (left, right, top, bottom)
    """
    c = contour
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])
    return extLeft, extRight, extTop, extBot


def getReferenceHeight(extTop, extBot, reference):
    """
    -computes index of row situated at desired height between minimum and maximum point of contour
    :param extTop: extreme point of contour (top)
    :param extBot: extreme point of contour (bottom)
    :param reference: desired height between extreme points (e.g., 1/3)
    :return: index of row at desired height
    """
    # reference = 0.0 --> row_index_at_reference_height = int(extBot[1])
    # reference = 1.0 --> row_index_at_reference_height = int(extBot[1] - abs(extTop[1] - extBot[1])) = int(extTop[1])
    row_index_at_reference_height = abs(extTop[1] - extBot[1]) * reference
    row_index_at_reference_height = int(extBot[1] - row_index_at_reference_height)
    return row_index_at_reference_height


def getPointsOnHull(shape_x, shape_y, hull, referenceHeight):
    """
    - determines (2) points on convex hull situated at desired height (row index) in frame
    :param shape_x: frame size in horizontal direction
    :param shape_y: frame size in vertical direction
    :param hull: contour points of convex hull
    :param referenceHeight: row index at which points are to be identified
    :return: left and right point on convex hull situated at desired height
    """
    mask = np.ones((shape_y, shape_x)).astype('uint8')
    mask = cv2.drawContours(mask, [hull], 0, 255, 1)
    maskRow = np.ones((shape_y, shape_x)).astype('uint8')
    maskRow[referenceHeight, :] = 255
    maskRow = cv2.bitwise_and(maskRow, mask)
    index = np.where(maskRow == 255)
    leftPoint = [index[1][0], referenceHeight]
    rightPoint = [index[1][-1], referenceHeight]
    return leftPoint, rightPoint


def getPointOnGlottisContour(shape_x, shape_y, glottisContour, hullPoint):
    """
    - computes closest point in glottis w. r. t. given point (hullPoint) on convex hull in frame
    :param shape_x: frame size in horizontal direction
    :param shape_y: frame size in vertical direction
    :param glottisContour: contour points of convex hull
    :param hullPoint: point on convex hull
    :return: closest point in glottis
    """
    mask = np.ones((shape_y, shape_x)).astype('uint8')
    mask = cv2.drawContours(mask, [glottisContour], 0, 255, 1)
    maskRow = np.ones((shape_y, shape_x)).astype('uint8')
    maskRow[hullPoint[1], :] = 255
    maskRow = cv2.bitwise_and(maskRow, mask)
    # obtain indices where 'maskRow' has value 255
    index = np.where(maskRow == 255)
    # if no pixel locations found
    if len(index[1]) == 0:
        return []
    else:
        # initialize variable 'max' with frame size in horizontal direction
        max = shape_x
        # initialize variable 'x' with horizontal coordinate of given point on convex hull of segmented glottis
        x = hullPoint[0]
        # for all found pixels on glottis contour that are on same row as given point on convex hull
        for i in range(0, len(index[1])):
            # calculate distance between horizontal coordinates of given point on convex hull and of point on contour
            if max > abs(hullPoint[0] - index[1][i]):
                # update distance between points
                max = abs(hullPoint[0] - index[1][i])
                # update variable 'x' with horizontal coordinate of point on contour
                x = index[1][i]
        return [x, hullPoint[1]]


def getBottomPointAngle(hull, referenceHeight, leftPoint, left):
    """
    - computes second edge-defining point on vocal fold edge
    :param hull: contour points of convex hull
    :param referenceHeight: reference height, point of interest must lie below this value (larger y value)
    :param leftPoint: first point on vocal fold edge
    :param left: flag for computation of point on edge of left (True) or right (False) vocal fold
    :return: second point on vocal fold edge
    """
    distance = 255
    next = None
    for point in hull:
        if point[0][1] > referenceHeight:
            distancePoints = math.sqrt(pow(abs(point[0][1] - leftPoint[1]), 2) + pow(abs(point[0][0] - leftPoint[0]), 2))
            if distancePoints < distance:
                distance = distancePoints
                next = point[0]
                if next[0] == leftPoint[0]:
                    if left:
                        next[0] = leftPoint[0] + 1
                    else:
                        next[0] = leftPoint[0] - 1
    return next


def getAngleBetweenPoints(point1, point2, point3):
    """
    - computes angle between two lines defined by three points (line 1: point 1 to point 2, line 2: point 2 to point 3)
    :param point1: point on line 1
    :param point2: point on both lines
    :param point3: point on line 2
    :return: angle between lines defined by points point1, point2 and point3 (in degrees)
    """
    # compute lengths of line segments between points
    a = math.sqrt(pow(abs(point1[0] - point2[0]), 2) + pow(abs(point1[1] - point2[1]), 2))
    b = math.sqrt(pow(abs(point2[0] - point3[0]), 2) + pow(abs(point2[1] - point3[1]), 2))
    c = math.sqrt(pow(abs(point1[0] - point3[0]), 2) + pow(abs(point1[1] - point3[1]), 2))
    # apply law of cosines to obtain angle between line segments
    angle = (math.acos((pow(a, 2) + pow(b, 2) - pow(c, 2))/(2 * a * b))) * (180/math.pi)
    return angle


def getPoint(points):
    """
    - determines anterior point on vocal fold edge, defining location of anterior commissure for glottal angle
    - angle calculated according to available number of points
    - anterior commissure/anterior point on vocal fold edge located at sharp bend in contour
    - sharp bend defined as angle of at least 160 degrees
    :param points: candidate points for anterior point on vocal fold edge, defining location of anterior commissure
    :return: anterior point on vocal fold edge, defining location of anterior commissure
    """
    if len(points) == 1:
        return points[0]
    if len(points) == 2:
        return points[1]
    if len(points) == 3:
        angle = getAngleBetweenPoints(points[0], points[1], points[2])
        if angle < 160:
            return points[1]
        else:
            return points[2]
    if len(points) == 4:
        angle = getAngleBetweenPoints(points[0], points[1], points[2])
        if angle < 160:
            return points[1]
        angle = getAngleBetweenPoints(points[1], points[2], points[3])
        if angle < 160:
            return points[2]
        else:
            return points[3]
    if len(points) == 5:
        angle = getAngleBetweenPoints(points[0], points[1], points[2])
        if angle < 160:
            return points[1]
        angle = getAngleBetweenPoints(points[1], points[2], points[3])
        if angle < 160:
            return points[2]
        angle = getAngleBetweenPoints(points[2], points[3], points[4])
        if angle < 160:
            return points[3]
        else:
            return points[4]
    else:
        for i in range(0, len(points) - 5):
            angle = getAngleBetweenPoints(points[i], points[i + 2], points[i + 4])
            if angle < 160:
                angle = getAngleBetweenPoints(points[i], points[i + 1], points[i + 2])
                return points[i+1]
        return points[-1]


def getDistancePoints(frame, glottisContour, referenceHeight):
    """
    - identifies points on vocal fold edges required for distance measurement (see Lohscheller et al.)
    :param frame: input frame
    :param glottisContour: glottis contour
    :param referenceHeight: reference height (given w. r. t. total glottis length) at which points are to be identified
    :return: left and right point on vocal fold edges
    """
    hull = getConvexHull(glottisContour)
    extLeft, extRight, extTop, extBot = getExtremePointsContour(glottisContour)
    reference_height_top = getReferenceHeight(extTop, extBot, referenceHeight)
    left_point_hull, right_point_hull = getPointsOnHull(frame.shape[0], frame.shape[1], hull, reference_height_top)
    left_point_glottis = getPointOnGlottisContour(frame.shape[0], frame.shape[1], glottisContour, left_point_hull)
    right_point_glottis = getPointOnGlottisContour(frame.shape[0], frame.shape[1], glottisContour, right_point_hull)
    return left_point_glottis, right_point_glottis


def getDistancePointsConstantHeight(frame, glottisContour, distanceHeight):
    """
    - identifies points on vocal fold edges required for distance measurement
    - in contrast to Lohscheller et al., distance defined at constant height in image (no glottis translation expected)
    :param frame: input frame
    :param glottisContour: glottis contour
    :param distanceHeight: vertical coordinate of vocal fold edge distance (constant over sequence)
    :return: left and right point on vocal fold edges
    """
    hull = getConvexHull(glottisContour)
    left_point_hull, right_point_hull = getPointsOnHull(frame.shape[0], frame.shape[1], hull, distanceHeight)
    left_point_glottis = getPointOnGlottisContour(frame.shape[0], frame.shape[1], glottisContour, left_point_hull)
    right_point_glottis = getPointOnGlottisContour(frame.shape[0], frame.shape[1], glottisContour, right_point_hull)
    return left_point_glottis, right_point_glottis


def getGlottalPoints(frame, glottisContour):
    """
    - identifies points on vocal fold edges required for glottal angle computation:
    anterior and posterior points (at 40 percent of total glottis height, seen from vertex) on vocal fold edges
    :param frame: input frame
    :param glottisContour: glottis contour points
    :return: points on vocal fold edges (left (posterior), right (posterior)
    (each at 40 percent of total glottis height, seen from vertex), left (anterior), right (anterior))
    """
    # calculation of convex hull of glottis contour
    hull = getConvexHull(glottisContour)
    # localization of extremal points on glottis contour
    extLeft, extRight, extTop, extBot = getExtremePointsContour(glottisContour)
    print(extTop, extBot)

    # localization of vertical coordinate of angle-defining points on vocal fold edges
    reference_height_top = getReferenceHeight(extTop, extBot, 0.4)
    # definition of threshold of vertical coordinate of vertex point of glottal angle
    reference_height_bottom = getReferenceHeight(extTop, extBot, 0.2)
    # localization of points on convex hull of glottis contour at height of angle-defining points on vocal fold edges
    left_point_hull, right_point_hull = getPointsOnHull(frame.shape[0], frame.shape[1], hull, reference_height_top)
    # localization of points on segmented glottis contour at height of angle-defining points on vocal fold edges
    left_point_glottis = getPointOnGlottisContour(frame.shape[0], frame.shape[1], glottisContour, left_point_hull)
    right_point_glottis = getPointOnGlottisContour(frame.shape[0], frame.shape[1], glottisContour, right_point_hull)

    # visualization of convex hull
    # frame_folds = frame.copy()
    # frame_folds = cv2.drawContours(frame_folds, [hull], 0, [0, 255, 0], 1)

    # localization of central point between left and right angle-defining point on vocal fold edges
    # (at height 'reference_height_top')
    mid_point = [int(left_point_glottis[0] + abs((left_point_glottis[0] - right_point_glottis[0])/2.0)),
                 int(left_point_glottis[1])]
    # initialize empty list
    left_up = list()
    left_up.append(left_point_glottis)
    for point_ in hull:
        point = point_[0]
        # if point left of central point (at height 'reference_height_top')
        if point[0] < mid_point[0]:
            # if point above central point (at height 'reference_height_top')
            if point[1] < mid_point[1]:
                left_up.append([point[0], point[1]])
    # return list 'left_up', sorted by descending value of vertical coordinate of points
    left_up = sorted(left_up, reverse=True, key=lambda x: x[1])
    # visualization of points in list 'left_up'
    # for point in left_up:
    #     frame_folds = cv2.circle(frame_folds, (int(point[0]), int(point[1])), 2, [0, 255, 0], -1)
    point_left_up = getPoint(left_up)
    # if point above point on vocal fold edge at threshold coordinate value 'reference_height_top'
    if point_left_up[1] < left_point_glottis[1]:
        # set 'point_left_up' to point on left vocal fold edge at height 'reference_height_top'
        point_left_up = left_point_glottis
    # frame_folds = cv2.circle(frame_folds, (int(point_left_up[0]), int(point_left_up[1])), 2, [0, 0, 255], -1)

    # initialize empty list
    left_bottom = list()
    left_bottom.append(left_point_glottis)
    for point_ in hull:
        point = point_[0]
        # if point left of extreme point at bottom of glottis contour
        if point[0] <= extBot[0]:
            # if point below central point
            if point[1] > mid_point[1]:
                left_bottom.append([point[0], point[1]])
    # return list 'left_bottom', sorted by ascending value of vertical coordinate of points
    left_bottom = sorted(left_bottom, key=lambda x: x[1])
    # for point in left_bottom:
    #      frame_folds = cv2.circle(frame_folds, (int(point[0]), int(point[1])), 2, [255, 255, 0], -1)
    point_left_bottom = getPoint(left_bottom)
    # frame_folds = cv2.circle(frame_folds, (int(point_left_bottom[0]), int(point_left_bottom[1])), 2, [0, 0, 255], -1)

    # initialize empty list
    right_up = list()
    right_up.append(right_point_glottis)
    for point_ in hull:
        point = point_[0]
        # if point right of central point
        if point[0] > mid_point[0]:
            # if point above central point
            if point[1] < mid_point[1]:
                right_up.append([point[0], point[1]])
    # return list 'right_up', sorted by descending value of vertical coordinate of points
    right_up = sorted(right_up, reverse=True, key=lambda x: x[1])
    # for point in right_up:
    #     frame_folds = cv2.circle(frame_folds, (int(point[0]), int(point[1])), 2, [0, 255, 255], -1)
    point_right_up = getPoint(right_up)
    # frame_folds = cv2.circle(frame_folds, (int(point_right_up[0]), int(point_right_up[1])), 2, [0, 0, 255], -1)
    # if point above point on vocal fold edge at threshold coordinate value 'reference_height_top'
    if point_right_up[1] < right_point_glottis[1]:
        point_right_up = right_point_glottis

    right_bottom = list()
    right_bottom.append(right_point_glottis)
    for point_ in hull:
        point = point_[0]
        # if point right of extreme point at bottom of glottis contour
        if point[0] >= extBot[0]:
            # if point below central point
            if point[1] > mid_point[1]:
                right_bottom.append([point[0], point[1]])
    right_bottom = sorted(right_bottom, key=lambda x: x[1])
    # for point in right_bottom:
    #     frame_folds = cv2.circle(frame_folds, (int(point[0]), int(point[1])), 2, [255, 0, 0], -1)
    point_right_bottom = getPoint(right_bottom)
    # frame_folds = cv2.circle(frame_folds, (int(point_right_bottom[0]), int(point_right_bottom[1])), 2,
    # [0, 0, 255], -1)

    if point_left_up == point_left_bottom:
        # locate closest point to 'left_point_glottis' below threshold 'reference_height_bottom'
        point_left_bottom = getBottomPointAngle(hull, reference_height_bottom, left_point_glottis, True)
    if point_right_up == point_right_bottom:
        # locate closest point to 'right_point_glottis' below threshold 'reference_height_bottom'
        point_right_bottom = getBottomPointAngle(hull, reference_height_bottom, right_point_glottis, False)

    print(point_left_up, point_right_up)
    return point_left_up, point_right_up, point_left_bottom, point_right_bottom


def get_slope(point_1, point_2):
    """
    - computes slope of a line defined by two points on line
    :param point_1: point 1 on line
    :param point_2: point 2 on line
    :return: slope of line (or 0 in case of strictly vertical line)
    """
    # identify x and y coordinates of points
    x_1 = point_1[0]
    y_1 = point_1[1]
    x_2 = point_2[0]
    y_2 = point_2[1]
    # handle singular case of vertical line
    if (x_2 - x_1) == 0:
        return 0
    else:
        return (y_2 - y_1) / (x_2 - x_1)


def get_y_intercept(point_1, slope):
    """
    - computes y-axis intercept of line defined by slope and point on line
    :param point_1: point on line
    :param slope: slope of line
    :return: y-axis intercept of line
    """
    # identify x and y coordinates of point
    x = point_1[0]
    y = point_1[1]
    return y - slope * x


def straight_line(m, b, x):
    """
    - returns y coordinate of point on line defined by x coordinate and slope and y-axis intercept of line
    :param m: slope of line
    :param b: y-axis intercept of line
    :param x: x coordinate
    :return: y coordinate of point on given line
    """
    return m*x+b


def get_coordinate_form_straight(slope, intercept):
    """
    - transforms slope-intercept form of line into parametric form
    :param slope: slope of line
    :param intercept: y-axis intercept of line
    :return: support and direction vectors of line in parametric form
    """
    # calculate x and y coordinates of support vector
    point_1_x = 0
    point_1_y = straight_line(slope, intercept, point_1_x)
    # calculate x and y coordinates of second point to obtain direction vector
    point_2_x = 255
    point_2_y = straight_line(slope, intercept, point_2_x)
    # support vector
    point_of_reference = (point_1_x, point_1_y)
    # direction vector
    direction_vector = (point_2_x - point_1_x, point_2_y - point_1_y)
    return point_of_reference, direction_vector


def interception_two_lines(slope_1, intercept_1, slope_2, intercept_2):
    """
    - computes point of intersection of two lines
    :param slope_1: slope of line 1
    :param intercept_1: y-axis intercept of line 1
    :param slope_2: slope of line 2
    :param intercept_2: y-axis intercept of line 2
    :return: tupel of x and y coordinates of point of intersection
    """
    # support and direction vectors of first line
    point_of_reference_1, direction_vector_1 = get_coordinate_form_straight(slope_1, intercept_1)
    # support and direction vectors of second line
    point_of_reference_2, direction_vector_2 = get_coordinate_form_straight(slope_2, intercept_2)
    # construct system of linear equations
    results = [point_of_reference_2[0]-point_of_reference_1[0], point_of_reference_2[1]-point_of_reference_1[1]]
    direction_vector_1 = [direction_vector_1[0], direction_vector_1[1]]
    direction_vector_2 = [-direction_vector_2[0], -direction_vector_2[1]]
    # solve system with NumPy
    result = np.linalg.solve(np.array([direction_vector_1, direction_vector_2]), np.array(results))
    # inject reesult and compute x and y coordinates of point of intersection
    x = point_of_reference_1[0]+result[1]*direction_vector_1[0]
    y = point_of_reference_1[1]+result[1]*direction_vector_1[1]
    return (x, y)


def getVertexPoint(left_point_glottis, right_point_glottis, left_point_bottom, right_point_bottom):
    """
    - computes vertex for the identification of the glottal angle
    :param left_point_glottis: left point on vocal fold edge (posterior)
    :param right_point_glottis: right point on vocal fold edge (posterior)
    :param left_point_bottom: left point on vocal fold edge (anterior)
    :param right_point_bottom: right point on vocal fold edge (anterior)
    :return: vertex of glottal angle
    """
    if left_point_glottis[0] == left_point_bottom[0]:
        left_point_bottom = [left_point_bottom[0] + 1, left_point_bottom[1]]
    if right_point_glottis[0] == right_point_bottom[0]:
        right_point_bottom = [right_point_bottom[0] - 1, right_point_bottom[1]]
    slope_left = get_slope(left_point_glottis, left_point_bottom)
    slope_right = get_slope(right_point_glottis, right_point_bottom)
    intercept_left = get_y_intercept(left_point_glottis, slope_left)
    intercept_right = get_y_intercept(right_point_glottis, slope_right)
    vertex_point = interception_two_lines(slope_left, intercept_left, slope_right, intercept_right)
    vertex_point = (int(vertex_point[0]), int(vertex_point[1]))
    return vertex_point


def angle_straight_lines(left_point_glottis, right_point_glottis, vertex_point):
    """
    - computes angle between two lines defined by two points and vertex point
    :param left_point_glottis: point on left vocal fold edge
    :param right_point_glottis: point on right vocal fold edge
    :param vertex_point: vertex of glottal angle
    :return: angle between lines 1 and 2 in degree
    """
    slope_1 = get_slope(left_point_glottis, vertex_point)
    slope_2 = get_slope(right_point_glottis, vertex_point)
    if slope_1 == 0:
        angle = math.atan2(abs((0 - slope_2)), abs(1 + 0 * slope_2))
        angle = (angle / math.pi) * 180
        angle = (-1) * (angle - 90)
    elif slope_2 == 0:
        angle = math.atan2(abs((slope_1 - 0)), abs(1 + slope_1 * 0))
        angle = (angle / math.pi) * 180
        angle = (-1) * (angle - 90)
    else:
        angle = math.atan2(abs((slope_1 - slope_2)), abs(1 + slope_1 * slope_2))
        angle = (angle / math.pi) * 180
    # if both lines exactly vertical or both lines exactly horizontal (parallel lines)
    if slope_1 == 0 and slope_2 == 0:
        angle = 0
    # if lines exactly perpendicular
    if (slope_1 * slope_2) == -1.0:
        angle = 90.0
    return angle


def angle_straight_lines_iterative(slope_1, slope_2):
    """
    - computes angle between two lines defined by slopes
    :param slope_1: slope of line 1
    :param slope_2: slope of line 2
    :return: angle between lines 1 and 2
    """
    if slope_1 == 0:
        angle = math.atan2(abs((0 - slope_2)), abs(1 + 0 * slope_2))
        angle = (angle / math.pi) * 180
        angle = (-1) * (angle - 90)
    elif slope_2 == 0:
        angle = math.atan2(abs((slope_1 - 0)), abs(1 + slope_1 * 0))
        angle = (angle / math.pi) * 180
        angle = (-1) * (angle - 90)
    else:
        angle = math.atan2(abs((slope_1 - slope_2)), abs(1 + slope_1 * slope_2))
        angle = (angle / math.pi) * 180
    # if both lines vertical or both lines horizontal
    if slope_1 == 0 and slope_2 == 0:
        angle = 0
    return angle
