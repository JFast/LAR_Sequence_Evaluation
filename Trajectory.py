import cv2
import math
import numpy as np

def getMaxCandidate(candidates):
    """
    - returns the largest object found by blob detection
    :param candidates: candidates (found by blob detection)
    :return: largest object found by blob detection
    """
    candidate_to_add = candidates[0]
    min = 0
    for candidate in candidates:
        if candidate.size > min:
            min = candidate.size
            candidate_to_add = candidate
    return candidate_to_add


def getDistanceBetweenDroplets(first_droplet, second_droplet):
    """
    - calculates Euclidean distance between two points
    :param first_droplet: point 1
    :param second_droplet: point 2
    :return: Euclidean distance between points 1 and 2
    """
    distance = math.sqrt(pow(abs(first_droplet[0] - second_droplet[0]), 2) +
                         pow(abs(first_droplet[1] - second_droplet[1]), 2))
    return distance


def getCentroidOfContour(contour):
    """
    - determines centroid of object contour
    :param contour: object contour
    :return: x and y coordinates of centroid
    """
    M = cv2.moments(contour)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    return cx, cy

def getDistanceInNextFrame(droplets_list, frame_number):
    """
    - calculates estimation of distance of droplet position in next frame to current position
    :param droplets_list: list with objects (droplets) and corresponding frame indices
    :param frame_number: current frame index
    :return: estimation of distance of droplet position in next frame to current position
    """
    last_droplet = droplets_list[-1][1]
    last_droplet_second = droplets_list[-2][1]
    last_frame = droplets_list[-1][0]
    last_frame_second = droplets_list[-2][0]
    velocity = math.sqrt(pow(abs(last_droplet_second[0] - last_droplet[0]), 2) + pow(
        abs(last_droplet_second[1] - last_droplet_second[1]), 2)) / (last_frame - last_frame_second)
    if velocity == 0:
        velocity = 1
    distance = velocity * (frame_number - last_frame)
    return distance


def getBestPointInNextFrame(droplets_list, frame_number):
    """
    - estimates droplet coordinates in next frame as function of previously detected droplet positions
    :param droplets_list: list with objects (droplets) and corresponding frame indices
    :param frame_number: current frame index
    :return: estimated droplet coordinates in next frame
    """
    last_droplet = droplets_list[-1][1]
    last_droplet_second = droplets_list[-2][1]
    last_frame = droplets_list[-1][0]
    last_frame_second = droplets_list[-2][0]
    best_point = [last_droplet[0] + ((last_droplet[0] - last_droplet_second[0]) / (last_frame - last_frame_second)) *
                  (frame_number - last_frame),
        last_droplet[1] + ((last_droplet[1] - last_droplet_second[1]) / (last_frame - last_frame_second)) *
                  (frame_number - last_frame)]
    return best_point


def getFitDropletList(droplets_list):
    """
    - computes linear fit of detected droplet positions
    - 'polyfit' function cannot be used for vertical droplet trajectories
    - hence, manual line parameter determination using last and first droplet in list is applied in this case
    - fit with lowest residual is retained
    :param droplets_list: list with object (droplet) positions
    :return: slope and value at x=0 of fit line
    """
    x_list = list()
    y_list = list()
    for droplet in droplets_list:
        x_list.append(droplet[1][0])
        y_list.append(droplet[1][1])
    fit = np.polyfit(x_list, y_list, 1)
    error_fit = 0
    for i in range(0, len(x_list)):
        error_point = abs(y_list[i] - (fit[0] * x_list[i] + fit[1]))
        error_fit = error_fit + error_point
    slope = get_slope(droplets_list[0][1], droplets_list[-1][1])
    intercept = get_y_intercept(droplets_list[0][1], slope)
    fit_slope = [slope, intercept]
    if not fit_slope[0] == 0:
        error_slope = 0
        for i in range(0, len(x_list)):
            error_point = abs(x_list[i] - ((y_list[i] - fit_slope[1])/fit_slope[0]))
            error_slope = error_slope + error_point
        if error_fit <= error_slope:
            return fit
        else:
            return fit_slope
    else:
        return fit

    # # calculate fit line (geometric fitting)
    # [vx, vy, x, y] = cv2.fitLine(droplets_list[1], cv2.DIST_L2, 0, 0.01, 0.01)
    # # calculate slope of fit line
    # slope = vy/vx
    # # calculate y axis intercept
    # intercept = y - vy/vx * x
    # fit = [slope, intercept]
    # return fit


def getAcceptanceAngle(droplets_list, frame_number):
    """
    - calculates "acceptance angle" of conical droplet search space
    - angle is function of number of frames elapsed since last valid droplet detection
    - droplets_list[-1][0]: index of frame with last valid droplet detection
    :param droplets_list: list of droplet objects and corresponding frame indices
    :param frame_number: current frame index
    :return: "acceptance angle" of conical droplet search space
    """
    acceptance_angle = 30
    if abs(droplets_list[-1][0] - frame_number) > 10:
        acceptance_angle = 28
    if abs(droplets_list[-1][0] - frame_number) > 20:
        acceptance_angle = 26
    if abs(droplets_list[-1][0] - frame_number) > 30:
        acceptance_angle = 24
    if abs(droplets_list[-1][0] - frame_number) > 40:
        acceptance_angle = 22
    return acceptance_angle


def getCylinderMask(frame, fit, distance, to_add, acceptance_angle, last_droplet, droplets_list):
    """
    - determines cylindrical/conical droplet search space
    :param frame: frame for frame size determination
    :param fit: slope and y-axis intercept of droplet trajectory fit line
    :param distance: distance to previous object
    :param to_add: distance to be added as search space
    :param acceptance_angle: angle for the determination of the cone edges
    :param last_droplet: coordinates of last detected object/droplet
    :param droplets_list: list with all detected objects/droplets
    :return: mask with search space as bright object
    """
    # angle of fit line with respect to horizontal direction in frame
    angle = (-1) * (math.atan2(abs((fit[0] - 0)), abs(1))) * (180 / math.pi)
    # calculate translation distances of edges of search cone with respect to horizontal direction
    x_schaetz = math.cos((math.pi / 180.0) * angle) * (distance + to_add)
    y_schaetz = math.sin((math.pi / 180.0) * angle) * (distance + to_add)
    x_first = math.cos((math.pi / 180.0) * (angle - acceptance_angle)) * (distance + to_add)
    y_first = math.sin((math.pi / 180.0) * (angle - acceptance_angle)) * (distance + to_add)
    x_second = math.cos((math.pi / 180.0) * (angle + acceptance_angle)) * (distance + to_add)
    y_second = math.sin((math.pi / 180.0) * (angle + acceptance_angle)) * (distance + to_add)
    # determination of cone tip and edge points, for each case
    schaetz_point = None
    if last_droplet[0] - droplets_list[0][1][0] < 0 and fit[0] <= 0:
        schaetz_point = [last_droplet[0] - x_schaetz, last_droplet[1] - y_schaetz]
        first_point = [last_droplet[0] - x_first, last_droplet[1] - y_first]
        second_point = [last_droplet[0] - x_second, last_droplet[1] - y_second]
    elif last_droplet[0] - droplets_list[0][1][0] < 0 and fit[0] > 0:
        schaetz_point = [last_droplet[0] - x_schaetz, last_droplet[1] + y_schaetz]
        first_point = [last_droplet[0] - x_first, last_droplet[1] + y_first]
        second_point = [last_droplet[0] - x_second, last_droplet[1] + y_second]
    elif last_droplet[0] - droplets_list[0][1][0] > 0 and fit[0] > 0:
        schaetz_point = [last_droplet[0] + x_schaetz, last_droplet[1] - y_schaetz]
        first_point = [last_droplet[0] + x_first, last_droplet[1] - y_first]
        second_point = [last_droplet[0] + x_second, last_droplet[1] - y_second]
    elif last_droplet[0] - droplets_list[0][1][0] > 0 and fit[0] <= 0:
        schaetz_point = [last_droplet[0] + x_schaetz, last_droplet[1] + y_schaetz]
        first_point = [last_droplet[0] + x_first, last_droplet[1] + y_first]
        second_point = [last_droplet[0] + x_second, last_droplet[1] + y_second]
    elif last_droplet[0] - droplets_list[0][1][0] == 0:
        if last_droplet[1] - droplets_list[0][1][1] > 0:
            schaetz_point = [last_droplet[0] + x_schaetz, last_droplet[1] + y_schaetz]
            first_point = [last_droplet[0] + x_first, last_droplet[1] + y_first]
            second_point = [last_droplet[0] + x_second, last_droplet[1] + y_second]
        elif last_droplet[1] - droplets_list[0][1][1] < 0:
            schaetz_point = [last_droplet[0] - x_schaetz, last_droplet[1] - y_schaetz]
            first_point = [last_droplet[0] - x_first, last_droplet[1] - y_first]
            second_point = [last_droplet[0] - x_second, last_droplet[1] - y_second]
    if not schaetz_point == None:
        # compute final plane of search cone (edge of cylindrical section of search space)
        distance_first = schaetz_point[1] - first_point[1]
        distance_second = schaetz_point[1] - second_point[1]
        abstand = int((int(first_point[0]) - int(second_point[0])) / 2.0)
        # compute mask with search space
        mask_to_check = np.zeros((frame.shape[0], frame.shape[1])).astype('uint8')
        mask_to_check = cv2.fillPoly(mask_to_check, [np.array([last_droplet, [last_droplet[0] + abstand, last_droplet[1] - distance_first], first_point, schaetz_point, second_point, [last_droplet[0] - abstand, last_droplet[1] - distance_second]]).astype('int32')], 255, 1)
        return mask_to_check
    else:
        mask_to_check = np.zeros((frame.shape[0], frame.shape[1])).astype('uint8')
        return mask_to_check


def angle_straight_lines(slope_1, slope_2):
    """
    - computes angle between two lines using line slopes in degrees
    :param slope_1: slope of line 1
    :param slope_2: slope of line 2
    :return: angle between lines 1 and 2 in degrees
    """
    # apply addition theorem for inverse tangent
    angle = math.atan2(abs((slope_1 - slope_2)), abs(1 + slope_1 * slope_2))
    return (angle / math.pi) * 180


def iterative_impact(trajectory, frame_numbers):
    """
    - executes iterative procedure to differentiate droplet impact and rebound events
    :param trajectory: sampling points on droplet trajectory
    :param frame_numbers: frame indices corresponding to sampling points
    :return: angle: droplet rebound angle (zero degrees if no rebound detected), list_first: sampling points on first fit line,
             list_second: sampling points on second fit line, frame_numbers[index]: frame index corresponding to droplet rebound
    """
    # convert sampling points into NumPy array
    sample = np.array(trajectory)
    # create lists
    error_sum = list()
    fits_first = list()
    fits_second = list()
    # iterate over all sampling points and add to lists for two subsets
    for i in range(0, len(sample) - 6):
        # first subset
        first_points = sample[0:i + 3]
        # second subset
        second_points = sample[i + 3:len(sample) - 1]
        # identify x and y coordinates of first subset
        x_first = list()
        y_first = list()
        for point in first_points:
            x_first.append(point[0])
            y_first.append(point[1])
        # fit first subset with line
        fit_first = np.polyfit(x_first, y_first, 1)
        # save slope and y-axis intercept of first list for later use
        fits_first.append((fit_first[0], fit_first[1]))
        # calculate error sum of first fit line
        error_sum_first = 0
        for point in first_points:
            # ordinary least squares approach
            error_sum_first = error_sum_first + abs(point[1] - (point[0] * fit_first[0] + fit_first[1]))

            # total least squares approach
            # calculate coordinates of point on line at orthogonal distance to current point
            # x_line = (point[0] + fit_first[0] * point[1] - fit_first[0] * fit_first[1]) / (1.0 + (fit_first[1])**2)
            # y_line = fit_first[0] * x_line + fit_first[1]
            # calculate orthogonal distance to line for current point
            # error_sum_first = error_sum_first + np.sqrt((point[0] - x_line)**2 + (point[1] - y_line)**2)

        # identify x and y coordinates of second subset
        x_second = list()
        y_second = list()
        for point in second_points:
            x_second.append(point[0])
            y_second.append(point[1])
        # fit second subset with line
        fit_second = np.polyfit(x_second, y_second, 1)
        # save slope and y-axis intercept of second list for later use
        fits_second.append((fit_second[0], fit_second[1]))
        # calculate error sum of second fit line
        error_sum_second = 0
        for point in second_points:
            # ordinary least squares approach
            error_sum_second = error_sum_second + abs(point[1] - (point[0] * fit_second[0] + fit_second[1]))

            # total least squares approach
            # calculate coordinates of point on line at orthogonal distance to current point
            # x_line = (point[0] + fit_second[0] * point[1] - fit_second[0] * fit_second[1]) / (1.0 + (fit_second[1])**2)
            # y_line = fit_second[0] * x_line + fit_second[1]
            # calculate orthogonal distance to line for current point
            # error_sum_second = error_sum_second + np.sqrt((point[0] - x_line)**2 + (point[1] - y_line)**2)

        # add first and second error sum to total error sum
        # add total error sum to list for later use
        error_sum.append(error_sum_first + error_sum_second)

    # identify index with lowest total error sum
    index = np.argmin(error_sum)
    # create lists for final subsets
    list_first = list()
    list_second = list()
    # assert which sampling point should be assigned to which subset
    for point in trajectory:
        # use ordinary least squares approach
        if abs(point[1] - (point[0] * fits_first[index][0] + fits_first[index][1])) <= \
                abs(point[1] - (point[0]*fits_second[index][0]+fits_second[index][1])):
            list_first.append(point)
        else:
            list_second.append(point)

        # # use total least squares approach
        # # calculate coordinates of point on first line at orthogonal distance to current point
        # x_line_first = (point[0] + fits_first[index][0] * point[1] - fits_first[index][0] * fits_first[index][1]) /
        # (1.0 + fits_first[index][1] ** 2)
        # y_line_first = fits_first[index][0] * x_line_first + fits_first[index][1]
        # # calculate coordinates of point on second line at orthogonal distance to current point
        # x_line_second = (point[0] + fits_second[index][0] * point[1] - fits_second[index][0] * fits_second[index][1])
        # / (1.0 + fits_second[index][1] ** 2)
        # y_line_second = fits_second[index][0] * x_line_second + fits_second[index][1]
        #
        # if (np.sqrt((point[0] - x_line_first) ** 2 + (point[1] - y_line_first) ** 2)) <= (np.sqrt((point[0] -
        # x_line_second) ** 2 + (point[1] - y_line_second) ** 2)):
        #     list_first.append(point)
        # else:
        #     list_second.append(point)

    # compute angle between directions
    angle = angle_straight_lines(fits_first[index][0], fits_second[index][0])
    # frame = cv2.line(frame, (0, int(0 * fits_first[index][0] + fits_first[index][1])),
    # (frame.shape[1], int(frame.shape[1]*fits_first[index][0] + fits_first[index][1])), [155, 88, 0], 1)
    # frame = cv2.line(frame, (0, int(0 * fits_second[index][0] + fits_second[index][1])),
    # (frame.shape[1], int(frame.shape[1] * fits_second[index][0] + fits_second[index][1])), [41, 123, 231],1)
    return angle, list_first, list_second, frame_numbers[index]


def getMainTrajectoryUntilImpact(droplets_list, frame_impact):
    """
    - determines objects on principal trajectory
    - (objects occuring before rebound event)
    :param droplets_list: list with all objects (before and after rebound event)
    :param frame_impact: instant of droplet impact/rebound
    :return: list of objects corresponding to principal trajectory
    """
    test_list = droplets_list
    test_list_abprall = list()
    for point in test_list:
        if point[0] <= frame_impact:
            test_list_abprall.append(point)
    droplets_list = test_list_abprall
    return droplets_list


def get_slope(point_1, point_2):
    """
    - computes slope of line based on two points on line
    :param point_1: point 1 on line
    :param point_2: point 2 on line
    :return: slope of line through points 1 and 2
    """
    # determine x and y coordinates of points
    x_1 = point_1[0]
    y_1 = point_1[1]
    x_2 = point_2[0]
    y_2 = point_2[1]
    if x_1 == x_2:
        x_1 = x_1 + 1
    return (y_2 - y_1) / (x_2 - x_1)


def get_y_intercept(point_1, slope):
    """
    - determines y-axis intercept of a line given by slope and point on line
    :param point_1: point on line
    :param slope: slope of line
    :return: y-axis intercept of line
    """
    # determine x and y coordinates of point
    x = point_1[0]
    y = point_1[1]
    return y - slope * x
