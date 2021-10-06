import cv2
import numpy as np
import math
from scipy import ndimage


def segment_glottis_point(watershed, glottis_point_x, glottis_point_y, size_x, size_y):
    """
    - identifies segment of watershed transformation containing reference point
    :param watershed: marker image with segments of watershed transformation
    :param glottis_point_x: x coordinate of reference point
    :param glottis_point_y: y coordinate of reference point
    :param size_x: frame size in horizontal direction
    :param size_y: frame size in vertical direction
    :return: index: index of segment of watershed transformation containing reference point
    """
    index = watershed[glottis_point_y, glottis_point_x]
    max_index = np.max(watershed)
    if index == -1:
        for i in range(1, max_index + 1):
            if i in watershed:
                mask_point = np.zeros((size_y, size_x)).astype('uint8')
                mask_point[glottis_point_y, glottis_point_x] = 255
                mask = np.zeros((size_y, size_x)).astype('uint8')
                mask[watershed == i] = 255
                kernel = np.ones((3, 3)).astype('uint8')
                mask = cv2.dilate(mask, kernel)
                result = cv2.bitwise_and(mask, mask_point)
                if cv2.countNonZero(result) > 0:
                    index = i
                    break
    return index


def watershed_segmentation(frame, labels):
    """
    - applies watershed transformation
    :param frame: input frame
    :param labels: marker image with "valleys"
    :return: marker image with watershed regions
    """
    return cv2.watershed(frame, labels)


def getLabels(frame, first, last):
    """
    - creates marker image for watershed transformation
    - 1. edge detection (Canny algorithm)
    - 2. distance transformation
    - 3. image normalization
    - 4. image binarization using global threshold
    - 5. erosion
    - 6. application of cv2.connectedComponents to obtain labels of "valleys" in image
    :param frame: input frame
    :param first: hysteresis threshold value (high)
    :param last: hysteresis threshold value (low)
    :return: marker image
    """
    # apply edge detection
    canny_image = cv2.Canny(frame, first, last)
    # apply distance transformation (distance to edge)
    distance_image = cv2.distanceTransform(cv2.bitwise_not(canny_image), cv2.DIST_L2, 3)
    # bring distance transformation to range [0, 1]
    normalized_image = cv2.normalize(distance_image, distance_image, 0, 1, cv2.NORM_MINMAX)
    # binarization: bring distance transformation to values of 0 or 255
    ret, thresh = cv2.threshold(normalized_image, 0.05, 255, cv2.THRESH_BINARY)
    # image erosion
    kernel = np.ones((3, 3)).astype('uint8')
    erode_image = cv2.erode(thresh, kernel=kernel)
    # create labels for "valleys" in image (preparation for watershed transform)
    ret, label = cv2.connectedComponents(np.array(erode_image).astype('uint8'))
    return label


def getBoundaryMask(shape_x, shape_y):
    """
    - computes mask to check if segment is located at frame boundary
    :param shape_x: frame size in horizontal direction
    :param shape_y: frame size in vertical direction
    :return: frame boundary mask
    """
    boundary_mask = np.zeros((shape_y, shape_x)).astype('uint8')
    boundary_mask[0, 0:shape_x - 1] = 255
    boundary_mask[shape_y - 1, 0:shape_x - 1] = 255
    boundary_mask[0:shape_y - 1, 0] = 255
    boundary_mask[0:shape_y - 1, shape_x - 1] = 255
    return boundary_mask


def isSegmentConnectedToBoundaryMask(shape_x, shape_y, watershed, index, boundary_mask):
    """
    - asserts if a watershed transformation segment is connected to frame boundaries
    :param shape_x: frame size in horizontal direction
    :param shape_y: frame size in vertical direction
    :param watershed: watershed transformation
    :param index: index in marker image (watershed transformation)
    :param boundary_mask: frame boundary mask to check connection to frame boundaries
    :return: True: connection to frame boundaries found; False: no connection to frame boundaries found
    """
    mask = np.zeros((shape_y, shape_x)).astype('uint8')
    mask[watershed == index] = 255
    kernel = np.ones((3, 3)).astype('uint8')
    mask = cv2.dilate(mask, kernel)
    mask = cv2.bitwise_and(mask, boundary_mask)
    if cv2.countNonZero(mask) > 0:
        return True
    else:
        return False


def getCannyThreshold(frame, x, y):
    """
    - iteratively identifies optimum hysteresis threshold value for Canny edge detector
    - for automated segmentation of glottis with watershed transformation approach
    - after segmentation with threshold value: reference point located in single segment with an average
    - intensity value deviating from intensity value of reference point by max. 30 (segment not connected
    - to frame boundaries) or 10 (segment connected to frame boundaries)
    :param frame: input frame
    :param x: x coordinate of reference point inside glottis
    :param y: y coordinate of reference point inside glottis
    :return: optimum hysteresis threshold value for Canny edge detector
    """
    get_canny = True
    thresh_array = [100, 95, 90, 85, 80, 75, 70, 65, 60, 55, 50, 45, 40, 35, 30, 25, 20, 15, 10, 5, 0]
    index = 0
    boundary_mask = getBoundaryMask(frame.shape[1], frame.shape[0])
    # initialize "mean" with reference point inside glottis
    mean = frame[y, x]
    while get_canny:
        if index > 20:
            canny_thresh = thresh_array[20]
            get_canny = False
        else:
            # iterate over threshold values in "thresh_array"
            thresh = thresh_array[index]
            # create labels of "valleys" in image
            label = getLabels(frame, thresh, thresh)
            # apply watershed transformation on image using labels
            watershed = watershed_segmentation(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR), label)
            # get index of segment containing reference point
            index_segment = segment_glottis_point(watershed, x, y, frame.shape[0], frame.shape[1])
            # assert if segment is connected to image boundaries
            connectedToBoundary = isSegmentConnectedToBoundaryMask(frame.shape[1], frame.shape[0], watershed,
                                                                   index_segment, boundary_mask)
            # if segment connected to boundaries
            if connectedToBoundary:
                if mean + 10 < np.mean(frame[watershed == index_segment]):
                    index = index + 1
                    get_canny = True
                else:
                    get_canny = False
            else:
                if mean + 30 < np.mean(frame[watershed == index_segment]):
                    index = index + 1
                    get_canny = True
                else:
                    get_canny = False
            if index < 21:
                canny_thresh = thresh_array[index]
            else:
                pass
    return canny_thresh, canny_thresh


def getGlottisContourFromWatershed(shape_x, shape_y, watershed, index):
    """
    - derives glottis contour from watershed segmentation result
    :param shape_x: size of original frame in horizontal direction
    :param shape_y: size of original frame in vertical direction
    :param watershed: watershed transformation
    :param index: label of segment corresponding to glottis
    :return: glottis contour
    """
    mask = np.zeros((shape_y, shape_x)).astype('uint8')
    mask[watershed == index] = 255
    glottis_contour = []
    # only retrieve external contour (RETR_EXTERNAL), horizontal/vertical segments are omitted (CHAIN_APPROX_SIMPLE)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    glottis_contour = contours[0]
    return glottis_contour


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


def getGridForRegionGrowing(shape_x, shape_y, extLeft, extRight, extTop, extBot):
    """
    - creates grid mask for the determination of seed points of region growing procedure
    :param shape_x: horizontal frame size
    :param shape_y: vertical frame size
    :param extLeft: left extremal point on glottis contour
    :param extRight: right extremal point on glottis contour
    :param extTop: upper extremal point on glottis contour
    :param extBot: lower extremal point on glottis contour
    :return: mask with grid points
    """
    mask_grid = np.zeros((shape_y, shape_x)).astype('uint8')
    # define factor for vertical number of grid points
    factor_i = (extBot[1] - extTop[1]) / 10.0
    # define factor for horizontal number of grid points
    factor_j = (extRight[0] - extLeft[0]) / 10.0
    # if every pixel would become a grid point in vertical direction (step size 1 px)
    if int(factor_i) == 1:
        # reduce number of grid points
        factor_i = (extBot[1] - extTop[1]) / 5.0
        if int(factor_i) == 1:
            # further reduce number of grid points
            factor_i = (extBot[1] - extTop[1]) / 2.0
            # abort
            if int(factor_i) == 1:
                factor_i = 0
    # if every pixel would become a grid point in horizontal direction (step size 1 px)
    if int(factor_j) == 1:
        # reduce number of grid points
        factor_j = (extBot[1] - extTop[1]) / 5.0
        if int(factor_j) == 1:
            # further reduce number of grid points
            factor_j = (extBot[1] - extTop[1]) / 2.0
            if int(factor_j) == 1:
                # abort
                factor_j = 0
    # if grid creation successful
    if not int(factor_i) == 0 and not int(factor_j) == 0:
        # set grid points to value of 255
        for i in range(extTop[1], extBot[1], int(factor_i)):
            for j in range(extLeft[0], extRight[0], int(factor_j)):
                mask_grid[i, j] = 255
    return mask_grid


def getSeedPointsRegionGrowingFirstFrame(shape_x, shape_y, glottis_contour, mask_grid):
    """
    - identifies seed points for region growing procedure from mask and glottis contour
    - (each white point is seed point)
    :param shape_x: frame size in horizontal direction
    :param shape_y: frame size in vertical direction
    :param glottis_contour: contour points of glottis
    :param mask_grid: mask with seed points
    :return: seed_points: list with seed points
    """
    # initialize mask of zero values with identical dimensions as frame
    mask_glottis = np.zeros((shape_y, shape_x)).astype('uint8')
    # draw glottis contour
    mask_glottis = cv2.drawContours(mask_glottis, [glottis_contour], 0, 255, -1)
    # calculate glottis area in px
    area = cv2.contourArea(glottis_contour)
    # if glottis area covers less than 1.5% of total frame surface
    if area/(shape_x * shape_y) < 0.015:
        kernel = np.ones((3, 3)).astype('uint8')
    else:
        kernel = np.ones((5, 5)).astype('uint8')
    # erode glottis mask
    mask_glottis = cv2.erode(mask_glottis, kernel)
    # resize eroded glottis mask by 200% and visualize to user
    mask_glottis_large = cv2.resize(mask_glottis, (int(2.0 * mask_glottis.shape[1]), int(2.0 * mask_glottis.shape[0])),
                             interpolation=cv2.INTER_LINEAR)
    cv2.imshow("Glottis", mask_glottis_large)
    # find grid points inside glottis area
    mask_glottis = cv2.bitwise_and(mask_glottis, mask_grid)
    # create list of seed points inside glottis
    index_glottis = np.where(mask_glottis == 255)
    seed_points = list()
    for i in range(0, len(index_glottis[0])):
        seed_points.append([index_glottis[1][i], index_glottis[0][i]])
    return seed_points


def regionGrowing(frame_gray, seed_points, homogeneity_criterion):
    """
    - region growing algorithm (own implementation)
    - region growing method starting from provided set of seed points
    :param frame_gray: input frame
    :param seed_points: seed points
    :return: segmented_frame: mask with segmented regions
    """
    # initialize output mask with 0 values (segmented pixels will be set to 255)
    segmented_frame = np.zeros((frame_gray.shape[0], frame_gray.shape[1])).astype('uint8')
    # iterate over all seed points
    for seed_point in seed_points:
        # pass if current seed point already included in output mask
        if segmented_frame[seed_point[1], seed_point[0]] == 255:
            pass
        # current location not yet included in output mask
        else:
            # initialize mask with 0 values
            mask_to_add = np.zeros((frame_gray.shape[0], frame_gray.shape[1])).astype('uint8')
            # set current location in mask to 255
            mask_to_add[seed_point[1], seed_point[0]] = 255
            # initialize mask with processed pixel locations with 0 values
            mask_done = np.zeros((frame_gray.shape[0], frame_gray.shape[1])).astype('uint8')
            # activate region growing for current seed point
            region_growing = True
            while region_growing:
                # initialize 4-connected connectivity
                kernel = np.zeros((3, 3)).astype('uint8')
                kernel[0, 1] = 1
                kernel[1, 0] = 1
                kernel[1, 1] = 1
                kernel[1, 2] = 1
                kernel[2, 1] = 1
                # dilate "mask to add" with 4-connected kernel
                mask_dilate = cv2.dilate(mask_to_add, kernel)
                # initialize "candidate mask" as dilated "mask to add", masked with inverted "mask to add"
                # yields candidate pixels around currently segmented region
                mask_candidates = cv2.bitwise_and(mask_dilate, cv2.bitwise_not(mask_to_add))
                # remove already processed pixel locations from set of candidate pixels
                mask_candidates = cv2.bitwise_and(mask_candidates, cv2.bitwise_not(mask_done))
                # define candidate pixels as pixels in "candidate mask" with value of 255
                candidates = np.where(mask_candidates == 255)
                # calculate mean intensity value of segmented area of grayscale input frame defined by "mask to add"
                mean = np.mean(frame_gray[mask_to_add == 255])
                # initialize Boolean for region growing procedure
                check = False
                # iterate over candidate pixels
                for i in range(len(candidates[0])):
                    # get intensity value of current candidate pixel
                    mean_point = np.mean(frame_gray[candidates[0][i], candidates[1][i]])
                    # if mean intensity value of segmented area above threshold
                    if not mean < 20:
                        # check criterion for candidate pixel
                        if mean_point <= mean * homogeneity_criterion:
                            # add candidate pixel to segmented region
                            mask_to_add[candidates[0][i], candidates[1][i]] = 255
                            check = True
                        else:
                            # do not add candidate pixel to segmented region, mark candidate pixel as processed
                            mask_done[candidates[0][i], candidates[1][i]] = 255
                    # if mean intensity value of segmented area below threshold
                    else:
                        # if intensity value of candidate point below threshold
                        if mean_point < 20:
                            # add candidate pixel to segmented region
                            mask_to_add[candidates[0][i], candidates[1][i]] = 255
                            check = True
                        else:
                            mask_done[candidates[0][i], candidates[1][i]] = 255
                if not check:
                    # stop region growing procedure for current seed point
                    region_growing = False
        # add newly segmented candidate pixels for current seed point to mask of global segmentation result
        segmented_frame[mask_to_add == 255] = 255
    # initialize circular kernel for morphological closing operation

    kernel = np.zeros((11, 11)).astype('uint8')
    kernel = cv2.circle(kernel, (5, 5), 5, 1, -1)
    segmented_frame = cv2.morphologyEx(segmented_frame, cv2.MORPH_CLOSE, kernel)
    frame_result = frame_gray.copy()
    frame_result[segmented_frame == 255] = 255
    return segmented_frame


def getGlottisContourRegionGrowing(segmented_image):
    """
    - determines glottis contour after region growing
    :param segmented_image: region growing result image
    :return: points of glottis contour
    """
    glottis_contour = []
    contours, hierarchy = cv2.findContours(segmented_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        glottis_contour = contours[0]
    return glottis_contour


def getGlottisContourRegionGrowingDense(segmented_image):
    """
    - determines dense glottis contour after region growing
    :param segmented_image: region growing result image
    :return: points of dense glottis contour (any two subsequent points in 'glottis_contour' will be neighboring pixels)
    """
    glottis_contour = []
    contours, hierarchy = cv2.findContours(segmented_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) > 0:
        glottis_contour = contours[0]
    return glottis_contour


def getHomogeneityCriterion(frame, seed_points):
    """
    - determines homogeneity criterion for glottal segmentation by region growing
    :param frame: input frame
    :param seed_points: seed points of region growing method
    :return: homogeneity criterion in range [1.02;1.15]
    """
    y_coordinates = list()
    # store vertical coordinates of all seed points in list
    for seed in seed_points:
        y_coordinates.append(seed[1])
    # remove redundant values from list of vertical coordinates
    y_coordinates = list(set(y_coordinates))
    # initialize value of homogeneity criterion
    min_value = 10
    # iterate over all rows of seed points in frame
    for y in y_coordinates:
        # initialize horizontal coordinate of leftmost seed point
        min = 255
        # iterate over all seed points
        for seed in seed_points:
            # if current seed point on current row y
            if seed[1] == y:
                # check horizontal position of current seed point on row y
                if min > seed[0]:
                    # update value of horizontal coordinate of leftmost seed point
                    min = seed[0]
        # initialize horizontal coordinate of rightmost seed point
        max = 0
        # iterate over all seed points
        for seed in seed_points:
            # if current seed point on current row y
            if seed[1] == y:
                # check horizontal position of current seed point on row y
                if max < seed[0]:
                    # update value of horizontal coordinate of rightmost seed point
                    max = seed[0]
        # compare average intensity value of five pixels on the left of leftmost seed point to value of seed point
        if np.mean(frame[y, min - 5:min])/frame[y, min] < min_value:
            # update value of homogeneity criterion (take minimum value of all seed point rows)
            min_value = np.mean(frame[y, min - 5:min])/frame[y, min]
        # compare average intensity value of five pixels on the right of leftmost seed point to value of seed point
        if np.mean(frame[y, max:max + 5])/frame[y, max] < min_value:
            # update value of homogeneity criterion (take minimum value of all seed point rows)
            min_value = np.mean(frame[y, max:max + 5])/frame[y, max]
    # update value of homogeneity criterion
    # value 2.06 determined empirically
    min_value = 2.06 - min_value
    if min_value > 1.15:
        min_value = 1.15
    if min_value < 1.02:
        min_value = 1.02
    return min_value


def getHomogeneityCriteriaRefined(frame, seed_points):
    """
    - determines homogeneity criteria for glottal segmentation by region growing (for each row of seed points)
    :param frame: input frame
    :param seed_points: seed points of region growing method
    :return: array of homogeneity criteria in range [1.02;1.15] for each row of seed points
    """
    y_coordinates = list()
    # store vertical coordinates of all seed points in list
    for seed in seed_points:
        y_coordinates.append(seed[1])
    # remove redundant values from list of vertical coordinates
    y_coordinates = list(set(y_coordinates))
    # sort vertical coordinates
    y_coordinates_sorted = sorted(y_coordinates)
    # initialize array with homogeneity criterion values for each row of seed points
    homogeneity_criteria = np.ones((len(y_coordinates_sorted), 1))
    homogeneity_criteria *= 10
    # initialize counter for seed point rows
    row_counter = 0
    # iterate over all sorted rows of seed points in frame (starting with lowest value)
    for y in y_coordinates_sorted:
        # initialize horizontal coordinate of leftmost seed point
        min = 255
        # iterate over all seed points
        for seed in seed_points:
            # if current seed point on current row y
            if seed[1] == y:
                # check horizontal position of current seed point on row y
                if min > seed[0]:
                    # update value of horizontal coordinate of leftmost seed point
                    min = seed[0]
        # initialize horizontal coordinate of rightmost seed point
        max = 0
        # iterate over all seed points
        for seed in seed_points:
            # if current seed point on current row y
            if seed[1] == y:
                # check horizontal position of current seed point on row y
                if max < seed[0]:
                    # update value of horizontal coordinate of rightmost seed point
                    max = seed[0]
        # compare average intensity value of five pixels on the left of leftmost seed point to value of seed point
        if np.mean(frame[y, min - 5:min])/frame[y, min] < homogeneity_criteria[row_counter]:
            # update value of homogeneity criterion for current row of seed points
            homogeneity_criteria[row_counter] = np.mean(frame[y, min - 5:min])/frame[y, min]
        # compare average intensity value of five pixels on the right of leftmost seed point to value of seed point
        if np.mean(frame[y, max:max + 5])/frame[y, max] < homogeneity_criteria[row_counter]:
            # update value of homogeneity criterion for current row of seed points
            homogeneity_criteria[row_counter] = np.mean(frame[y, max:max + 5])/frame[y, max]
        # update value of homogeneity criterion for current row of seed points
        # value 2.06 determined empirically
        homogeneity_criteria[row_counter] = 2.06 - homogeneity_criteria[row_counter]
        if homogeneity_criteria[row_counter] > 1.15:
            homogeneity_criteria[row_counter] = 1.15
        if homogeneity_criteria[row_counter] < 1.02:
            homogeneity_criteria[row_counter] = 1.02
        # increment seed point row counter
        row_counter += 1
    return homogeneity_criteria


def regionGrowingRefined(frame_gray, seed_points, homogeneity_criteria):
    """
    - region growing algorithm (own implementation)
    - region growing method starting from provided set of seed points
    :param frame_gray: input frame
    :param seed_points: seed points
    :return: segmented_frame: mask with segmented regions
    """
    # obtain sorted list of vertical coordinates of seed points
    y_coordinates = list()
    for seed in seed_points:
        y_coordinates.append(seed[1])
    # remove redundant values from list of vertical coordinates
    y_coordinates = list(set(y_coordinates))
    # sort vertical coordinates
    y_coordinates_sorted = sorted(y_coordinates)
    # initialize output mask with 0 values (segmented pixels will be set to 255)
    segmented_frame = np.zeros((frame_gray.shape[0], frame_gray.shape[1])).astype('uint8')
    # iterate over all seed points
    for seed_point in seed_points:
        # pass if current seed point already included in output mask
        if segmented_frame[seed_point[1], seed_point[0]] == 255:
            pass
        # current location not yet included in output mask
        else:
            # initialize mask with 0 values
            mask_to_add = np.zeros((frame_gray.shape[0], frame_gray.shape[1])).astype('uint8')
            # set current location in mask to 255
            mask_to_add[seed_point[1], seed_point[0]] = 255
            # initialize mask with processed pixel locations with 0 values
            mask_done = np.zeros((frame_gray.shape[0], frame_gray.shape[1])).astype('uint8')
            # activate region growing for current seed point
            region_growing = True
            # initialize 4-connected connectivity
            kernel = np.zeros((3, 3)).astype('uint8')
            kernel[0, 1] = 1
            kernel[1, 0] = 1
            kernel[1, 1] = 1
            kernel[1, 2] = 1
            kernel[2, 1] = 1
            while region_growing:
                # dilate "mask to add" with 4-connected kernel
                mask_dilate = cv2.dilate(mask_to_add, kernel)
                # initialize "candidate mask" as dilated "mask to add", masked with inverted "mask to add"
                # yields candidate pixels around currently segmented region
                mask_candidates = cv2.bitwise_and(mask_dilate, cv2.bitwise_not(mask_to_add))
                # remove already processed pixel locations from set of candidate pixels
                mask_candidates = cv2.bitwise_and(mask_candidates, cv2.bitwise_not(mask_done))
                # define candidate pixels as pixels in "candidate mask" with value of 255
                candidates = np.where(mask_candidates == 255)
                # calculate mean intensity value of segmented area of grayscale input frame defined by "mask to add"
                mean = np.mean(frame_gray[mask_to_add == 255])
                # initialize Boolean for region growing procedure
                check = False
                # iterate over candidate pixels
                for i in range(len(candidates[0])):
                    # get intensity value of current candidate pixel
                    mean_point = np.mean(frame_gray[candidates[0][i], candidates[1][i]])
                    # if mean intensity value of segmented area above threshold
                    if not mean < 20:
                        # identify homogeneity criterion value for current candidate pixel
                        min_distance = 100
                        # find seed point row with lowest vertical distance to current candidate pixel
                        # candidates: [0] is vertical coordinate, seed points: [1] is vertical coordinate!
                        for y_seed_point_rows in y_coordinates_sorted:
                            if (abs(candidates[0][i]-y_seed_point_rows)) < min_distance:
                                min_distance = abs(candidates[0][i]-y_seed_point_rows)
                                y_coordinate_min_distance = y_seed_point_rows
                        # identify index of vertical coordinate of closest row of seed points
                        if y_coordinates_sorted.index(y_coordinate_min_distance) < len(homogeneity_criteria):
                            current_homogeneity_criterion = homogeneity_criteria[y_coordinates_sorted.index(y_coordinate_min_distance)]
                        else:
                            current_homogeneity_criterion = homogeneity_criteria[-1]
                        # check homogeneity criterion for candidate pixel
                        if mean_point <= mean * current_homogeneity_criterion:
                            # add candidate pixel to segmented region
                            mask_to_add[candidates[0][i], candidates[1][i]] = 255
                            check = True
                        else:
                            # do not add candidate pixel to segmented region, mark candidate pixel as processed
                            mask_done[candidates[0][i], candidates[1][i]] = 255
                    # if mean intensity value of segmented area below threshold
                    else:
                        # if intensity value of candidate point below threshold
                        if mean_point < 20:
                            # add candidate pixel to segmented region
                            mask_to_add[candidates[0][i], candidates[1][i]] = 255
                            check = True
                        else:
                            mask_done[candidates[0][i], candidates[1][i]] = 255
                if not check:
                    # stop region growing procedure for current seed point
                    region_growing = False
        # add newly segmented candidate pixels for current seed point to mask of global segmentation result
        segmented_frame[mask_to_add == 255] = 255
    # initialize circular kernel for morphological closing operation
    # ORIGINAL 3 LINES BELOW
    # kernel = np.ones((11, 11)).astype('uint8')
    # kernel = cv2.circle(kernel, (5, 5), 5, 255, -1)
    # segmented_frame = cv2.morphologyEx(segmented_frame, cv2.MORPH_CLOSE, kernel)

    kernel = np.zeros((11, 11)).astype('uint8')
    kernel = cv2.circle(kernel, (5, 5), 5, 1, -1)
    segmented_frame = cv2.morphologyEx(segmented_frame, cv2.MORPH_CLOSE, kernel)
    frame_result = frame_gray.copy()
    frame_result[segmented_frame == 255] = 255
    return segmented_frame


def getGlottalOrientation(glottis_contour):
    """
    - calculates orientation of glottal midline in frame
    - function based on https://docs.opencv.org/4.5.2/d1/dee/tutorial_introduction_to_pca.html, accessed on 08/03/2021
    :param glottis_contour: glottis outline, given as set of points
    :return: glottal_rotation_angle: inclination angle of glottal midline with respect to vertical direction in degrees
    """
    # identify number of points in glottis contour
    size_contour = len(glottis_contour)
    # initialize empty NumPy array for storage of re-ordered contour points
    contour_pts = np.empty((size_contour, 2), dtype=np.float64)
    # insert contour points in new array 'contour_pts'
    for i in range(contour_pts.shape[0]):
        # horizontal coordinates of contour points
        contour_pts[i, 0] = glottis_contour[i, 0, 1]
        # vertical coordinates of contour points
        contour_pts[i, 1] = glottis_contour[i, 0, 0]
    print("Contour points for PCA: ", contour_pts)
    # Perform PCA analysis
    mean = np.empty((0))
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(contour_pts, mean)
    # calculate inclination of glottal midline with respect to vertical direction (in radians)
    glottal_rotation_angle = math.atan2(eigenvectors[0, 1], eigenvectors[0, 0])
    # return angle
    return math.degrees(glottal_rotation_angle)


def rotate_frame(frame, angle):
    """
    - rotates image using given angle (OpenCV-based function)
    - source: https://stackoverflow.com/questions/9041681/opencv-python-rotate-image-by-x-degrees-around-specific-point
    - (accessed on 08/03/2021)
    :param frame: frame to be rotated
    :param angle: desired rotation angle in degrees
    :return: frame_rotated: frame after rotation by 'angle' degrees (dimensions match original frame dimensions)
    """
    frame_center = tuple(np.array(frame.shape[1::-1])*0.5)
    rot_mat = cv2.getRotationMatrix2D(frame_center, angle, 1.0)
    frame_rotated = cv2.warpAffine(frame, rot_mat, frame.shape[1::-1], flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=[255, 255, 255])
    return frame_rotated


def rotate_point(center, point, angle):
    """
    - rotates point around center by given angle
    :param center: center of rotation
    :param point: point to be rotated
    :param angle: desired rotation angle in degrees
    :return: point: point after rotation by given angle around given center of rotation
    """
    # translate point
    x_translated = point[0] - center[0]
    y_translated = point[1] - center[1]
    # rotate point
    point[0] = x_translated * math.cos(math.radians(angle)) + y_translated * math.sin(math.radians(angle))
    point[1] = -x_translated * math.sin(math.radians(angle)) + y_translated * math.cos(math.radians(angle))
    # translate point back
    point[0] = int(round(point[0] + center[0]))
    point[1] = int(round(point[1] + center[1]))
    return point
