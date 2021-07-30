import numpy as np


def getMaskForUser(shape_x, shape_y):
    """
    - creates grid mask for interactive glottis contour definition by user
    :param shape_x: size of original frame in horizontal direction
    :param shape_y: size of original frame in vertical direction
    :return: grid mask
    """
    mask = np.zeros((shape_y, shape_x)).astype('uint8')
    for i in range(0, shape_y, 5):
        for j in range(0, shape_x, 5):
            mask[i, j] = 255
    return mask


def getSeedsFromMask(mask):
    """
    - processes user-chosen points on grid mask
    :param mask: mask with user-chosen points
    :return: seed points for region growing procedure
    """
    # find points in mask with value 255
    index = np.where(mask == 255)
    print(index)
    seed_points = list()
    # add points in mask with value 255 to list of seed points for region growing procedure
    for i in range(0, len(index[0])):
        seed_points.append([index[1][i], index[0][i]])
    return seed_points
