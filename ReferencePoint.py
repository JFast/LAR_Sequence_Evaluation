from scipy.optimize import curve_fit
import numpy as np


def calculateIntensityColumn(frame):
    """
    - computes average intensity per column of single frame
    :param frame: frame (matrix) as array
    :return: array with average column intensity per frame (sum of intensity per column / number of rows)
    """
    height = frame.shape[0]
    width = frame.shape[1]
    intensity_array = np.zeros((1, width))
    for i in range(0, width):
        sum_column = 0
        for j in range(0, height):
            sum_column = sum_column + frame[j, i]
        sum_column = sum_column / height
        intensity_array[0, i] = sum_column
    return intensity_array


def calculateIntensityRow(frame):
    """
    - computes average intensity per row of single frame
    :param frame: frame (matrix) as array
    :return: array with average row intensity per frame (sum of intensity per row / number of columns)
    """
    height = frame.shape[0]
    width = frame.shape[1]
    intensityArray = np.zeros((1, height))
    for i in range(0, height):
        sum_row = 0
        for j in range(0, width):
            sum_row = sum_row + frame[i, j]
        sum_row = sum_row/width
        intensityArray[0, i] = sum_row
    return intensityArray


def getIntensityMatrix(matrix, intensity):
    """
    - extends intensity variation matrix by one row
    :param matrix: current intensity variation matrix
    :param intensity: array with average intensity values of frame to be appended
    :return: extended intensity variation matrix
    """
    if len(matrix) == 0:
        matrix = intensity
    else:
        matrix = np.concatenate((matrix, intensity), axis=0)
    return matrix


def calculateAverageIntensity(intensityMatrix):
    """
    - computes average intensity variation over frame sequence
    - (sum of intensity variation per column / number of frames to be processed)
    :param intensityMatrix: intensity variation matrix
    :return: average intensity variation vector (array/row vector)
    """
    height = intensityMatrix.shape[0]
    width = intensityMatrix.shape[1]
    averageIntensity = np.zeros((1, width))
    for j in range(0, width):
        sum_intensities = 0
        for i in range(0, height):
            sum_intensities = sum_intensities + intensityMatrix[i][j]
        averageIntensity[0, j] = sum_intensities
    return averageIntensity/height


def calculateTotalIntensity(intensityMatrix, averageIntensity):
    """
    - computes total intensity variation over frame sequence
    :param intensityMatrix: intensity variation matrix S
    :param averageIntensity: average intensity variation vector (array/row vector)
    :return: total intensity variation (array/row vector)
    """
    height = intensityMatrix.shape[0]
    width = intensityMatrix.shape[1]
    totalIntensity = np.zeros((1, width))
    for i in range(0, height):
        totalIntensity = totalIntensity + (abs(intensityMatrix[i, :] - averageIntensity))
    totalIntensity = totalIntensity/height
    return totalIntensity


def gaussianFunction(x, H, A, x0, sigma):
    """
    - gives structure of Gaussian distribution as function of its parameters
    :param x: array with values for which a Gaussian distribution is to be computed
    :param A: span
    :param x0: average value
    :param sigma: standard deviation
    :return: values of Gaussian distribution (array)
    """
    return H + A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))


def fitGaussianFunction(x, y, max_sigma):
    """
    - fits Gaussian distribution to input values
    :param x: column/row indices
    :param y: total intensity variation values for column/row
    :param max_sigma: maximum sigma value
    :return: parameters of fitted Gaussian distribution
    """
    # fit Gaussian distribution to given data points
    # mean = sum(x * y) / sum(y)
    popt, pcov = curve_fit(gaussianFunction, x, y, [min(y), max(y) - min(y), np.argmax(y), max_sigma])
    print(popt)
    return popt


def checkSigma(sigma, value):
    """
    - checks sigma value
    :param sigma: standard deviation of Gaussian fit
    :return: standard deviation (maximum: frame length/frame width)
    """
    if sigma > value:
        sigma = value
    return sigma


def checkX0(x0, totalIntensity):
    """
    - asserts if reference point located inside exploited region of frames
    :param x0: currently computed reference point coordinate of fit
    :param totalIntensity: array with total intensity variations for correction step
    :return: coordinate within exploited row/column range
    """
    if x0 < 0 or x0 > totalIntensity.shape[1]:
        x0 = np.argmax(totalIntensity[0, :])
    return x0


def getBoundaries(mean, sigma, toleranceInterval1, toleranceInterval2):
    """
    - computes boundaries (row/column index) of ROI
    :param mean: average value
    :param sigma: standard deviation
    :param toleranceInterval1: tolerance interval to the left/top
    :param toleranceInterval2: tolerance interval to the right/bottom
    :return: first and second value of boundary
    """
    if sigma < 0:
        first_value = int(mean + toleranceInterval1 * sigma)
        second_value = int(mean - toleranceInterval2 * sigma)
    else:
        first_value = int(mean - toleranceInterval1 * sigma)
        second_value = int(mean + toleranceInterval2 * sigma)
    return first_value, second_value


def checkBoundaries(minBoundary, maxBoundary):
    """
    - checks ROI boundaries
    - boundaries must not be located beyond frame size to guarantee correct cropping
    :param minBoundary: current minimum boundary
    :param maxBoundary: current maximum boundary
    :return: minimum/maximum boundary
    """
    if minBoundary < 0:
        minBoundary = 0
    if maxBoundary > 255:
        maxBoundary = 255
    return minBoundary, maxBoundary
