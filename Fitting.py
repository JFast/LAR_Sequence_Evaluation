import numpy as np
import scipy.optimize
import scipy.signal as sg
from scipy import optimize
from scipy import misc
import sympy as sym
import numpy.polynomial.polynomial as poly
import math


def sigmoid(x, a, b, c):
    """
    - computes symmetrical sigmoid function without vertical offset based on input parameters
    :param x: input value
    :param a: maximum function value
    :param b: slope
    :param c: horizontal offset
    :return: function value
    """
    # y = (a + ((-a) * (1.0 / (1.0 + np.exp(b * (-(x - c)))))))
    y = a - a/(1.0 + np.exp((-b) * (x - c)))
    return y


def sigmoid_offset(x, a, b, c, d):
    """
    - computes symmetrical sigmoid function with vertical offset based on input parameters
    :param x: input value
    :param a: (maximum function value) - d
    :param b: slope
    :param c: horizontal offset
    :param d: vertical offset
    :return: function value
    """
    # y = (a+d) + ((-a) * (1.0 / (1.0 + np.exp(b * (-(x - c))))))
    y = a - a/(1.0 + np.exp((-b) * (x - c))) + d
    return y


def generalized_logistic_function(x, a, b, c, d, nu):
    """
    - computes asymmetrical sigmoid function with vertical offset based on input parameters
    - (generalized logistic function/Richards growth curve)
    - see Richards (1959) and Goudriaan/van Laar (1994) for details on this fit function
    :param x: input value
    :param a: function span
    :param b: slope
    :param c: horizontal offset
    :param d: vertical offset
    :param nu: skew coefficient
    :return: function value
    """
    y = a - a/pow((1.0 + nu * np.exp((-b) * (x - c))), (1.0/nu)) + d
    return y


def gompertz(x, a, b, c, d):
    """
    - computes 4-parameter asymmetrical Gompertz sigmoid function with vertical offset based on input parameters
    - see Tan (2008, https://www.ncs-conference.org/2008/slides/24/2/CT.pdf) for details on this fit function
    - here: b set to positive value -> a<d and a corresponding to vertical offset for falling evolution
    - falling evolution -> "right side" of function always "more curved" than "left side"
    :param x: input value
    :param a: (total vertical span + d)
    :param b: slope/"asymmetry parameter"
    :param c: horizontal/temporal offset
    :param d: vertical offset
    :return: function value
    """
    y = d + (a-d) * pow(2.0, (-np.exp((-b) * (x - c))))
    return y


def cubic(x, a, b, c, d):
    """
    - computes cubic function based on input parameters
    :param x: input value
    :param a: coefficient of x^3
    :param b: coefficient of x^2
    :param c: coefficient of x
    :param d: constant coefficient
    :return: function value
    """
    return a * pow(x, 3.0) + b * pow(x, 2.0) + c * x + d


def filtering(signal):
    """
    - executes median filtering of a signal for "outlier" removal
    :param signal: signal to be filtered
    :return: median filtered signal
    """
    return sg.medfilt(signal, kernel_size=5)


def getStopFrame(frames, signal):
    """
    - identifies frame index of last frame of adduction phase
    :param frames: frame indices corresponding to signal
    :param signal: signal (data points)
    :return: frame index up to which the signal is to be processed (minimum value of signal,
    or last frame before glottal closure is achieved)
    """
    frame_to_stop = len(frames) - 1
    # for all frames of sequence
    for i in range(0, len(frames) - 1):
        # if no signal could be identified in more than 200 consecutive frames: stop before gap
        if frames[i+1]-frames[i] > 200:
            frame_to_stop = i + 1
    # if no gaps found in signal: choose frame associated with minimum signal value as stop frame
    if frame_to_stop == len(frames) - 1:
        index = np.argmin(np.array(signal))
        frame_to_stop = index
    return frame_to_stop


def getInflectionPoint(signal, frame_to_stop):
    """
    - returns frame index of inflection point of sigmoid function
    - (inflection point expected at 50 percent of maximum signal value)
    :param signal: signal (data points)
    :param frame_to_stop: frame index up to which the signal is to be processed
    :return: frame index of inflection point
    """
    # identify 50 % of maximum function value
    half = (np.max(signal[0:frame_to_stop]) - np.min(signal[0:frame_to_stop]))/2.0
    inflection_point = 0
    inflection_value_found = False
    for i in range(frame_to_stop, -1, -1):
        if not inflection_value_found:
            if signal[i] >= half:
                inflection_point = i
                inflection_value_found = True
    return inflection_point


def sigmoid_fit(frames, signal):
    """
    - computes fit function of available data using symmetrical sigmoid approach without vertical offset
    :param frames: frame indices
    :param signal: signal (data points)
    :return: x: x values of fit function, y: y values of fit function, popt: parameters of fit function
    """
    signal = filtering(signal)
    frame_to_stop = getStopFrame(frames, signal)
    inflection_point = getInflectionPoint(signal, frame_to_stop)
    inflection_point = frames[inflection_point]
    # use SciPy function for fitting
    popt, pcov = optimize.curve_fit(sigmoid, frames[0:frame_to_stop], signal[0:frame_to_stop],
                                    p0=[np.max(signal[0:frame_to_stop]) - np.min(signal[0:frame_to_stop]), 1.0,
                                        inflection_point])
    x = np.linspace(0, frames[frame_to_stop], num=frames[frame_to_stop] + 1)
    y = sigmoid(x, *popt)
    return x, y, popt


def sigmoid_fit_offset(frames, signal):
    """
    - computes fit function of available data using symmetrical sigmoid approach with vertical offset
    :param frames: frame indices
    :param signal: signal (data points)
    :return: x: x values of fit function, y: y values of fit function, popt: parameters of fit function
    """
    signal = filtering(signal)
    frame_to_stop = getStopFrame(frames, signal)
    inflection_point = getInflectionPoint(signal, frame_to_stop)
    inflection_point = frames[inflection_point]
    # use SciPy function for fitting
    popt, pcov = optimize.curve_fit(sigmoid_offset, frames[0:frame_to_stop], signal[0:frame_to_stop],
                                    p0=[np.max(signal[0:frame_to_stop]) - np.min(signal[0:frame_to_stop]), 1.0,
                                        inflection_point, np.min(signal[0:frame_to_stop])],
                                    bounds=([-np.inf, -np.inf, -np.inf, 0.0], [np.inf, np.inf, np.inf, np.inf]))
    x = np.linspace(0, frames[frame_to_stop], num=frames[frame_to_stop] + 1)
    y = sigmoid_offset(x, *popt)
    return x, y, popt


def fit_generalized_logistic_function(frames, signal):
    """
    - computes fit function of available data using asymmetrical sigmoid approach with vertical offset
    - see Richards (1959) and Goudriaan/van Laar (1994) for details on this fit function
    :param frames: frame indices
    :param signal: signal (data points)
    :return: x: x values of fit function, y: y values of fit function, popt: parameters of fit function
    """
    signal = filtering(signal)
    frame_to_stop = getStopFrame(frames, signal)
    inflection_point = getInflectionPoint(signal, frame_to_stop)
    inflection_point = frames[inflection_point]
    # use SciPy function for fitting
    popt, pcov = optimize.curve_fit(generalized_logistic_function, frames[0:frame_to_stop], signal[0:frame_to_stop],
                                    p0=[np.max(signal[0:frame_to_stop]) - np.min(signal[0:frame_to_stop]), 1.0,
                                        inflection_point, np.min(signal[0:frame_to_stop]), 0.5],
                                    bounds=([0.0, 0.0, -np.inf, 0.0, 0.0],
                                            [np.inf, np.inf, np.inf, np.inf, 1.0]))
    x = np.linspace(0, frames[frame_to_stop], num=frames[frame_to_stop] + 1)
    y = generalized_logistic_function(x, *popt)
    return x, y, popt


def fit_gompertz(frames, signal):
    """
    - computes fit function of available data using 4-parameter asymmetrical Gompertz sigmoid function with
    - vertical offset based on input parameters
    - see Tan (2008, https://www.ncs-conference.org/2008/slides/24/2/CT.pdf) for details on this fit function
    - here: a and b set to positive values -> a corresponds to vertical offset in case of falling evolution
    - falling evolution -> "right side" of function always "more curved" than "left side"
    :param frames: frame indices
    :param signal: signal (data points)
    :return: x: x values of fit function, y: y values of fit function, popt: parameters of fit function
    """
    signal = filtering(signal)
    frame_to_stop = getStopFrame(frames, signal)
    inflection_point = getInflectionPoint(signal, frame_to_stop)
    inflection_point = frames[inflection_point]
    # use SciPy function for fitting
    popt, pcov = optimize.curve_fit(gompertz, frames[0:frame_to_stop], signal[0:frame_to_stop],
                                    p0=[np.min(signal[0:frame_to_stop]), 1.0,
                                        inflection_point, np.max(signal[0:frame_to_stop])],
                                    bounds=([0.0, 0.0, -np.inf, 0.0],
                                            [np.inf, np.inf, np.inf, np.inf]))
    x = np.linspace(0, frames[frame_to_stop], num=frames[frame_to_stop] + 1)
    y = gompertz(x, *popt)
    return x, y, popt


def fit_cubic(frames, signal):
    """
    - computes cubic fit function of available data
    :param frames: frame indices
    :param signal: signal (data points)
    :return: x: x values of fit function, y: y values of fit function, popt: parameters of fit function
    """
    signal = filtering(signal)
    frame_to_stop = getStopFrame(frames, signal)
    # use SciPy function for fitting up to frame 'frame_to_stop'
    popt, pcov = optimize.curve_fit(cubic, frames[0:frame_to_stop], signal[0:frame_to_stop])
    x = np.linspace(0, frames[frame_to_stop], num=frames[frame_to_stop] + 1)
    y = cubic(x, *popt)
    return x, y, popt


def straight_line(m, b, x):
    """
    - calculates function value for given equation of a 2D line in slope-intercept form
    :param m: slope of line
    :param b: function value of line at x=0
    :param x: x value
    :return: y value: function value for given x
    """
    return m*x+b


def get_coordinate_form_straight(slope, intercept):
    """
    - transforms equation of a 2D line in slope-intercept form into parametric form
    :param slope: slope of line
    :param intercept: function value of 2D line at x=0
    :return: support and direction vectors of parametric form of 2D line
    """
    # calculate x and y coordinates of support vector
    point_1_x = 0
    point_1_y = straight_line(slope, intercept, point_1_x)
    # calculate second point on line to obtain direction vector
    point_2_x = 255
    point_2_y = straight_line(slope, intercept, point_2_x)
    # support vector
    point_of_reference = (point_1_x, point_1_y)
    # direction vector
    direction_vector = (point_2_x - point_1_x, point_2_y - point_1_y)
    return point_of_reference, direction_vector


def interception_two_lines(slope_1, intercept_1, slope_2, intercept_2):
    """
    - calculates intersection of two 2D lines
    :param slope_1: slope of first line
    :param intercept_1: function value of first line at x=0
    :param slope_2: slope of second line
    :param intercept_2: function value of second line at x=0
    :return: tupel with x and y coordinates of point of intersection
    """
    # support and direction vectors of first line (parametric form)
    point_of_reference_1, direction_vector_1 = get_coordinate_form_straight(slope_1, intercept_1)
    # support and direction vectors of second line (parametric form)
    point_of_reference_2, direction_vector_2 = get_coordinate_form_straight(slope_2, intercept_2)
    # construct system of linear equations (variables on one side, result on other side)
    results = [point_of_reference_2[0]-point_of_reference_1[0], point_of_reference_2[1]-point_of_reference_1[1]]
    direction_vector_1 = [direction_vector_1[0], direction_vector_1[1]]
    direction_vector_2 = [-direction_vector_2[0], -direction_vector_2[1]]
    # solve system of linear equations with NumPy
    result = np.linalg.solve(np.array([direction_vector_1, direction_vector_2]), np.array(results))
    # inject result and compute x and y coordinates of intersection
    x = point_of_reference_1[0]+result[1]*direction_vector_1[0]
    y = point_of_reference_1[1]+result[1]*direction_vector_1[1]
    return (x, y)


def iterative(trajectory):
    """
    - identifies linear fit for droplet trajectory and searches for rebound events
    :param trajectory: sampling point on droplet trajectory
    :return: point of intersection of two fit lines, parameters of two identified fit lines for two subsets
    """
    # transform sampling points into NumPy array
    sample = np.array(trajectory)
    # create lists
    error_sum = list()
    fits_first = list()
    fits_second = list()
    # iterate over all sampling points and add points to lists (separate in two subsets)
    for i in range(0, len(sample) - 6):
        # first subset
        first_points = sample[0:i + 3]
        # second subset
        second_points = sample[i + 3:len(sample) - 1]
        # get x and y coordinates of first subset
        x_first = list()
        y_first = list()
        for point in first_points:
            x_first.append(point[0])
            y_first.append(point[1])
        # compute fit line for first subset
        fit_first = np.polyfit(x_first, y_first, 1)
        # store slope and value at x=0 for later use (fit_first[0] is coefficient of x)
        fits_first.append((fit_first[0], fit_first[1]))
        # calculate error sum for first fit line
        error_sum_first = 0
        for point in first_points:
            # calculate distance from current point to line along y-axis (ordinary least squares approach)
            error_sum_first = error_sum_first + abs(point[1] - (point[0] * fit_first[0] + fit_first[1]))
            # calculate coordinates of point on line at orthogonal distance to current point
            # x_line = (point[0] + fit_first[0] * point[1] - fit_first[0] * fit_first[1]) / (1.0 + pow(fit_first[1], 2))
            # y_line = fit_first[0] * x_line + fit_first[1]
            # print("Coordinates of point on first line at orthogonal distance to current point: ", x_line, y_line)
            # calculate orthogonal distance to line for current point
            # error_sum_first += np.sqrt(pow((point[0] - x_line), 2) + pow((point[1] - y_line), 2))
        # get x and y coordinates of second subset
        x_second = list()
        y_second = list()
        for point in second_points:
            x_second.append(point[0])
            y_second.append(point[1])
        # compute fit line for second subset
        fit_second = np.polyfit(x_second, y_second, 1)
        # store slope and value at x=0 for later use (fit_second[0] is coefficient of x)
        fits_second.append((fit_second[0], fit_second[1]))
        # calculate error sum for second fit line
        error_sum_second = 0
        for point in second_points:
            # calculate distance from current point to line along y-axis (ordinary least squares approach)
            error_sum_second = error_sum_second + abs(point[1] - (point[0] * fit_second[0] + fit_second[1]))
            # calculate coordinates of point on line at orthogonal distance to current point
            # x_line = (point[0] + fit_second[0] * point[1] - fit_second[0] * fit_second[1]) / (1.0 + pow(fit_second[1], 2))
            # y_line = fit_second[0] * x_line + fit_second[1]
            # print("Coordinates of point on second line at orthogonal distance to current point: ", x_line, y_line)
            # calculate orthogonal distance to line for current point
            # error_sum_second += np.sqrt(pow((point[0] - x_line), 2) + pow((point[1] - y_line), 2))
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
        # use total least squares approach
        # calculate coordinates of point on first line at orthogonal distance to current point
        # x_line_first = (point[0] + fits_first[index][0] * point[1] - fits_first[index][0] * fits_first[index][1]) / (1.0 + pow(fits_first[index][1], 2))
        # y_line_first = fits_first[index][0] * x_line_first + fits_first[index][1]
        # print("Coordinates of point on first line at orthogonal distance to current first subset point: ", x_line_first, y_line_first)
        # calculate coordinates of point on second line at orthogonal distance to current point
        # x_line_second = (point[0] + fits_second[index][0] * point[1] - fits_second[index][0] * fits_second[index][1]) / (1.0 + pow((fits_second[index][1]), 2))
        # y_line_second = fits_second[index][0] * x_line_second + fits_second[index][1]

        # if np.sqrt(pow((point[0] - x_line_first), 2) + pow((point[1] - y_line_first), 2)) <= np.sqrt(pow((point[0] - x_line_second), 2) + pow((point[1] - y_line_second), 2)):
            # list_first.append(point)
        # else:
            # list_second.append(point)
        # use ordinary least squares approach
        if abs(point[1] - (point[0] * fits_first[index][0] + fits_first[index][1])) <= abs(point[1] - (point[0]*fits_second[index][0]+fits_second[index][1])):
            list_first.append(point)
        else:
            list_second.append(point)
    intercept = interception_two_lines(fits_first[index][0], fits_first[index][1], fits_second[index][0], fits_second[index][1])
    return intercept, (fits_first[index][0], fits_first[index][1], fits_second[index][0], fits_second[index][1])


def linear_fit(frames, signal):
    """
    - computes linear fit
    :param frames: frame indices corresponding to signal
    :param signal: signal (data points)
    :return: impact: point of intersection of the two fit lines, linear_lar_begin: parameters of fit lines
    """
    frame_to_stop = getStopFrame(frames, signal)
    coordinates_to_fit = list()
    for i in range(frame_to_stop, -1, -1):
        coordinates_to_fit.append((frames[i], signal[i]))
    impact, linear_lar_begin = iterative(coordinates_to_fit)
    return impact, linear_lar_begin


def getValueNoVerticalOffset(x, y, popt, value):
    """
    - identifies a value of interest in a sigmoidal fit curve without vertical offset
    - drop of maximum value (parameter a of symmetrical sigmoid function without vertical offset)
    :param x: frame indices
    :param y: values associated with frame indices x
    :param popt: parameters of fit curve
    :param value: drop in percent (relative to total span of curve) to be evaluated
    :return: frame index corresponding to desired function value
    """
    lar_begin = 0
    lar_found = False
    # for all frames
    for i in range(0, len(y) - 1):
        if not lar_found:
            if y[i] <= (popt[0] * value):
                lar_begin = x[i]
                lar_found = True
    return lar_begin


def getValueWithVerticalOffset(x, y, popt, value):
    """
    - identifies a value of interest in a sigmoidal fit curve with vertical offset
    :param x: frame indices
    :param y: function values associated with frame indices x
    :param popt: parameters of fit curve
    :param value: drop in percent (relative to maximum deflection) to be evaluated
    :return: frame index corresponding to desired function value
    """
    lar_begin = 0
    lar_found = False
    for i in range(0, len(y) - 1):
        if not lar_found:
            if y[i] <= ((popt[0] * value) + popt[3]):
                lar_begin = x[i]
                lar_found = True
    return lar_begin


def getValueGompertz(x, y, popt, value):
    """
    - identifies a value of interest in a Gompertz-like fit curve with vertical offset
    :param x: frame indices
    :param y: function values associated with frame indices x
    :param popt: parameters of fit curve
    :param value: drop in percent (relative to maximum deflection) to be evaluated
    :return: frame index corresponding to desired function value
    """
    lar_begin = 0
    lar_found = False
    for i in range(0, len(y) - 1):
        if not lar_found:
            if y[i] <= (popt[3] * value):
                lar_begin = x[i]
                lar_found = True
    return lar_begin


def getValueCubic(x, y, popt, value):
    """
    - identifies a value of interest in a cubic fit function
    :param x: frame indices
    :param y: function values associated with frame indices x
    :param popt: parameters of fit curve
    :param value: drop in percent (relative to rightmost stationary point of cubic polynomial) to be evaluated
    :return: frame index corresponding to desired function value
    """
    lar_begin = 0
    lar_found = False
    # determine roots of derivative of cubic fit function
    roots_of_derivative = poly.polyroots([popt[2], 2.0*popt[1], 3.0*popt[0]])
    # calculate x coordinate of rightmost stationary point of cubic fit function
    x_coord_rightmost_stationary_point = np.max(roots_of_derivative)
    # only retain values on the right of rightmost stationary point of cubic fit function in array y_red
    if not np.iscomplex(x_coord_rightmost_stationary_point):
        if int(x_coord_rightmost_stationary_point)-1 >= 0:
            y_red = y[int(x_coord_rightmost_stationary_point)-1:len(y)]
        else:
            y_red = y[0:len(y)]
    else:
        y_red = y[0:len(y)]
    for i in range(0, len(y_red) - 1):
        if not lar_found:
            # check if current value below (function value at rightmost stationary point of cubic fit function * value)
            if y_red[i] <= (cubic(x_coord_rightmost_stationary_point, *popt) * value):
                if not np.iscomplex(x_coord_rightmost_stationary_point):
                    if int(x_coord_rightmost_stationary_point-1) >= 0:
                        lar_begin = x[i + int(x_coord_rightmost_stationary_point)-1]
                    else:
                        lar_begin = x[i]
                else:
                    lar_begin = x[i]
                lar_found = True
    return lar_begin


def derivativeSigmoid(x, a, b, c):
    """
    - computes derivative of given sigmoid fit function with or without vertical offset
    :param a: parameter of sigmoid fit function
    :param b: parameter of sigmoid fit function
    :param c: parameter of sigmoid fit function
    :param x: x coordinate value
    :return: second derivative of sigmoid fit function without vertical offset
    """
    return (-a) * b * (np.exp((-b) * (x - c)) / pow(1.0 + np.exp((-b) * (x - c)), 2.0))


def derivativeSigmoidVertOffset(x, a, b, c, d):
    """
    - computes derivative of given sigmoid fit function with vertical offset
    :param a: parameter of sigmoid fit function with vertical offset
    :param b: parameter of sigmoid fit function with vertical offset
    :param c: parameter of sigmoid fit function with vertical offset
    :param c: parameter of sigmoid fit function with vertical offset
    :param x: x coordinate value
    :return: second derivative of sigmoid fit function without vertical offset
    """
    return (-a) * b * (np.exp((-b) * (x - c)) / pow(1.0 + np.exp((-b) * (x - c)), 2.0))


def getMaxAngVelocitySigmoidNoVertOffset(a, b, c):
    """
    - identifies maximum slope of a sigmoid fit function without vertical offset
    :param a: parameter of sigmoid fit curve without vertical offset
    :param b: parameter of sigmoid fit curve without vertical offset
    :param c: parameter of sigmoid fit curve without vertical offset
    :return: maximum slope of fit function
    """
    # maximum slope occurs at x = c
    max_slope = derivativeSigmoid(c, a, b, c)
    return max_slope


def getMaxAngVelocitySigmoidWithVertOffset(a, b, c, d):
    """
    - identifies maximum slope of a sigmoid fit function with vertical offset
    :param a: parameter of sigmoid fit curve without vertical offset
    :param b: parameter of sigmoid fit curve without vertical offset
    :param c: parameter of sigmoid fit curve without vertical offset
    :param d: parameter of sigmoid fit curve without vertical offset
    :return: maximum slope of fit function
    """
    # maximum slope occurs at x = c
    max_slope_offset = derivativeSigmoid(c, a, b, c)
    return max_slope_offset


def getMaxAngVelocityGeneralizedLogisticFunction(a, b, c, d, nu):
    """
    - identifies maximum slope of a generalized logistic function
    :param a: parameter of generalized logistic function
    :param b: parameter of generalized logistic function
    :param c: parameter of generalized logistic function
    :param d: parameter of generalized logistic function
    :param nu: parameter of generalized logistic function
    :return: maximum slope of generalized logistic function
    """
    def derivativeGeneralizedLogisticFunction(x):
        """
        - returns derivative of given generalized logistic function
        :param x: x coordinate value
        :return: derivative of generalized logistic function
        """
        return (-a) * b * np.exp((-b) * (x - c)) * pow(1.0 + nu * np.exp((-b) * (x - c)), (((-1.0) - nu) / nu))

    # find minimum in derivative
    min_derivative_glf = optimize.minimize_scalar(derivativeGeneralizedLogisticFunction)
    # print("Frame index with maximum angular velocity (generalized logistic function): ", min_derivative_glf.x)
    max_slope_glf = derivativeGeneralizedLogisticFunction(min_derivative_glf.x)
    return max_slope_glf


def derivativeGLF(x, a, b, c, d, nu):
    """
    - returns value of derivative of given generalized logistic function at x
    :param x: x coordinate value
    :param a: parameter a of generalized logistic function
    :param b: parameter b of generalized logistic function
    :param c: parameter c of generalized logistic function
    :param d: parameter d of generalized logistic function
    :param nu: parameter nu of generalized logistic function
    :return: derivative of generalized logistic function at x
    """
    return (-a) * b * np.exp((-b) * (x - c)) * pow(1.0 + nu * np.exp((-b) * (x - c)), (((-1.0) - nu) / nu))


def derivativeGompertz(x, a, b, c, d):
    """
    - returns value of derivative of given Gompertz-like fit function at x
    :param x: x coordinate value
    :param a: parameter a of Gompertz-like fit function
    :param b: parameter b of Gompertz-like fit function
    :param c: parameter c of Gompertz-like fit function
    :param d: parameter d of Gompertz-like fit function
    :return: derivative of Gompertz-like fit function at x
    """
    # symbolic calculation of derivatives of Gompertz-like fit function
    x_symbol_gompertz = sym.Symbol('x_symbol_gompertz')
    # calculate first derivative
    diff_gompertz = sym.diff(d + (a - d) * 2.0 ** (-sym.exp((-b) * (x_symbol_gompertz - c))), x_symbol_gompertz)
    # calculate value of first derivative at x
    derivative_at_x_gompertz = diff_gompertz.evalf(subs={x_symbol_gompertz: x})
    return float(derivative_at_x_gompertz)


def derivativeCubic(x, a, b, c, d):
    """
    - returns value of derivative of given cubic fit function at x
    :param x: x coordinate value
    :param a: parameter a of cubic fit function
    :param b: parameter b of cubic fit function
    :param c: parameter c of cubic fit function
    :param c: parameter d of cubic fit function
    :return: derivative of cubic fit function at x
    """
    return 3 * a * pow(x, 2.0) + 2 * b * x + c


def getMaxAngVelocityGompertz(a, b, c, d):
    """
    - identifies maximum slope of a Gompertz-like fit function
    :param a: parameter of Gompertz-like fit function
    :param b: parameter of Gompertz-like fit function
    :param c: parameter of Gompertz-like fit function
    :param d: parameter of Gompertz-like fit function
    :return: maximum slope of generalized logistic function
    """
    # symbolic calculation of derivatives of Gompertz-like fit function
    x_symbol_gompertz = sym.Symbol('x_symbol_gompertz')
    # first derivative (to obtain maximum slope)
    diff_gompertz = sym.diff(d + (a - d) * 2.0**(-sym.exp((-b) * (x_symbol_gompertz - c))), x_symbol_gompertz)
    # second derivative (to find location of maximum slope in fit function)
    diff2_gompertz = sym.diff(d + (a - d) * 2.0**(-sym.exp((-b) * (x_symbol_gompertz - c))), x_symbol_gompertz, 2)

    # find root of second derivative
    roots_gompertz = sym.solveset(diff2_gompertz, x_symbol_gompertz)
    # print("Roots of diff2_gompertz: ", roots_gompertz)

    # calculate maximum slope at root of second derivative
    max_slope_gompertz = diff_gompertz.evalf(subs={x_symbol_gompertz: roots_gompertz.args[0]})
    return float(max_slope_gompertz)


def getRMSE(fit_type, frame_list, data_list, last_index_fit, popt):
    """
    - calculates root-mean-square error (RMSE) of fit function vs. data used for fitting
    :param fit_type: type of function used for fitting
    :param frame_list: list of frames with available data points
    :param data_list: list of available data points
    :param last_index_fit: index of last data point used for fitting
    :param popt: parameters of fit function
    :return: root-mean-square error (RMSE) of fit function
    """
    error_sum = 0
    for i in range(0, last_index_fit):
        if fit_type == 'sigmoid':
            error_sum += pow(abs(data_list[i] - sigmoid(frame_list[i], *popt)), 2)
        elif fit_type == 'sig_offset':
            error_sum += pow(abs(data_list[i] - sigmoid_offset(frame_list[i], *popt)), 2)
        elif fit_type == 'glf':
            error_sum += pow(abs(data_list[i] - generalized_logistic_function(frame_list[i], *popt)), 2)
        elif fit_type == 'gompertz':
            error_sum += pow(abs(data_list[i] - gompertz(frame_list[i], *popt)), 2)
        else:
            error_sum = 0
    error_sum /= last_index_fit
    rmse = math.sqrt(error_sum)
    return rmse


def getMAE(fit_type, frame_list, data_list, last_index_fit, popt):
    """
    - calculates mean absolute error (MAE) of fit function vs. data used for fitting
    :param fit_type: type of function used for fitting
    :param frame_list: list of frames with available data points
    :param data_list: list of available data points
    :param last_index_fit: index of last data point used for fitting
    :param popt: parameters of fit function
    :return: mean absolute error (MAE) of fit function
    """
    mae = 0
    for i in range(0, last_index_fit):
        if fit_type == 'sigmoid':
            mae += abs(data_list[i] - sigmoid(frame_list[i], *popt))
        elif fit_type == 'sig_offset':
            mae += abs(data_list[i] - sigmoid_offset(frame_list[i], *popt))
        elif fit_type == 'glf':
            mae += abs(data_list[i] - generalized_logistic_function(frame_list[i], *popt))
        elif fit_type == 'gompertz':
            mae += abs(data_list[i] - gompertz(frame_list[i], *popt))
        else:
            mae = 0
    mae /= last_index_fit
    return mae
