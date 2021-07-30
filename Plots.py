from matplotlib import pyplot as plt
import numpy as np
import Fitting as fit


def plotTotalIntensity(totalIntensity, xlabel, saving_path):
    """
    - plots total intensity variation (reference point identification)
    :param totalIntensity: vector of total intensity variation
    :return: void
    """
    plt.plot(range(0, totalIntensity.shape[1]), totalIntensity[0, :]/(np.max(totalIntensity[0, :])))
    plt.xlabel(xlabel)
    plt.ylabel("total intensity variation (normalized)")
    # plt.title("Totale Intensitäsunterschiede")
    plt.xticks(range(0, totalIntensity.shape[1], 20))
    plt.axis([0, totalIntensity.shape[1], 0, 1.1])
    plt.grid(True)
    plt.savefig(saving_path)
    plt.close()


def plotSigmoidFit(frames, signal, x, y, yLabel, title):
    """
    - creates general plot of sigmoid fit function
    :param frames: frame indices
    :param signal: signal (data points)
    :param x: x values of sigmoid fit function
    :param y: y values of sigmoid fit function
    :param yLabel: y-axis label
    :param title: plot title
    :return: void
    """
    plt.plot(frames, signal, 'x', color ='#00589b', label="data")
    plt.plot(x, y, '-', color='#e77b29', label="sigmoid fit")
    plt.axis([0, x[len(x) - 1] + 50, 0, np.max(signal) + 10])
    plt.xlabel('frame index')
    plt.ylabel(yLabel)
    plt.title(title)
    plt.legend(loc=1)
    plt.grid(True)
    plt.show()
    plt.close()


def plotSigmoidFitAngle(frames, signal, x, y, saving_path):
    """
    - plots sigmoidal fit of glottal angle
    :param frames: frame indices
    :param signal: signal (data points)
    :param x: x values of sigmoid fit function
    :param y: y values of sigmoid fit function
    :param saving_path: path to output file
    :return: void
    """
    plt.plot(frames, signal, '+', color ='#00589b', label="data")
    plt.plot(x, y, '-', color='#e77b29', label="fit")
    plt.axis([0, x[len(x) - 1] + 50, 0, np.max(signal) + 5])
    plt.xlabel('frame index')
    plt.ylabel("glottal angle in degree")
    # plt.title("symm. sigmoidal fit of glottal angle (no vert. offset)")
    plt.legend(loc=1, framealpha=1.0)
    plt.grid(True)
    plt.savefig(saving_path)
    plt.show()
    plt.close()


def plotSigmoidFitAngleOffset(frames, signal, x, y, saving_path):
    """
    - plots sigmoidal fit of glottal angle with vertical offset
    :param frames: frame indices
    :param signal: signal (data points)
    :param x: x values of sigmoid fit function
    :param y: y values of sigmoid fit function
    :param saving_path: path to output file
    :return: void
    """
    plt.plot(frames, signal, '+', color='#00589b', label="data")
    plt.plot(x, y, '-', color='#e77b29', label="fit")
    plt.axis([0, x[len(x) - 1] + 50, 0, np.max(signal) + 5])
    plt.xlabel('frame index')
    plt.ylabel("glottal angle in degree")
    # plt.title("symm. sigmoidal fit of glottal angle (vert. offset)")
    plt.legend(loc=1, framealpha=1.0)
    plt.grid(True)
    plt.savefig(saving_path)
    plt.show()
    plt.close()


def plotGLFFitAngle(frames, signal, x, y, saving_path):
    """
    - plots generalized logistic function fit of glottal angle
    :param frames: frame indices
    :param signal: signal (data points)
    :param x: x values of sigmoid fit function
    :param y: y values of sigmoid fit function
    :param saving_path: path to output file
    :return: void
    """
    plt.plot(frames, signal, '+', color='#00589b', label="data")
    plt.plot(x, y, '-', color='#e77b29', label="fit")
    plt.axis([0, x[len(x) - 1] + 50, 0, np.max(signal) + 5])
    plt.xlabel('frame index')
    plt.ylabel("glottal angle in degree")
    # plt.title("generalized logistic function fit of glottal angle")
    plt.legend(loc=1, framealpha=1.0)
    plt.grid(True)
    plt.savefig(saving_path)
    plt.show()
    plt.close()


def plotGompertzFitAngle(frames, signal, x, y, saving_path):
    """
    - plots Gompertz-like fit of glottal angle
    :param frames: frame indices
    :param signal: signal (data points)
    :param x: x values of sigmoid fit function
    :param y: y values of sigmoid fit function
    :param saving_path: path to output file
    :return: void
    """
    plt.plot(frames, signal, '+', color='#00589b', label="data")
    plt.plot(x, y, '-', color='#e77b29', label="fit")
    plt.axis([0, x[len(x) - 1] + 50, 0, np.max(signal) + 5])
    plt.xlabel('frame index')
    plt.ylabel("glottal angle in degree")
    # plt.title("Gompertz-like fit of glottal angle")
    plt.legend(loc=1, framealpha=1.0)
    plt.grid(True)
    plt.savefig(saving_path)
    plt.show()
    plt.close()


def plotCubicFitAngle(frames, signal, x, y, saving_path):
    """
    - plots cubic fit of glottal angle
    :param frames: frame indices
    :param signal: signal (data points)
    :param x: x values of cubic fit function
    :param y: y values of cubic fit function
    :param saving_path: path to output file
    :return: void
    """
    plt.plot(frames, signal, '+', color ='#00589b', label="data")
    plt.plot(x, y, '-', color='#e77b29', label="fit")
    plt.axis([0, x[len(x) - 1] + 50, 0, np.max(signal) + 5])
    plt.xlabel('frame index')
    plt.ylabel("glottal angle in degree")
    # plt.title("cubic fit of glottal angle")
    plt.legend(loc=1, framealpha=1.0)
    plt.grid(True)
    plt.savefig(saving_path)
    plt.show()
    plt.close()


def plotTwoFitsAngle(frames, signal, x1, y1, x2, y2, saving_path):
    """
    - plots two fit functions of glottal angle in one set of axes
    :param frames: frame indices
    :param signal: signal (data points)
    :param x1: x values of fit function 1
    :param y1: y values of fit function 1
    :param x2: x values of fit function 2
    :param y2: y values of fit function 2
    :param saving_path: path to output file
    :return: void
    """
    # plot data points
    plt.plot(frames, signal, '+', color ='#00589b', label="data")
    # plot fit function 1
    plt.plot(x1, y1, '-', color='tab:red', label="fit 1")
    plt.plot(x2, y2, '-', color='tab:blue', label="fit 2")
    plt.axis([0, x1[len(x1) - 1] + 50, 0, np.max(signal) + 5])
    plt.xlabel('frame index')
    plt.ylabel("glottal angle in degree")
    # plt.title("fits of glottal angle")
    plt.legend(loc=1, framealpha=1.0)
    plt.grid(True)
    plt.savefig(saving_path)
    plt.show()
    plt.close()


def plotSigmoidFitDistance(frames, signal, x, y, saving_path):
    """
    - plots sigmoidal fit of distance between vocal fold edges
    :param frames: frame indices
    :param signal: signal (data points)
    :param x: x values of sigmoid fit function
    :param y: y values of sigmoid fit function
    :param saving_path: path to output file
    :return: void
    """
    plt.plot(frames, signal, '+', color ='#00589b', label="data")
    plt.plot(x, y, '-', color='#e77b29', label="fit")
    plt.axis([0, x[len(x) - 1] + 50, 0, np.max(signal) + 5])
    plt.xlabel('frame index')
    plt.ylabel("vocal fold edge distance in pixel")
    # plt.title("symm. sigmoidal fit of vocal fold edge distance (no vert. offset)")
    plt.legend(loc=1, framealpha=1.0)
    plt.grid(True)
    plt.savefig(saving_path)
    plt.show()
    plt.close()


def plotSigmoidFitDistanceOffset(frames, signal, x, y, saving_path):
    """
    - plots sigmoidal fit of distance between vocal fold edges
    :param frames: frame indices
    :param signal: signal (data points)
    :param x: x values of sigmoid fit function
    :param y: y values of sigmoid fit function
    :param saving_path: path to output file
    :return: void
    """
    plt.plot(frames, signal, '+', color='#00589b', label="data")
    plt.plot(x, y, '-', color='#e77b29', label="fit")
    plt.axis([0, x[len(x) - 1] + 50, 0, np.max(signal) + 5])
    plt.xlabel('frame index')
    plt.ylabel("vocal fold edge distance in pixel")
    # plt.title("symm. sigmoidal fit of vocal fold edge distance (vert. offset)")
    plt.legend(loc=1, framealpha=1.0)
    plt.grid(True)
    plt.savefig(saving_path)
    plt.show()
    plt.close()


def plotGLFFitDistance(frames, signal, x, y, saving_path):
    """
    - plots generalized logistic function fit of distance between vocal fold edges
    :param frames: frame indices
    :param signal: signal (data points)
    :param x: x values of sigmoid fit function
    :param y: y values of sigmoid fit function
    :param saving_path: path to output file
    :return: void
    """
    plt.plot(frames, signal, '+', color='#00589b', label="data")
    plt.plot(x, y, '-', color='#e77b29', label="fit")
    plt.axis([0, x[len(x) - 1] + 50, 0, np.max(signal) + 5])
    plt.xlabel('frame index')
    plt.ylabel("vocal fold edge distance in pixel")
    # plt.title("generalized logistic function fit of vocal fold edge distance")
    plt.legend(loc=1, framealpha=1.0)
    plt.grid(True)
    plt.savefig(saving_path)
    plt.show()
    plt.close()


def plotGompertzFitDistance(frames, signal, x, y, saving_path):
    """
    - plots generalized logistic function fit of distance between vocal fold edges
    :param frames: frame indices
    :param signal: signal (data points)
    :param x: x values of sigmoid fit function
    :param y: y values of sigmoid fit function
    :param saving_path: path to output file
    :return: void
    """
    plt.plot(frames, signal, '+', color='#00589b', label="data")
    plt.plot(x, y, '-', color='#e77b29', label="fit")
    plt.axis([0, x[len(x) - 1] + 50, 0, np.max(signal) + 5])
    plt.xlabel('frame index')
    plt.ylabel("vocal fold edge distance in pixel")
    # plt.title("Gompertz-like fit of vocal fold edge distance")
    plt.legend(loc=1, framealpha=1.0)
    plt.grid(True)
    plt.savefig(saving_path)
    plt.show()
    plt.close()


def plotCubicFitDistance(frames, signal, x, y, saving_path):
    """
    - plots cubic fit of vocal fold edge distance
    :param frames: frame indices
    :param signal: signal (data points)
    :param x: x values of cubic fit function
    :param y: y values of cubic fit function
    :param saving_path: path to output file
    :return: void
    """
    plt.plot(frames, signal, '+', color='#00589b', label="data")
    plt.plot(x, y, '-', color='#e77b29', label="fit")
    plt.axis([0, x[len(x) - 1] + 50, 0, np.max(signal) + 5])
    plt.xlabel('frame index')
    plt.ylabel("vocal fold edge distance in pixel")
    # plt.title("cubic fit of of vocal fold edge distance")
    plt.legend(loc=1, framealpha=1.0)
    plt.grid(True)
    plt.savefig(saving_path)
    plt.show()
    plt.close()


def plotArea(frames, signal, saving_path):
    """
    - plots data points representing glottal area
    :param frames: frame indices
    :param signal: signal (data points)
    :param saving_path: path to output file
    :return: void
    """
    plt.plot(frames, signal, '+', color ='#00589b', label="data")
    plt.axis([0, frames[len(frames) - 1], 0, np.max(signal) + 0.2])
    plt.xlabel('frame index')
    plt.ylabel("glottal surface area in percent of total frame area (reverse analysis)")
    # plt.title("glottal surface area (reverse analysis)")
    plt.legend(loc=1, framealpha=1.0)
    plt.grid(True)
    plt.savefig(saving_path)
    plt.show()
    plt.close()


def plotSigmoidFitArea(frames, signal, x, y, saving_path):
    """
    - plots symm. sigmoidal fit of glottal area without vertical offset
    :param frames: frame indices
    :param signal: signal (data points)
    :param x: x values of sigmoid fit function
    :param y: y values of sigmoid fit function
    :param saving_path: path to output file
    :return: void
    """
    plt.plot(frames, signal, '+', color ='#00589b', label="data")
    plt.plot(x, y, '-', color='#e77b29', label="fit")
    plt.axis([0, x[len(x) - 1] + 50, 0, np.max(signal) + 0.2])
    plt.xlabel('frame index')
    plt.ylabel("glottal surface area in percent of total frame area")
    # plt.title("symm. sigmoidal fit of glottal surface area (no vert. offset)")
    plt.legend(loc=1, framealpha=1.0)
    plt.grid(True)
    plt.savefig(saving_path)
    plt.show()
    plt.close()


def plotSigmoidFitAreaOffset(frames, signal, x, y, saving_path):
    """
    - plots symm. sigmoidal fit of glottal area with vertical offset
    :param frames: frame indices
    :param signal: signal (data points)
    :param x: x values of sigmoid fit function
    :param y: y values of sigmoid fit function
    :param saving_path: path to output file
    :return: void
    """
    plt.plot(frames, signal, '+', color ='#00589b', label="data")
    plt.plot(x, y, '-', color='#e77b29', label="fit")
    plt.axis([0, x[len(x) - 1] + 50, 0, np.max(signal) + 0.2])
    plt.xlabel('frame index')
    plt.ylabel("glottal surface area in percent of total frame area")
    # plt.title("symm. sigmoidal fit of glottal surface area (vert. offset)")
    plt.legend(loc=1, framealpha=1.0)
    plt.grid(True)
    plt.savefig(saving_path)
    plt.show()
    plt.close()

def plotGLFFitArea(frames, signal, x, y, saving_path):
    """
    - plots generalized logistic fit function of glottal area without vertical offset
    :param frames: frame indices
    :param signal: signal (data points)
    :param x: x values of generalized logistic fit function
    :param y: y values of generalized logistic fit function
    :param saving_path: path to output file
    :return: void
    """
    plt.plot(frames, signal, '+', color ='#00589b', label="data")
    plt.plot(x, y, '-', color='#e77b29', label="fit")
    plt.axis([0, x[len(x) - 1] + 50, 0, np.max(signal) + 0.2])
    plt.xlabel('frame index')
    plt.ylabel("glottal surface area in percent of total frame area")
    # plt.title("generalized logistic function fit of glottal surface area")
    plt.legend(loc=1, framealpha=1.0)
    plt.grid(True)
    plt.savefig(saving_path)
    plt.show()
    plt.close()


def plotGompertzFitArea(frames, signal, x, y, saving_path):
    """
    - plots Gompertz-like fit of glottal area
    :param frames: frame indices
    :param signal: signal (data points)
    :param x: x values of Gompertz-like fit function
    :param y: y values of Gompertz-like fit function
    :param saving_path: path to output file
    :return: void
    """
    plt.plot(frames, signal, '+', color ='#00589b', label="data")
    plt.plot(x, y, '-', color='#e77b29', label="fit")
    plt.axis([0, x[len(x) - 1] + 50, 0, np.max(signal) + 0.2])
    plt.xlabel('frame index')
    plt.ylabel("glottal surface area in percent of total frame area")
    # plt.title("Gompertz-like fit of glottal surface area")
    plt.legend(loc=1, framealpha=1.0)
    plt.grid(True)
    plt.savefig(saving_path)
    plt.show()
    plt.close()


def plotCubicFitArea(frames, signal, x, y, saving_path):
    """
    - plots cubic fit of glottal area
    :param frames: frame indices
    :param signal: signal (data points)
    :param x: x values of cubic fit function
    :param y: y values of cubic fit function
    :param saving_path: path to output file
    :return: void
    """
    plt.plot(frames, signal, '+', color='#00589b', label="data")
    plt.plot(x, y, '-', color='#e77b29', label="fit")
    plt.axis([0, x[len(x) - 1] + 50, 0, np.max(signal) + 0.2])
    plt.xlabel('frame index')
    plt.ylabel("glottal surface area in percent of total frame area")
    # plt.title("cubic fit of of of glottal surface area")
    plt.legend(loc=1, framealpha=1.0)
    plt.grid(True)
    plt.savefig(saving_path)
    plt.show()
    plt.close()


def linearFitAngle(frames, signal, lines_lar_begin, impact, saving_path):
    """
    - plots linear fit of glottal angle
    :param frames: frame indices
    :param signal: signal (data points)
    :param lines_lar_begin: linear fit functions
    :param impact: coordinates of instant of droplet impact
    :param saving_path: path to output file
    :return: void
    """
    plt.plot(frames, signal, '+', color ='#00589b', label="data")
    frame_to_stop = fit.getStopFrame(frames, signal)
    plt.plot((0, frames[frame_to_stop] + 100), (0 * lines_lar_begin[2] + lines_lar_begin[3],(frames[frame_to_stop] + 100) * lines_lar_begin[2] + lines_lar_begin[3]), '--', color='#e77b29', label="linear fit")
    plt.plot((0, frames[frame_to_stop] + 100), (0 * lines_lar_begin[0] + lines_lar_begin[1], (frames[frame_to_stop] + 100) * lines_lar_begin[0] + lines_lar_begin[1]), '--', color='#e77b29', label="linear fit")
    plt.plot((impact[0]), (impact[1]), 'k', marker='X', label="intersection")
    plt.grid(True)
    plt.title("linear fit of glottal angle")
    plt.xlabel("frame index")
    plt.ylabel("glottal angle in degree")
    plt.axis([0, frames[frame_to_stop] + 100, 0, impact[1] + 5])
    plt.legend(loc=3, framealpha=1.0)
    plt.savefig(saving_path)
    plt.show()
    plt.close()


def linearFitArea(frames, signal, lines_lar_begin, impact, saving_path):
    """
    - plots linear fit of glottal area
    :param frames: frame indices
    :param signal: signal (data points)
    :param lines_lar_begin: linear fit functions
    :param impact: coordinates of instant of droplet impact
    :param saving_path: path to output file
    :return: void
    """
    plt.plot(frames, signal, '+', color ='#00589b', label="data")
    frame_to_stop = fit.getStopFrame(frames, signal)
    plt.plot((0, frames[frame_to_stop] + 100), (0 * lines_lar_begin[2] + lines_lar_begin[3], (frames[frame_to_stop] + 100) * lines_lar_begin[2] + lines_lar_begin[3]), '--', color='#e77b29', label="linear fit")
    plt.plot((0, frames[frame_to_stop] + 100), (0 * lines_lar_begin[0] + lines_lar_begin[1], (frames[frame_to_stop] + 100) * lines_lar_begin[0] + lines_lar_begin[1]), '--', color='#e77b29', label="linear fit")
    plt.plot((impact[0]), (impact[1]), 'k', marker='X', label="intersection")
    plt.grid(True)
    plt.title("linear fit of glottal surface area")
    plt.xlabel("frame index")
    plt.ylabel("glottal surface area in percent")
    plt.axis([0, frames[frame_to_stop] + 100, 0, impact[1] + 2])
    plt.legend(loc=3, framealpha=1.0)
    plt.savefig(saving_path)
    plt.show()
    plt.close()


def linearFitDistance(frames, signal, lines_lar_begin, impact, saving_path):
    """
    - plots linear fit of vocal fold edge distance
    :param frames: frame indices
    :param signal: signal (data points)
    :param lines_lar_begin: linear fit functions
    :param impact: coordinates of instant of droplet impact
    :param saving_path: path to output file
    :return: void
    """
    plt.plot(frames, signal, '+', color ='#00589b', label="data")
    frame_to_stop = fit.getStopFrame(frames, signal)
    plt.plot((0, frames[frame_to_stop] + 100), (0 * lines_lar_begin[2] + lines_lar_begin[3], (frames[frame_to_stop] + 100) * lines_lar_begin[2] + lines_lar_begin[3]), '--', color='#e77b29', label="linear fit")
    plt.plot((0, frames[frame_to_stop] + 100), (0 * lines_lar_begin[0] + lines_lar_begin[1], (frames[frame_to_stop] + 100) * lines_lar_begin[0] + lines_lar_begin[1]), '--', color='#e77b29', label="linear fit")
    plt.plot((impact[0]), (impact[1]), 'k', marker='X', label="intersection")
    plt.grid(True)
    # plt.title("linear fit of vocal fold edge distance")
    plt.xlabel("frame index")
    plt.ylabel("vocal fold edge distance in pixel")
    plt.axis([0, frames[frame_to_stop] + 100, 0, impact[1] + 5])
    plt.legend(loc=3, framealpha=1.0)
    plt.savefig(saving_path)
    plt.show()
    plt.close()
