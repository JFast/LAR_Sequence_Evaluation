# LAR_Sequence_Evaluation
This repository contains source code for the [computational analysis of endoscopic high-speed sequences showing the laryngeal adductor reflex (LAR) response after droplet-based stimulation.](https://www.doi.org/10.1002/lary.30041)

# Table of Contents
* [Requirements](#requirements)
* [How to Cite this Repository](#how-to-cite-this-repository)
* [Descriptions of Source Files](#descriptions-of-source-files)
* [Notes](#notes)

# Requirements

Required special packages/libraries: *OpenCV*, *NumPy*, *SciPy*, *SymPy*, *Matplotlib*, *Openpyxl*.

Source code developed in *Python* (version 3.8.8) using the *PyCharm Professional* IDE (version 2021.1, JetBrains s.r.o., Prague, Czech Republic).

Versions of packages/libraries used in this study: *opencv-python 4.5.2.52*, *numpy 1.20.2*, *scipy 1.6.2*, *sympy 1.8*, *matplotlib 3.4.1*, *openpyxl 3.0.7*.

All code executable on a standard desktop PC (no GPU required).

# How to Cite this Repository

```BibTeX
@misc{
     FastOltmann.2022, 
     author={Jacob F. Fast and Andra Oltmann and Svenja Spindeldreier and Martin Ptok}, 
     year={2022},
     title={Code repository associated with the contribution "Computational Analysis of the Droplet-Stimulated Laryngeal Adductor Reflex in High-Speed Sequences", DOI: 10.1002/lary.30041}
     publisher={GitHub}
     }
```

# Description of Source Files

## Main Algorithms

### `LAR_Stimulation_Detection.py`

Performs automatic localization and tracking of MIT-LAR stimulation droplet after background subtraction. Differentiates impact/rebound events. Searches for additional droplets. Returns results (frame sequences, text file containing result summary, image files).

### `LAR_Onset_Detection.py`

Performs automatic detection of glottal reference point and contour and attempts automatic correction of glottal midline orientation (strictly vertical orientation desired). Estimates temporal evolution of glottal area, vocal fold edge angle, and vocal fold edge distance over the course of the provided frame sequence. Performs analytical modeling of identified time courses using different fit functions. Returns results (frame sequences, text file containing result summary, plots, image files).

## Auxiliary Source Files

### `EndoCam_Calibration.py`

Performs standard calibration of the MIT-LAR laryngoscope using a set of images of an asymmetrical circle grid. Uses identified distortion coefficients for undistortion of sample image. Returns result.

### `Parameters.py`

Contains preset parameter values for background model and MIT-LAR stimulation droplet detection.

### `Preprocessing.py`

Contains auxiliary functions for frame preprocessing: contrast enhancement, channel/grayscale conversion, data type conversion, and filtering.

### `Trajectory.py`

Contains auxiliary functions for MIT-LAR stimulation droplet detection and tracking, trajectory estimation and linear orthogonal distance regression, and rebound/impact distinction.

### `ReferencePoint.py`

Contains auxiliary functions required for the execution of a method proposed by Andrade-Miranda and Godino-Llorente ("ROI detection in high speed laryngeal images", 2014 IEEE 11th International Symposium on Biomedical Imaging, 2014, pp. 477???480, DOI: [10.1109/ISBI.2014.6867912](https://doi.org/10.1109/ISBI.2014.6867912)), which provides a glottal reference point to guide the subsequent glottis segmentation procedure.

### `Segmentation.py`

Contains auxiliary functions for the automatic estimation of the glottal contour (label/grid/seed point creation, watershed segmentation, region growing, edge detection, glottal midline orientation detection, etc.).

### `UserInteraction.py`

Contains auxiliary functions for user interaction during a potential correction of the initial glottis segmentation by a combination of watershed/region growing procedures.

### `VocalFolds.py`

Contains auxiliary functions for convex hull identification, landmark localization, and angle/distance/slope calculation, as required for the estimation of glottal parameters.

### `Fitting.py`

Contains functions for the analytical approximation of data points representing the temporal evolution of the glottal area and the vocal fold edge angle and distance using different model functions. Contains auxiliary functions for identification of relevant points (intersections, inflections, etc.), conversion between representations, iterative identification of linear segments of MIT-LAR stimulation droplet trajectories, calculation of derivatives, estimation of characteristical angular velocities, and RMSE/MAE calculation.

### `DisplayVideo.py`

Contains auxiliary functions for window handling, frame sequence input/output, and visualization of (intermediate) results of algorithm for automatic extraction of glottal parameters.

### `Plots.py`

Contains auxiliary functions for result visualization using diagrams.

# Notes

Glottal angle/glottal area/vocal fold edge distance returned in degrees/percent of total frame area/pixel, respectively.

All provided files are extensively commented to facilitate code comprehension.
