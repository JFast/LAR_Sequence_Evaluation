# LAR_Sequence_Evaluation
This repository contains source code for the computational analysis of endoscopic high-speed sequences showing the laryngeal adductor reflex (LAR) response after droplet-based stimulation.

THIS REPOSITORY IS PROVIDED PRIOR TO ACCEPTANCE OF THE ASSOCIATED MANUSCRIPT.

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
     FastOltmannIEEE.2021, 
     author={Jacob F. Fast and Andra Oltmann and Svenja Spindeldreier and Martin Ptok}, 
     year={2021},
     title={Code repository associated with the contribution "Computational Analysis of the Droplet-Stimulated Laryngeal Adductor Reflex in High-Speed Sequences"}, 
     DOI={},
     publisher={GitHub}
     }
```

# Description of Source Files

## Auxiliary Source Files

### `Parameters.py`

Contains preset parameter values for background model and stimulation droplet detection.

### `Preprocessing.py`

Contains auxiliary functions for frame preprocessing: contrast enhancement, channel/grayscale conversion, data type conversion, filtering, 

### `EndoCam_Calibration.py`

Performs standard calibration of the MIT-LAR laryngoscope using a set of images of an asymmetrical circle grid. Uses identified distortion coefficients for undistortion of sample image. Returns result.

### `Fitting.py`

Contains functions for the analytical approximation of data points representing the temporal evolution of the glottal area and the vocal fold edge angle and distance using different model functions. Contains auxiliary functions for identification of relevant points (intersections, inflections, etc.), conversion between representations, iterative identification of linear segments of MIT-LAR stimulation droplet trajectories, calculation of derivatives, estimation of characteristical angular velocities, and RMSE/MAE calculation.

### `DisplayVideo.py`

Contains auxiliary functions for window handling, frame sequence input/output, and visualization of (intermediate) results of algorithm for automatic extraction of glottal parameters.

### `Plots.py`

Contains auxiliary functions for result visualization using diagrams.

## Main Algorithms

### `LAR_Stimulation_Detection.py`

Performs automatic localization and tracking of MIT-LAR stimulation droplet after background subtraction. Differentiates impact/rebound events. Searches for additional droplets. Returns results (frame sequences, text file containing result summary, image files).

### `LAR_Onset_Detection.py`

Performs automatic detection of glottal reference point and contour and attempts automatic correction of glottal midline orientation (strictly vertical orientation desired). Estimates temporal evolution of glottal area and vocal fold edge angle and distance over frame sequence. Performs analytical modeling of identified time courses using different fit functions. Returns results (frame sequences, text file containing result summary, plots, image files).

