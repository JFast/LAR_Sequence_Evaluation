import cv2
import math
import numpy as np
import Parameters as params
import Preprocessing as prepro
import Trajectory as traj


# initialization of background subtraction
background = cv2.createBackgroundSubtractorMOG2(history=params.BGSM_HISTORY, detectShadows=False)
background.setBackgroundRatio(params.BGSM_BACKGROUND_RATIO)

# initialization of blob detection
params_blob = cv2.SimpleBlobDetector_Params()
# filtering according to blob size
params_blob.filterByArea = True
params_blob.minArea = params.MIN_AREA_BLOB
# filtering according to blob circularity
params_blob.filterByCircularity = True
params_blob.minCircularity = params.MIN_CIRCULARITY
# filtering according to blob convexity
params_blob.filterByConvexity = True
params_blob.minConvexity = params.MIN_CONVEXITY
# filtering according to ratio of blob axes
params_blob.filterByInertia = True
params_blob.minInertiaRatio = params.MIN_INERTIA_RATIO
detector = cv2.SimpleBlobDetector_create(params_blob)

# PATH DEFINITION
pat = "03"
sequence_number = "14"
# use avi file
video_path = r"F:/LARvideos/videos_annotated/pat_" + pat + "\es_01_pat_" + pat + "_seq_" + sequence_number + \
             "\es_01_pat_" + pat + "_seq_" + sequence_number + ".avi"
# use mp4 file
# video_path = r"F:/LARvideos/videos_annotated/pat_" + pat + "\es_01_pat_" + pat + "_seq_" + sequence_number +
# "\es_01_pat_" + pat + "_seq_" + sequence_number + ".mp4"

# load frame sequence
video = cv2.VideoCapture(video_path)

# FILE FOR RESULT STORAGE
saving_path = r"F:/Masterarbeit_Andra_Oltmann/Results_TMI/LAR_Stimulation_Detection/"
file = open(saving_path + pat + "_" + sequence_number + "_Evaluation.txt", "w")
file.write("EVALUATION\n")
file.write("Sequence identifier: es_01_pat_" + pat + "_seq_" + sequence_number + "\n\n")

# instantiate VideoWriter objects
# frame rate 60 fps to be congruent to output of LAR evaluation script (15 fps, every fourth frame retained)
output_foreground = cv2.VideoWriter(saving_path + pat + "_" + sequence_number + '_foreground.mp4',
                                    cv2.VideoWriter_fourcc(*"mp4v"), 60.0, (512, 512))
output_fusion = cv2.VideoWriter(saving_path + pat + "_" + sequence_number + '_foreground_fusion.mp4',
                                    cv2.VideoWriter_fourcc(*"mp4v"), 60.0, (512, 512))
# frame rate 15 fps for easy assessment of detection result
output_all = cv2.VideoWriter(saving_path + pat + "_" + sequence_number + '_all.mp4',
                                    cv2.VideoWriter_fourcc(*"mp4v"), 15.0, (512, 512))

# INITIAL DETECTION OF DROPLETS (FIRST LOOP OVER SEQUENCE)
# initialize frame index
frame_number = 0
first_frame_detect = False
frame_result = None
first_frame = 0
# iterate over frames of sequence
while 1:
    # load frame
    ret, frame = video.read()
    # break loop if no frame available
    if not ret:
        print("break")
        break
    else:
        # increment and print frame index
        frame_number = frame_number + 1
        # print(frame_number)

        frame_mean = prepro.getPreForBackgroundSubtraction(frame)

        # apply background subtraction to obtain moving droplet
        fgmask = background.apply(frame_mean)

        # optimize foreground
        # apply morphological closing to obtain homogeneous, circular objects
        fgmask_opt = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, params.KERNEL_CIRCLE)

        droplets = []

        # if more than params.BGSM_HISTORY frames were assessed
        # (background construction was able to assess params.BGSM_HISTORY frames)
        if frame_number > params.BGSM_HISTORY:
            # apply OpenCV blob detection on inverted frame after background subtraction to find blobs in foreground
            droplets = detector.detect(cv2.bitwise_not(fgmask_opt))
            for droplet in droplets:
                # if no valid droplet detected yet
                if not first_frame_detect:
                    # only retain candidates with minimum radius MIN_DROPLET_RADIUS
                    if droplet.size / 2.0 > params.MIN_DROPLET_RADIUS:
                        frame_result = frame.copy()
                        # draw found blobs as circles
                        frame_result = cv2.circle(frame_result, (int(droplet.pt[0]), int(droplet.pt[1])),
                                                  int(droplet.size/2.0), [0, 0, 255], 1)
                        # resize frame
                        frame_result_large = cv2.resize(frame_result,
                                                        (int(2.0 * frame_result.shape[1]),
                                                         int(2.0 * frame_result.shape[0])),
                                                        interpolation=cv2.INTER_LINEAR)
                        # write detected droplets to output sequence (15 identical frames)
                        for i in range(0, 15):
                            output_all.write(frame_result_large)
                            i += 1

                        first_frame = frame_number
                        first_frame_detect = True
                        # update background
                        background_image = background.getBackgroundImage()

        # create fusion of frame and foreground objects
        frame_fusion = frame.copy()
        # frame_fusion[foreground_pixels] = [0, 255, 0]
        mask_fusion = np.zeros((256, 256, 3)).astype('uint8')
        foreground_pixels = np.where(fgmask_opt == 255)
        mask_fusion[foreground_pixels] = [0, 255, 0]
        frame_fusion = cv2.addWeighted(frame_fusion, 1.0, mask_fusion, 0.3, 0.0)

        # resize frames
        frame_fusion_large = cv2.resize(frame_fusion,
                                        (int(2.0 * frame_fusion.shape[1]), int(2.0 * frame_fusion.shape[0])),
                                        interpolation=cv2.INTER_LINEAR)
        frame_large = cv2.resize(frame, (int(2.0 * frame.shape[1]), int(2.0 * frame.shape[0])),
                                   interpolation=cv2.INTER_LINEAR)
        fgmask_opt_large = cv2.resize(fgmask_opt, (int(2.0 * frame.shape[1]), int(2.0 * frame.shape[0])),
                                   interpolation=cv2.INTER_LINEAR)

        # write foreground to output sequence
        fgmask_opt_bgr = cv2.cvtColor(fgmask_opt_large, cv2.COLOR_GRAY2BGR)
        output_foreground.write(fgmask_opt_bgr)
        # write fusion to output sequence
        output_fusion.write(frame_fusion_large)

        # show frame and foreground
        cv2.imshow("Frame", frame_large)
        cv2.imshow("Foreground", fgmask_opt_large)
        cv2.waitKey(1)

# close video output objects for foreground sequence and sequence showing fusion with original sequence
if output_foreground:
    output_foreground.release()
if output_fusion:
    output_fusion.release()

tracking = True

# if no droplet found: end of algorithm
if first_frame == 0:
    file.write("No droplet detected in sequence!\n")
    tracking = False

# DETECTION OF FURTHER DROPLET POSITIONS (SECOND LOOP OVER SEQUENCE)
if tracking:
    cv2.imwrite(saving_path + pat + "_" + sequence_number + "_Initial_Detection.png", frame_result)
    file.write("Frame index of first droplet detection: " + str(first_frame))

    # SAMPLING POINTS ON PRINCIPAL DROPLET TRAJECTORY
    # initialize background subtraction
    background = cv2.createBackgroundSubtractorMOG2(history=params.BGSM_HISTORY, detectShadows=False)
    background.setBackgroundRatio(params.BGSM_BACKGROUND_RATIO)

    # load frame sequence
    video = cv2.VideoCapture(video_path)
    # initialize frame index
    frame_number = 0
    # list for sampling points on principal droplet trajectory
    droplets_list = list()
    # iterate over all frames of sequence
    while 1:
        # load frame
        ret, frame = video.read()
        # break loop if not frame available
        if not ret:
            print("break")
            break
        else:
            # increment and print frame index
            frame_number = frame_number + 1
            # print(frame_number)

            frame_mean = prepro.getPreForBackgroundSubtraction(frame)

            # apply background subtraction to obtain moving droplet
            fgmask = background.apply(frame_mean)

            # optimize foreground
            # apply morphological closing to obtain homogeneous, circular objects
            fgmask_opt = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, params.KERNEL_CIRCLE)

            # initial droplet detection
            if frame_number == first_frame:
                # detect all droplets in inverted frame after background subtraction
                droplets = []
                droplets = detector.detect(cv2.bitwise_not(fgmask_opt))
                # retain all candidates with minimum radius params.MIN_DROPLET_RADIUS
                candidates = list()
                for droplet in droplets:
                    # if current candidate larger than pre-defined minimum droplet radius
                    if droplet.size / 2.0 > params.MIN_DROPLET_RADIUS:
                        candidates.append(droplet)
                # select largest candidate
                candidate_to_add = traj.getMaxCandidate(candidates)
                # add centroid coordinates to 'droplets_list'
                droplets_list.append([frame_number, (int(candidate_to_add.pt[0]), int(candidate_to_add.pt[1]))])

            # detect additional sampling points
            check_frame = False
            if frame_number > first_frame:
                # detection of second sampling point
                if len(droplets_list) == 1:
                    # evaluate temporal distance to last detected droplet
                    if abs(droplets_list[-1][0] - frame_number) > params.TIME_BETWEEN_BLOBS:
                        # stop search for droplet positions if temporal distance is above pre-defined threshold value
                        check_frame = True
                    else:
                        droplets = []
                        # blob detection on foreground mask
                        droplets = detector.detect(cv2.bitwise_not(fgmask_opt))
                        print(droplets)
                        candidates = list()
                        for droplet in droplets:
                            last_droplet = droplets_list[-1][1]
                            # calculate Euclidean distance between current keypoint and last detected droplet
                            distance = traj.getDistanceBetweenDroplets(last_droplet, [droplet.pt[0], droplet.pt[1]])
                            # check if current keypoint is within circle around first droplet position
                            if distance < params.MAX_RADIUS_SECOND_DROPLET_MAIN:
                                candidates.append(droplet)
                        # if only one candidate found: add to list
                        if len(candidates) == 1:
                            droplets_list.append([frame_number, (int(candidates[0].pt[0]), int(candidates[0].pt[1]))])
                        # if more than one candidate found: add to list according to selection criteria
                        elif len(candidates) > 1:
                            last_droplet = droplets_list[-1][1]
                            # initialize min_distance with length of frame diagonal
                            min_distance = math.sqrt(pow(frame.shape[0], 2) + pow(frame.shape[1], 2))
                            # initialize 'candidate_to_add' with first candidate inside circular search area
                            candidate_to_add = candidates[0]
                            # go through all candidates and check distance to frame center
                            for candidate in candidates:
                                distance = traj.getDistanceBetweenDroplets([frame.shape[0]/2.0, frame.shape[1]/2.0],
                                                                           [candidate.pt[0], candidate.pt[1]])
                                if distance < min_distance:
                                    min_distance = distance
                                    # select candidate with lowest distance to frame center as candidate to add
                                    candidate_to_add = candidate
                            droplets_list.append([frame_number,
                                                  (int(candidate_to_add.pt[0]), int(candidate_to_add.pt[1]))])
                # sampling points 3 to 5
                # use findContours() instead of blob detector for droplet centroid detection
                elif 1 < len(droplets_list) <= 4:
                    # stop search for droplet positions if temporal distance is above pre-defined threshold value
                    if abs(droplets_list[-1][0] - frame_number) > params.TIME_BETWEEN_BLOBS:
                        check_frame = True
                    else:
                        # estimate distance of droplet centroid to last sampling point in current frame
                        distance = traj.getDistanceInNextFrame(droplets_list, frame_number)
                        # apply contour finding step
                        contours, hier = cv2.findContours(fgmask_opt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        candidates = list()
                        if len(contours) > 1:
                            for contour in contours:
                                if cv2.contourArea(contour) > 0:
                                    cx, cy = traj.getCentroidOfContour(contour)
                                    # calculate Euclidean distance between centroid of current contour
                                    # and last sampling point
                                    distance_point = traj.getDistanceBetweenDroplets(droplets_list[-1][1], [cx, cy])
                                    # if current contour centroid closer than search radius to last sampling point:
                                    # append centroid of current contour
                                    if distance_point <= distance + params.ADD_TO_SEARCH_RADIUS:
                                        candidates.append(contour)
                        # if only one contour detected
                        elif len(contours) == 1:
                            for contour in contours:
                                if cv2.contourArea(contour) > 0:
                                    M = cv2.moments(contour)
                                    # calculate centroid coordinates
                                    cx = int(M['m10'] / M['m00'])
                                    cy = int(M['m01'] / M['m00'])
                                    distance_point = traj.getDistanceBetweenDroplets(droplets_list[-1][1], [cx, cy])
                                    # append centroid of current contour if closer than 60 pixels to last sampling point
                                    if distance_point <= 60:
                                        candidates.append(contour)
                        # one candidate only (add always)
                        if len(candidates) == 1:
                            cx, cy = traj.getCentroidOfContour(candidates[0])
                            droplets_list.append([frame_number, (int(cx), int(cy))])
                        # several candidates (only add candidate closest to optimum point)
                        elif len(candidates) > 1:
                            best_point = traj.getBestPointInNextFrame(droplets_list, frame_number)
                            # initialize 'min_distance' with length of frame diagonal
                            min_distance = math.sqrt(pow(frame.shape[0], 2) + pow(frame.shape[1], 2))
                            # initialize 'candidate_to_add' with first candidate
                            candidate_to_add = candidates[0]
                            for candidate in candidates:
                                cx, cy = traj.getCentroidOfContour(candidate)
                                # calculate distance between predicted droplet position and current droplet centroid
                                distance = math.sqrt(pow(abs(best_point[0] - cx), 2) + pow(abs(best_point[1] - cy), 2))
                                if distance < min_distance:
                                    min_distance = distance
                                    candidate_to_add = candidate
                            cx, cy = traj.getCentroidOfContour(candidate_to_add)
                            droplets_list.append([frame_number, (int(cx), int(cy))])
                # sampling points 6 and onwards
                elif len(droplets_list) > 4:
                    # check if temporal distance is below threshold value
                    if abs(droplets_list[-1][0] - frame_number) > params.TIME_BETWEEN_BLOBS:
                        check_frame = True
                    else:
                        # estimate distance of droplet centroid to last sampling point in current frame
                        distance = traj.getDistanceInNextFrame(droplets_list, frame_number)
                        # calculate linear fit of 'droplets_list'
                        fit = traj.getFitDropletList(droplets_list)
                        # draw fit line
                        frame = cv2.line(frame, (0, int(0 * fit[0] + fit[1])),
                                         (frame.shape[1], int(frame.shape[1] * fit[0] + fit[1])), [0, 200, 0], 1)
                        # calculate "acceptance angle" of search space
                        acceptance_angle = traj.getAcceptanceAngle(droplets_list, frame_number)
                        # create search mask
                        mask_to_check = traj.getCylinderMask(frame, fit, distance, params.ADD_TO_SEARCH_RADIUS,
                                                             acceptance_angle, droplets_list[-1][1], droplets_list)
                        # draw search area
                        mask_search_area = np.zeros((256, 256, 3)).astype('uint8')
                        mask_search_area[mask_to_check == 255] = [0, 200, 0]
                        frame = cv2.addWeighted(frame, 1.0, mask_search_area, 0.3, 0.0)

                        # apply contour finding step
                        contours, hier = cv2.findContours(fgmask_opt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        candidates = list()
                        for contour in contours:
                            if cv2.contourArea(contour) > 0:
                                cx, cy = traj.getCentroidOfContour(contour)
                                mask = np.zeros((frame.shape[0], frame.shape[1])).astype('uint8')
                                mask[cy, cx] = 255
                                result = cv2.bitwise_and(mask, mask_to_check)
                                # if centroid of found contour inside search area: append contour to 'candidates'
                                if cv2.countNonZero(result) > 0:
                                    candidates.append(contour)
                        # one candidate only (add always)
                        if len(candidates) == 1:
                            cx, cy = traj.getCentroidOfContour(candidates[0])
                            droplets_list.append([frame_number, (int(cx), int(cy))])
                        # several candidates (only add candidate closest to optimum point)
                        elif len(candidates) > 1:
                            best_point = traj.getBestPointInNextFrame(droplets_list, frame_number)
                            # initialize 'min_distance' with length of frame diagonal
                            min_distance = math.sqrt(pow(frame.shape[0], 2) + pow(frame.shape[1], 2))
                            # initialize 'candidate_to_add' with first candidate
                            candidate_to_add = candidates[0]
                            for candidate in candidates:
                                cx, cy = traj.getCentroidOfContour(candidate)
                                # calculate distance between predicted droplet position and current droplet centroid
                                distance = math.sqrt(
                                    pow(abs(best_point[0] - cx), 2) + pow(abs(best_point[1] - cy), 2))
                                if distance < min_distance:
                                    min_distance = distance
                                    candidate_to_add = candidate
                            cx, cy = traj.getCentroidOfContour(candidate_to_add)
                            droplets_list.append([frame_number, (int(cx), int(cy))])

                if not check_frame:
                    for point in droplets_list:
                        droplet = point[1]
                        frame = cv2.circle(frame, (droplet[0], droplet[1]), 1, [0, 255, 0], -1)
                    frame_large = cv2.resize(frame, (int(2.0 * frame.shape[1]), int(2.0 * frame.shape[0])),
                                             interpolation=cv2.INTER_LINEAR)
                    fgmask_opt_large = cv2.resize(fgmask_opt, (int(2.0 * frame.shape[1]), int(2.0 * frame.shape[0])),
                                                  interpolation=cv2.INTER_LINEAR)
                    # write frame to output sequence
                    output_all.write(frame_large)
                    # show current frame
                    cv2.imshow("Frame", frame_large)
                    cv2.imshow("Foreground", fgmask_opt_large)
                    # remove '1' to wait for user confirmation in this case
                    cv2.waitKey(1)
                else:
                    for point in droplets_list:
                        droplet = point[1]
                        frame = cv2.circle(frame, (droplet[0], droplet[1]), 1, [0, 255, 0], -1)
                    frame_large = cv2.resize(frame, (int(2.0 * frame.shape[1]), int(2.0 * frame.shape[0])),
                                             interpolation=cv2.INTER_LINEAR)
                    fgmask_opt_large = cv2.resize(fgmask_opt, (int(2.0 * frame.shape[1]), int(2.0 * frame.shape[0])),
                                                  interpolation=cv2.INTER_LINEAR)
                    # show current frame
                    cv2.imshow("Frame", frame_large)
                    cv2.imshow("Foreground", fgmask_opt_large)
                    cv2.waitKey(1)

    # save result (include trajectory as frame overlay on original sequence)
    video = cv2.VideoCapture(video_path)
    ret, frame = video.read()
    frame_result = frame.copy()
    for point in droplets_list:
        droplet = point[1]
        frame = cv2.circle(frame, (droplet[0], droplet[1]), 1, [255, 255, 255], -1)
    frame_large = cv2.resize(frame, (int(2.0 * frame.shape[1]), int(2.0 * frame.shape[0])),
                             interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(saving_path + pat + "_" + sequence_number + "_Sampling_Points_Principal_Trajectory.png", frame_large)

    file.write("\n\nEVALUATION OF SAMPLING POINTS ON PRINCIPAL TRAJECTORY\n\n")
    file.write("List of sampling points on principal trajectory:\n")
    file.write(str(droplets_list))
    file.write("\n\n")

    tracking = True

    # if more than four sampling points available
    if len(droplets_list) > 3:
        # remove faulty detections at final section of droplet trajectory
        while droplets_list[-1][0] - droplets_list[-2][0] > 10 or droplets_list[-2][0] - droplets_list[-3][0] > 10:
            # remove last two elements from list
            droplets_list.remove(droplets_list[-1])
            droplets_list.remove(droplets_list[-1])

    # if not enough sampling points found
    if len(droplets_list) <= 14:
        file.write("Evaluation of principal trajectory not possible (not enough sampling points).\n")
        print("Tracking not possible!")
        tracking = False

    # IMPACT/REBOUND DISTINCTION PROCEDURE
    if tracking:
        points = list()
        frames = list()

        # avoid noisy sampling points by excluding first six points
        for i in range(6, len(droplets_list)):
            droplet = droplets_list[i]
            points.append(droplet[1])
            frames.append(droplet[0])
        # separate sampling points into two subsets, calculate acute angle between trajectory segments
        angle, list_first, list_second, impact_number = traj.iterative_impact(points, frames)

        # file.write("Angle of principal trajectory: " + str(angle) + "\n\n")
        file.write("Detected (acute) angle of trajectory segments: " + str(angle) + "\n\n")
        # if acute angle between identified linear trajectory segments inferior to 40 degrees: no rebound expected
        if angle < 40:
            # potential impact: take last frame index as impact frame
            frame_impact = droplets_list[-1][0]
            # store result for second impact/rebound identification loop
            angle_greater_40 = False
        # acute angle between identified linear trajectory segments greater than 40 degrees: rebound expected
        else:
            # potential rebound: take frame index identified by function iterative_impact() as impact frame
            frame_impact = impact_number
            # remove frames after rebound from 'droplets_list'
            droplets_list = traj.getMainTrajectoryUntilImpact(droplets_list, frame_impact)
            # store result for second impact/rebound identification loop
            angle_greater_40 = True
        file.write("Tentative frame index of droplet impact (principal trajectory): " + str(frame_impact) + "\n\n")

        # NEW SECTION
        print("Last frame index in 'droplets_list': ", droplets_list[-1][0])
        print("\n")
        print("'points' (first loop): ")
        print(points)
        print("\n")
        print("'list_first' (first loop): ")
        print(list_first)
        print("'list_second' (first loop): ")
        print(list_second)
        print("\n")
        print("'frame_impact' (first loop): ", impact_number)
        print("\n")

        # draw fit lines
        # get list of sampling points up to impact
        droplets_list_before_impact = traj.getMainTrajectoryUntilImpact(droplets_list, impact_number)
        # get linear fit for sampling points up to impact
        fit_before = traj.getFitDropletList(droplets_list_before_impact)
        # get first frame of sequence
        video = cv2.VideoCapture(video_path)
        ret, frame = video.read()
        # draw all sampling points before impact on frame
        for point in droplets_list_before_impact:
            droplet = point[1]
            frame = cv2.circle(frame, (droplet[0], droplet[1]), 1, [0, 255, 0], -1)
        # draw fit line for points before impact
        frame = cv2.line(frame, (0, int(0 * fit_before[0] + fit_before[1])),
                         (frame.shape[1], int(frame.shape[1] * fit_before[0] + fit_before[1])), [0, 200, 0], 1)

        # get list of sampling points after impact
        droplets_list_after_impact = []
        droplets_list_after_impact = droplets_list[len(droplets_list_before_impact):]
        if droplets_list_after_impact:
            # get linear fit for sampling points after impact
            fit_after = traj.getFitDropletList(droplets_list_after_impact)
            # draw all sampling points after impact on frame
            for point in droplets_list_after_impact:
                droplet = point[1]
                frame = cv2.circle(frame, (droplet[0], droplet[1]), 1, [0, 0, 255], -1)
            # draw fit line for points after impact
            frame = cv2.line(frame, (0, int(0 * fit_after[0] + fit_after[1])),
                             (frame.shape[1], int(frame.shape[1] * fit_after[0] + fit_after[1])), [0, 0, 200], 1)
        # resize frame
        frame_large = cv2.resize(frame, (int(2.0 * frame.shape[1]), int(2.0 * frame.shape[0])),
                             interpolation=cv2.INTER_LINEAR)
        # show result
        cv2.imshow("Impact/rebound trajectories (first loop)", frame_large)
        cv2.waitKey(2000)
        cv2.destroyWindow("Impact/rebound trajectories (first loop)")
        # save result
        cv2.imwrite(saving_path + pat + "_" + sequence_number + "_Impact_Rebound_First_Loop.png",
                    frame_large)
        # END OF NEW SECTION

        # print(droplets_list)

        # store image showing identified principal trajectory
        # read first frame of sequence
        video = cv2.VideoCapture(video_path)
        ret, frame = video.read()
        # add found droplets to first frame
        for point in droplets_list:
            droplet = point[1]
            frame = cv2.circle(frame, (droplet[0], droplet[1]), 1, [0, 255, 0], -1)
        frame_large = cv2.resize(frame, (int(2.0 * frame.shape[1]), int(2.0 * frame.shape[0])),
                             interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(saving_path + pat + "_" + sequence_number +
                    "_Sampling_Points_on_Principal_Trajectory_After_Post-Processing.png", frame_large)
        file.write("List of sampling points on principal trajectory after post-processing:\n")
        file.write(str(droplets_list))
        file.write("\n")

        # IDENTIFY SAMPLING POINTS ON DROPLET REBOUND TRAJECTORY (THIRD LOOP)
        # initialize background subtraction
        background = cv2.createBackgroundSubtractorMOG2(history=params.BGSM_HISTORY, detectShadows=False)
        background.setBackgroundRatio(params.BGSM_BACKGROUND_RATIO)
        # load sequence
        video = cv2.VideoCapture(video_path)
        # initialize frame index
        frame_number = 0
        # create list for sampling points on rebound trajectory
        rebound_list = list()
        # iterate over frames of sequence
        while 1:
            # load frame
            ret, frame = video.read()
            # break loop if no frame available
            if not ret:
                print("break")
                break
            else:
                # increment and print frame index
                frame_number = frame_number + 1
                # print(frame_number)

                # apply background subtraction to obtain moving foreground objects
                frame_mean = prepro.getPreForBackgroundSubtraction(frame)
                fgmask = background.apply(frame_mean)

                # optimize foreground
                # apply morphological closing to obtain homogeneous circular objects
                fgmask_opt = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, params.KERNEL_CIRCLE)

                # if current frame equal to frame showing last identified droplet on principal trajectory
                if frame_number == droplets_list[-1][0]:
                    # add first sampling point (= impact point)
                    rebound_list.append([frame_number, droplets_list[-1][1]])

                # identify additional sampling points after first droplet contact with laryngeal mucosa
                if frame_number > droplets_list[-1][0]:
                    # detection of second sampling point
                    if len(rebound_list) == 1:
                        # check if temporal distance is below threshold value
                        if abs(rebound_list[-1][0] - frame_number) > params.TIME_BETWEEN_BLOBS:
                            pass
                        else:
                            contours, hier = cv2.findContours(fgmask_opt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            candidates = list()
                            for contour in contours:
                                if cv2.contourArea(contour) > 0:
                                    cx, cy = traj.getCentroidOfContour(contour)
                                    distance = traj.getDistanceBetweenDroplets([rebound_list[-1][1][0],
                                                                                rebound_list[-1][1][1]], [cx, cy])
                                    if distance < params.MAX_RADIUS_SECOND_DROPLET_REBOUND:
                                        candidates.append(contour)
                            if len(candidates) == 1:
                                cx, cy = traj.getCentroidOfContour(candidates[0])
                                rebound_list.append([frame_number, (int(cx), int(cy))])
                            elif len(candidates) > 1:
                                last_droplet = rebound_list[-1][1]
                                min_distance = math.sqrt(pow(frame.shape[0], 2) + pow(frame.shape[1], 2))
                                candidate_to_add = candidates[0]
                                for candidate in candidates:
                                    cx, cy = traj.getCentroidOfContour(candidate)
                                    distance = traj.getDistanceBetweenDroplets([rebound_list[-1][1][0],
                                                                                rebound_list[-1][1][1]], [cx, cy])
                                    if distance < min_distance:
                                        min_distance = distance
                                        candidate_to_add = candidate
                                cx, cy = traj.getCentroidOfContour(candidate_to_add)
                                rebound_list.append([frame_number, (int(cx), int(cy))])
                    # sampling points 3 to 5
                    elif 1 < len(rebound_list) <= 4:
                        # check if temporal distance is below threshold value
                        if abs(rebound_list[-1][0] - frame_number) > params.TIME_BETWEEN_BLOBS:
                            pass
                        else:
                            print(rebound_list)
                            distance = traj.getDistanceInNextFrame(rebound_list, frame_number)
                            contours, hier = cv2.findContours(fgmask_opt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            candidates = list()
                            for contour in contours:
                                if cv2.contourArea(contour) > 0:
                                    cx, cy = traj.getCentroidOfContour(contour)
                                    distance_point = traj.getDistanceBetweenDroplets(rebound_list[-1][1], [cx, cy])
                                    if distance_point <= distance + params.ADD_TO_SEARCH_RADIUS:
                                        candidates.append(contour)
                            # one candidate only (add always)
                            if len(candidates) == 1:
                                cx, cy = traj.getCentroidOfContour(candidates[0])
                                rebound_list.append([frame_number, (int(cx), int(cy))])
                            # several candidates (add candidate closest to optimum point)
                            elif len(candidates) > 1:
                                best_point = traj.getBestPointInNextFrame(rebound_list, frame_number)
                                min_distance = math.sqrt(pow(frame.shape[0], 2) + pow(frame.shape[1], 2))
                                candidate_to_add = candidates[0]
                                for candidate in candidates:
                                    cx, cy = traj.getCentroidOfContour(candidate)
                                    # calculate distance between predicted droplet position and current droplet centroid
                                    distance = math.sqrt(pow(abs(best_point[0] - cx), 2) +
                                                         pow(abs(best_point[1] - cy), 2))
                                    if distance < min_distance:
                                        min_distance = distance
                                        candidate_to_add = candidate
                                cx, cy = traj.getCentroidOfContour(candidate_to_add)
                                rebound_list.append([frame_number, (int(cx), int(cy))])
                    # sampling points 6 and onwards
                    elif len(rebound_list) > 4:
                        # check if temporal distance is below threshold value
                        if abs(rebound_list[-1][0] - frame_number) > params.TIME_BETWEEN_BLOBS:
                            pass
                        else:
                            distance = traj.getDistanceInNextFrame(rebound_list, frame_number)
                            fit = traj.getFitDropletList(rebound_list)
                            # draw line
                            frame = cv2.line(frame, (0, int(0 * fit[0] + fit[1])),
                                             (frame.shape[1], int(frame.shape[1] * fit[0] + fit[1])), [200, 200, 0], 1)
                            acceptance_angle = traj.getAcceptanceAngle(rebound_list, frame_number)
                            mask_to_check = traj.getCylinderMask(frame, fit, distance,
                                                                 params.ADD_TO_SEARCH_RADIUS, acceptance_angle,
                                                                 rebound_list[-1][1], rebound_list)
                            contours, hier = cv2.findContours(fgmask_opt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            candidates = list()
                            for contour in contours:
                                if cv2.contourArea(contour) > 0:
                                    cx, cy = traj.getCentroidOfContour(contour)
                                    mask = np.zeros((frame.shape[0], frame.shape[1])).astype('uint8')
                                    mask[cy, cx] = 255
                                    result = cv2.bitwise_and(mask, mask_to_check)
                                    if cv2.countNonZero(result) > 0:
                                        candidates.append(contour)
                            # one candidate only (add always)
                            if len(candidates) == 1:
                                cx, cy = traj.getCentroidOfContour(candidates[0])
                                rebound_list.append([frame_number, (int(cx), int(cy))])
                            # several candidates (add candidate closest to optimum point)
                            elif len(candidates) > 1:
                                best_point = traj.getBestPointInNextFrame(rebound_list, frame_number)
                                min_distance = math.sqrt(pow(frame.shape[0], 2) + pow(frame.shape[1], 2))
                                candidate_to_add = candidates[0]
                                for candidate in candidates:
                                    cx, cy = traj.getCentroidOfContour(candidate)
                                    # calculate distance between predicted droplet position and current droplet centroid
                                    distance = math.sqrt(pow(abs(best_point[0] - cx), 2) +
                                                         pow(abs(best_point[1] - cy), 2))
                                    if distance < min_distance:
                                        min_distance = distance
                                        candidate_to_add = candidate
                                cx, cy = traj.getCentroidOfContour(candidate_to_add)
                                rebound_list.append([frame_number, (int(cx), int(cy))])

                # show complete trajectory analysis result
                for point in droplets_list:
                    droplet = point[1]
                    frame = cv2.circle(frame, (droplet[0], droplet[1]), 1, [0, 255, 0], -1)
                for point in rebound_list:
                    droplet = point[1]
                    frame = cv2.circle(frame, (droplet[0], droplet[1]), 1, [255, 255, 0], -1)
                frame_large = cv2.resize(frame, (int(2.0 * frame.shape[1]), int(2.0 * frame.shape[0])),
                                         interpolation=cv2.INTER_LINEAR)
                fgmask_opt_large = cv2.resize(fgmask_opt, (int(2.0 * frame.shape[1]), int(2.0 * frame.shape[0])),
                                              interpolation=cv2.INTER_LINEAR)

                cv2.imshow("Frame", frame_large)
                cv2.imshow("Foreground", fgmask_opt_large)

                # write frame to output sequence, if current frame contains detected rebound droplet position
                if len(rebound_list) > 0:
                    if rebound_list[-1][0] == frame_number:
                        output_all.write(frame_large)
                cv2.waitKey(1)

        # reset video reader object and read first frame only
        video = cv2.VideoCapture(video_path)
        ret, frame = video.read()

        # draw droplet positions on main trajectory in first frame
        for point in droplets_list:
            droplet = point[1]
            frame = cv2.circle(frame, (droplet[0], droplet[1]), 1, [155, 88, 0], -1)
        # draw droplet positions on rebound trajectory in first frame
        for point in rebound_list:
            droplet = point[1]
            frame = cv2.circle(frame, (droplet[0], droplet[1]), 1, [0, 0, 255], -1)

        cv2.imwrite(saving_path + pat + "_" + sequence_number + "_Rebound_Trajectory.png", frame)
        cv2.imwrite("Cover_Image.png", frame)

        # EVALUATION OF DROPLET REBOUND TRAJECTORY
        file.write("\nEVALUATION OF SAMPLING POINTS ON REBOUND TRAJECTORY\n\n")
        file.write("List of sampling points on rebound trajectory:\n")
        file.write(str(rebound_list))
        file.write("\n\n")

        tracking = True

        if len(rebound_list) <= 5:
            file.write("Not enough sampling points available on rebound trajectory.\n")
            # file.write("Evaluation of impact type not possible (not enough sampling points).\n")
            tracking = False
        else:
            points = list()
            frames = list()
            # add main trajectory to 'points'
            # 'droplets_list' may have been shortened if angle > 40°
            for i in range(6, len(droplets_list)):
                droplet = droplets_list[i]
                points.append(droplet[1])
                frames.append(droplet[0])
            # add rebound trajectory to 'points'
            # skip first point (redundant)
            for i in range(1, len(rebound_list)):
                droplet = rebound_list[i]
                points.append(droplet[1])
                frames.append(droplet[0])

        # if less than six sampling points detected on rebound trajectory:
        # set stimulation type to impact
        if not len(points) >= 7:
            tracking = False

        stimulation = "impact"

        if tracking:
            # identify angle between principal and rebound trajectories after rebound trajectory refinement
            angle, list_first, list_second, impact_number = traj.iterative_impact(points, frames)
            file.write("Angle between principal and rebound trajectories: " + str(angle) + "\n")

            # NEW SECTION
            print("\n")
            print("'points' (second loop): ")
            print(points)
            print("\n")
            print("'list_first' (second loop): ")
            print(list_first)
            print("'list_second' (second loop): ")
            print(list_second)
            print("\n")
            print("'frame_impact' (second loop): ", impact_number)
            print("\n")

            # draw fit lines
            complete_list = droplets_list[6:] + rebound_list[1:]
            # get list of sampling points up to impact
            droplets_list_before_impact = traj.getMainTrajectoryUntilImpact(complete_list, impact_number)
            # get linear fit for sampling points up to impact
            fit_before = traj.getFitDropletList(droplets_list_before_impact)
            # get first frame of sequence
            video = cv2.VideoCapture(video_path)
            ret, frame = video.read()
            # draw all sampling points before impact on frame
            for point in droplets_list_before_impact:
                droplet = point[1]
                frame = cv2.circle(frame, (droplet[0], droplet[1]), 1, [0, 255, 0], -1)
            # draw fit line for points before impact
            frame = cv2.line(frame, (0, int(0 * fit_before[0] + fit_before[1])),
                             (frame.shape[1], int(frame.shape[1] * fit_before[0] + fit_before[1])), [0, 200, 0], 1)

            # get list of sampling points after impact
            droplets_list_after_impact = complete_list[len(droplets_list_before_impact):]
            if droplets_list_after_impact:
                # get linear fit for sampling points after impact
                fit_after = traj.getFitDropletList(droplets_list_after_impact)
                # draw all sampling points after impact on frame
                for point in droplets_list_after_impact:
                    droplet = point[1]
                    frame = cv2.circle(frame, (droplet[0], droplet[1]), 1, [0, 0, 255], -1)
                # draw fit line for points after impact
                frame = cv2.line(frame, (0, int(0 * fit_after[0] + fit_after[1])),
                                 (frame.shape[1], int(frame.shape[1] * fit_after[0] + fit_after[1])), [0, 0, 200], 1)
            # resize frame
            frame_large = cv2.resize(frame, (int(2.0 * frame.shape[1]), int(2.0 * frame.shape[0])),
                                     interpolation=cv2.INTER_LINEAR)
            # show result
            cv2.imshow("Impact/rebound trajectories (second loop)", frame_large)
            cv2.waitKey(2000)
            cv2.destroyWindow("Impact/rebound trajectories (second loop)")
            # save result
            cv2.imwrite(saving_path + pat + "_" + sequence_number + "_Impact_Rebound_Second_Loop.png",
                        frame_large)

            # END OF NEW SECTION

            # if acute angle between principal and rebound trajectories below 15 degrees: impact identified
            if angle < 15:
                stimulation = "impact"
                # if deviation of more than 40 degrees found in first impact/rebound identification loop:
                # impact/rebound identification not reliable
                if angle_greater_40:
                    stimulation = "unclear"
            # if acute angle between principal and rebound trajectories greater than 15 degrees
            else:
                # calculate Euclidean distance between first and last sampling point on principal trajectory
                distance_main = traj.getDistanceBetweenDroplets([droplets_list[0][1][0], droplets_list[0][1][1]],
                                                                [droplets_list[-1][1][0], droplets_list[-1][1][1]])
                # calculate Euclidean distance between first and last sampling point on rebound trajectory
                distance_rebound = traj.getDistanceBetweenDroplets([rebound_list[0][1][0], rebound_list[0][1][1]],
                                                                   [rebound_list[-1][1][0], rebound_list[-1][1][1]])
                # classify as impact if length of main trajectory less than 1/7 or more than 7 times the length of
                # rebound trajectory (error handling)
                if distance_rebound/distance_main < (1/7):
                    stimulation = "impact"
                    tracking = False
                else:
                    stimulation = "rebound"

        file.write("\nFINAL RESULT\n\n")
        file.write("Stimulation type: " + stimulation + "\n\n")
        if tracking:
            file.write("Final frame index of droplet impact: " + str(impact_number) + "\n\n")
        else:
            file.write("Final frame index of droplet impact: " + str(frame_impact) + "\n\n")
        if stimulation == "rebound":
            file.write("Rebound angle: " + str(angle) + "\n\n")

        # DETECTION OF ADDITIONAL DROPLETS (FOURTH LOOP)
        main_list = list()
        # add all detected droplet positions on main and rebound trajectories to 'main_list'
        for point in droplets_list:
            main_list.append(point)
        for point in rebound_list:
            main_list.append(point)

        # initialize background subtraction
        background = cv2.createBackgroundSubtractorMOG2(history=params.BGSM_HISTORY, detectShadows=False)
        background.setBackgroundRatio(params.BGSM_BACKGROUND_RATIO)

        # load sequence
        video = cv2.VideoCapture(video_path)
        # initialize frame index
        frame_number = 0
        further_droplets_list = list()
        while 1:
            # load frame
            ret, frame = video.read()
            # break loop if no frame available
            if not ret:
                print("break")
                break
            else:
                # increment and print frame index
                frame_number = frame_number + 1
                # print(frame_number)

                frame_mean = prepro.getPreForBackgroundSubtraction(frame)

                # apply background subtraction to obtain moving foreground
                fgmask = background.apply(frame_mean)

                # optimize foreground
                # apply morphological closing to obtain homogeneous circular objects
                fgmask_opt = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, params.KERNEL_CIRCLE)

                fgmask_opt[:, 0] = 0
                fgmask_opt[0, :] = 0
                fgmask_opt[:, frame.shape[1]-1] = 0
                fgmask_opt[frame.shape[0] - 1 - 5, :] = 0

                # if current frame contains previously detected droplet position on main or rebound trajectory
                if main_list[0][0] <= frame_number <= main_list[-1][0]:
                    # first detection of additional droplets
                    if len(further_droplets_list) == 0:
                        droplets = []
                        # search for additional droplets
                        droplets = detector.detect(cv2.bitwise_not(fgmask_opt))
                        candidates = list()
                        # identify detected main droplet position in current frame
                        for i in range(0, len(main_list)):
                            if main_list[i][0] == frame_number:
                                index = i
                        for droplet in droplets:
                            # retain all candidate blobs with preset minimum radius
                            if droplet.size / 2.0 > 3:
                                distance = traj.getDistanceBetweenDroplets([droplet.pt[0],
                                                                            droplet.pt[1]],
                                                                           [main_list[index][1][0],
                                                                            main_list[index][1][1]])
                                # if Euclidean distance of current candidate to main droplet larger than preset value:
                                # append candidate to list of additional droplets
                                if distance > 40:
                                    candidates.append(droplet)
                        # if one candidate only
                        if len(candidates) == 1:
                            # append always
                            further_droplets_list.append([frame_number, (int(candidates[0].pt[0]),
                                                                         int(candidates[0].pt[1]))])
                        # if more than one candidate:
                        # select most plausible candidate to append
                        elif len(candidates) > 1:
                            last_droplet = droplets_list[-1][1]
                            min_distance = math.sqrt(pow(frame.shape[0], 2) + pow(frame.shape[1], 2))
                            candidate_to_add = candidates[0]
                            for candidate in candidates:
                                distance = traj.getDistanceBetweenDroplets([frame.shape[0] / 2.0, frame.shape[1] / 2.0],
                                                                           [candidate.pt[0], candidate.pt[1]])
                                if distance < min_distance:
                                    min_distance = distance
                                    candidate_to_add = candidate
                            # add candidate which is closest to frame center
                            further_droplets_list.append([frame_number,
                                                             (int(candidate_to_add.pt[0]),
                                                              int(candidate_to_add.pt[1]))])

                # if current frame contains previously detected droplet position on main or rebound trajectory
                if main_list[0][0] <= frame_number <= main_list[-1][0]:
                    # later detections of additional droplets
                    if len(further_droplets_list) > 0 and frame_number > further_droplets_list[-1][0]:
                        # second detection of additional droplets
                        if len(further_droplets_list) == 1:
                            # check if temporal distance is below threshold value
                            if abs(further_droplets_list[-1][0] - frame_number) > params.TIME_BETWEEN_BLOBS:
                                pass
                            else:
                                droplets = detector.detect(cv2.bitwise_not(fgmask_opt))
                                candidates = list()
                                for i in range(0, len(main_list)):
                                    if main_list[i][0] == frame_number:
                                        index = i
                                for droplet in droplets:
                                    # retain all candidate blobs with preset minimum radius
                                    if droplet.size / 2.0 > 3:
                                        last_droplet = further_droplets_list[-1][1]
                                        distance = traj.getDistanceBetweenDroplets(last_droplet,
                                                                                   [droplet.pt[0], droplet.pt[1]])
                                        # check if detected blob is within circle around first droplet position
                                        if distance < params.MAX_RADIUS_SECOND_DROPLET_MAIN:
                                            distance = traj.getDistanceBetweenDroplets([droplet.pt[0], droplet.pt[1]],
                                                                                       [main_list[index][1][0],
                                                                                        main_list[index][1][1]])
                                            # if Euclidean distance of current candidate to main droplet larger than
                                            # preset value:
                                            # append candidate to list of additional droplets
                                            if distance > 40:
                                                candidates.append(droplet)
                                # if one candidate only: append always
                                if len(candidates) == 1:
                                    further_droplets_list.append([frame_number,
                                                                  (int(candidates[0].pt[0]), int(candidates[0].pt[1]))])
                                # if more than one candidate:
                                # select most plausible candidate to append
                                elif len(candidates) > 1:
                                    last_droplet = droplets_list[-1][1]
                                    min_distance = math.sqrt(pow(frame.shape[0], 2) + pow(frame.shape[1], 2))
                                    candidate_to_add = candidates[0]
                                    for candidate in candidates:
                                        distance = traj.getDistanceBetweenDroplets([frame.shape[0] / 2.0,
                                                                                    frame.shape[1] / 2.0],
                                                                                   [candidate.pt[0], candidate.pt[1]])
                                        if distance < min_distance:
                                            min_distance = distance
                                            candidate_to_add = candidate
                                    # add candidate which is closest to frame center
                                    further_droplets_list.append([frame_number,
                                                                     (int(candidate_to_add.pt[0]),
                                                                      int(candidate_to_add.pt[1]))])
                        # additional droplet position 3 to 5
                        elif 1 < len(further_droplets_list) <= 4:
                            # check if temporal distance is below threshold value
                            if abs(further_droplets_list[-1][0] - frame_number) > params.TIME_BETWEEN_BLOBS:
                                pass
                            else:
                                print("List of detected additional droplet positions:")
                                print(further_droplets_list)
                                distance = traj.getDistanceInNextFrame(further_droplets_list, frame_number)
                                contours, hier = cv2.findContours(fgmask_opt,
                                                                  cv2.RETR_EXTERNAL,
                                                                  cv2.CHAIN_APPROX_SIMPLE)
                                candidates = list()
                                for contour in contours:
                                    if cv2.contourArea(contour) > 0:
                                        cx, cy = traj.getCentroidOfContour(contour)
                                        distance_point = traj.getDistanceBetweenDroplets(further_droplets_list[-1][1],
                                                                                         [cx, cy])
                                        if distance_point <= distance + params.ADD_TO_SEARCH_RADIUS:
                                            candidates.append(contour)
                                # one candidate only (add always)
                                if len(candidates) == 1:
                                    cx, cy = traj.getCentroidOfContour(candidates[0])
                                    further_droplets_list.append([frame_number, (int(cx), int(cy))])
                                # several candidates (add candidate closest to optimum point)
                                elif len(candidates) > 1:
                                    best_point = traj.getBestPointInNextFrame(further_droplets_list, frame_number)
                                    min_distance = math.sqrt(pow(frame.shape[0], 2) + pow(frame.shape[1], 2))
                                    candidate_to_add = candidates[0]
                                    for candidate in candidates:
                                        cx, cy = traj.getCentroidOfContour(candidate)
                                        # calculate distance between predicted droplet position and
                                        # current droplet centroid
                                        distance = math.sqrt(pow(abs(best_point[0] - cx), 2) +
                                                             pow(abs(best_point[1] - cy), 2))
                                        if distance < min_distance:
                                            min_distance = distance
                                            candidate_to_add = candidate
                                    cx, cy = traj.getCentroidOfContour(candidate_to_add)
                                    further_droplets_list.append([frame_number, (int(cx), int(cy))])
                        # sampling point 6 and onwards
                        elif len(further_droplets_list) > 4:
                            # check if temporal distance is below threshold value
                            if abs(further_droplets_list[-1][0] - frame_number) > params.TIME_BETWEEN_BLOBS:
                                pass
                            else:
                                distance = traj.getDistanceInNextFrame(further_droplets_list, frame_number)
                                fit = traj.getFitDropletList(further_droplets_list)
                                # draw line
                                frame = cv2.line(frame, (0, int(0 * fit[0] + fit[1])),
                                                 (frame.shape[1], int(frame.shape[1] * fit[0] + fit[1])),
                                                 [0, 200, 200], 1)
                                acceptance_angle = traj.getAcceptanceAngle(further_droplets_list, frame_number)
                                mask_to_check = traj.getCylinderMask(frame, fit, distance, params.ADD_TO_SEARCH_RADIUS,
                                                                     acceptance_angle, further_droplets_list[-1][1],
                                                                     further_droplets_list)
                                # draw search area
                                # frame[mask_to_check == 255] = [0, 255, 255]
                                mask_search_area = np.zeros((256, 256, 3)).astype('uint8')
                                mask_search_area[mask_to_check == 255] = [0, 200, 200]
                                frame = cv2.addWeighted(frame, 1.0, mask_search_area, 0.3, 0.0)

                                contours, hier = cv2.findContours(fgmask_opt,
                                                                  cv2.RETR_EXTERNAL,
                                                                  cv2.CHAIN_APPROX_SIMPLE)
                                candidates = list()
                                for contour in contours:
                                    if cv2.contourArea(contour) > 0:
                                        cx, cy = traj.getCentroidOfContour(contour)
                                        mask = np.zeros((frame.shape[0], frame.shape[1])).astype('uint8')
                                        mask[cy, cx] = 255
                                        result = cv2.bitwise_and(mask, mask_to_check)
                                        if cv2.countNonZero(result) > 0:
                                            candidates.append(contour)
                                # one candidate only (add always)
                                if len(candidates) == 1:
                                    cx, cy = traj.getCentroidOfContour(candidates[0])
                                    further_droplets_list.append([frame_number, (int(cx), int(cy))])
                                # several candidates (add candidate closest to optimum point)
                                elif len(candidates) > 1:
                                    best_point = traj.getBestPointInNextFrame(droplets_list, frame_number)
                                    min_distance = math.sqrt(pow(frame.shape[0], 2) + pow(frame.shape[1], 2))
                                    candidate_to_add = candidates[0]
                                    for candidate in candidates:
                                        cx, cy = traj.getCentroidOfContour(candidate)
                                        # calculate distance between predicted droplet position and
                                        # current droplet centroid
                                        distance = math.sqrt(
                                            pow(abs(best_point[0] - cx), 2) + pow(abs(best_point[1] - cy), 2))
                                        if distance < min_distance:
                                            min_distance = distance
                                            candidate_to_add = candidate
                                    cx, cy = traj.getCentroidOfContour(candidate_to_add)
                                    further_droplets_list.append([frame_number, (int(cx), int(cy))])

                        for point in further_droplets_list:
                            frame = cv2.circle(frame, (int(point[1][0]), int(point[1][1])), 1, [0, 255, 255])

                        frame_large = cv2.resize(frame, (int(2.0 * frame.shape[1]), int(2.0 * frame.shape[0])),
                                                 interpolation=cv2.INTER_LINEAR)
                        fgmask_opt_large = cv2.resize(fgmask_opt,
                                                      (int(2.0 * frame.shape[1]), int(2.0 * frame.shape[0])),
                                                      interpolation=cv2.INTER_LINEAR)

                        # write frame to output sequence
                        output_all.write(frame_large)

                        cv2.imshow("Frame", frame_large)
                        cv2.imshow("Foreground", fgmask_opt_large)
                        cv2.waitKey(1)

        # if more than five frames with additional droplet(s) detected
        if len(further_droplets_list) > 5:
            video = cv2.VideoCapture(video_path)
            ret, frame = video.read()
            for point in further_droplets_list:
                frame = cv2.circle(frame, (int(point[1][0]), int(point[1][1])), 1, [0, 255, 255])
            cv2.imwrite(saving_path + pat + "_" + sequence_number + "_Further_Droplets.png", frame)
            file.write("Further droplets detected: yes\n\n")
            file.write("Sampling points of further droplets:\n")
            file.write(str(further_droplets_list))
        else:
            file.write("Further droplets detected: no")

    # close result file
    if file:
        file.close()

    # close video output object
    if output_all:
        output_all.release()
