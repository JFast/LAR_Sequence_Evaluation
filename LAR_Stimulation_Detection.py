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

# load frame sequence
# path to file
pat = "05"
sequence_number = "01"
# use avi file
video_path = "E:/LARvideos/videos_annotated/pat_" + pat + "\es_01_pat_" + pat + "_seq_" + sequence_number + "\es_01_pat_" + pat + "_seq_" + sequence_number + ".avi"
# use mp4 file
# video_path = "E:/LARvideos/videos_annotated/pat_" + pat + "\es_01_pat_" + pat + "_seq_" + sequence_number + "\es_01_pat_" + pat + "_seq_" + sequence_number + ".mp4"
video = cv2.VideoCapture(video_path)


# FILE FOR RESULT STORAGE
saving_path = r"E:/Masterarbeit_Andra_Oltmann/Results_TMI/LAR_Stimulation_Detection/"
file = open(saving_path + pat + "_" + sequence_number + "Evaluation.txt", "w")
file.write("EVALUATION\n")
file.write("Sequence identifier: es_01_pat_" + pat + "_seq_" + sequence_number + "\n\n")

# INITIAL DETECTION OF DROPLET
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
        print(frame_number)

        frame_mean = prepro.getPreForBackgroundSubtraction(frame)

        # apply background subtraction to obtain moving droplet
        fgmask = background.apply(frame_mean)

        # optimize foreground
        # apply morphological closing to obtain homogeneous, circular objects
        fgmask_opt = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, params.KERNEL_CIRCLE)

        droplets = []
        if frame_number > params.BGSM_HISTORY:
            # apply OpenCV blob detection on inverted frame after background subtraction to find blobs in foreground
            droplets = detector.detect(cv2.bitwise_not(fgmask_opt))
            for droplet in droplets:
                if not first_frame_detect:
                    # retain all candidates with minimum radius MIN_DROPLET_RADIUS
                    if droplet.size / 2.0 > params.MIN_DROPLET_RADIUS:
                        frame_result = frame.copy()
                        # draw found blobs as circles
                        frame_result = cv2.circle(frame_result, (int(droplet.pt[0]), int(droplet.pt[1])), int(droplet.size/2.0), [0, 0, 255], 1)
                        first_frame = frame_number
                        first_frame_detect = True
                        # update background
                        background_image = background.getBackgroundImage()

        cv2.imshow("Frame", frame)
        cv2.imshow("Foreground", fgmask_opt)
        cv2.waitKey(1)

tracking = True

# if no droplet found: end of algorithm
if first_frame == 0:
    file.write("No droplet detected!\n")
    tracking = False

# DETECTION OF FURTHER DROPLET POSITIONS
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
            print(frame_number)

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
                # retain all candidates with minimum radius MIN_DROPLET_RADIUS
                candidates = list()
                for droplet in droplets:
                    if droplet.size / 2.0 > params.MIN_DROPLET_RADIUS:
                        candidates.append(droplet)
                # select largest candidate
                candidate_to_add = traj.getMaxCandidate(candidates)
                droplets_list.append([frame_number, (int(candidate_to_add.pt[0]), int(candidate_to_add.pt[1]))])

            # detect additional sampling points
            check_frame = False
            if frame_number > first_frame:
                # detection of second sampling point
                if len(droplets_list) == 1:
                    # check if temporal distance is below threshold value
                    if abs(droplets_list[-1][0] - frame_number) > params.TIME_BETWEEN_BLOBS:
                        check_frame = True
                    else:
                        droplets = []
                        droplets = detector.detect(cv2.bitwise_not(fgmask_opt))
                        print(droplets)
                        candidates = list()
                        for droplet in droplets:
                            last_droplet = droplets_list[-1][1]
                            distance = traj.getDistanceBetweenDroplets(last_droplet, [droplet.pt[0], droplet.pt[1]])
                            # check if detected blob is within circle around first droplet position
                            if distance < params.MAX_RADIUS_SECOND_DROPLET_MAIN:
                                candidates.append(droplet)
                        # if only one candidate found: add to list
                        if len(candidates) == 1:
                            droplets_list.append([frame_number, (int(candidates[0].pt[0]), int(candidates[0].pt[1]))])
                        # if more than one candidate found: add to list
                        elif len(candidates) > 1:
                            last_droplet = droplets_list[-1][1]
                            # initialize min_distance with length of frame diagonal
                            min_distance = math.sqrt(pow(frame.shape[0], 2) + pow(frame.shape[1], 2))
                            # select first candidate
                            candidate_to_add = candidates[0]
                            # go through all candidates and check distance to frame center
                            for candidate in candidates:
                                distance = traj.getDistanceBetweenDroplets([frame.shape[0]/2.0, frame.shape[1]/2.0],
                                                                           [candidate.pt[0], candidate.pt[1]])
                                if distance < min_distance:
                                    min_distance = distance
                                    # select candidate with lowest distance to frame center as candidate to add
                                    candidate_to_add = candidate
                            droplets_list.append([frame_number, (int(candidate_to_add.pt[0]), int(candidate_to_add.pt[1]))])
                # sampling points 2 to 5
                elif len(droplets_list) > 1 and len(droplets_list) <= 4:
                    # check if temporal distance is below threshold value
                    if abs(droplets_list[-1][0] - frame_number) > params.TIME_BETWEEN_BLOBS:
                        check_frame = True
                    else:
                        distance = traj.getDistanceInNextFrame(droplets_list, frame_number)
                        contours, hier = cv2.findContours(fgmask_opt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        candidates = list()
                        if len(contours) > 1:
                            for contour in contours:
                                if cv2.contourArea(contour) > 0:
                                    cx, cy = traj.getCentroidOfContour(contour)
                                    distance_point = traj.getDistanceBetweenDroplets(droplets_list[-1][1], [cx, cy])
                                    if distance_point <= distance + params.ADD_TO_SEARCH_RADIUS:
                                        candidates.append(contour)
                        elif len(contours) == 1:
                            for contour in contours:
                                if cv2.contourArea(contour) > 0:
                                    M = cv2.moments(contour)
                                    # calculate centroid coordinates
                                    cx = int(M['m10'] / M['m00'])
                                    cy = int(M['m01'] / M['m00'])
                                    distance_point = traj.getDistanceBetweenDroplets(droplets_list[-1][1], [cx, cy])
                                    if distance_point <= 60:
                                        candidates.append(contour)
                        # one candidate only (add always)
                        if len(candidates) == 1:
                            cx, cy = traj.getCentroidOfContour(candidates[0])
                            droplets_list.append([frame_number, (int(cx), int(cy))])
                        # several candidates (only add candidate closest to optimum point)
                        elif len(candidates) > 1:
                            best_point = traj.getBestPointInNextFrame(droplets_list, frame_number)
                            min_distance = math.sqrt(pow(frame.shape[0], 2) + pow(frame.shape[1], 2))
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
                        distance = traj.getDistanceInNextFrame(droplets_list, frame_number)
                        fit = traj.getFitDropletList(droplets_list)
                        frame = cv2.line(frame, (0, int(0 * fit[0] + fit[1])),
                                         (frame.shape[1], int(frame.shape[1] * fit[0] + fit[1])), [255, 0, 0], 1)
                        acceptance_angle = traj.getAcceptanceAngle(droplets_list, frame_number)
                        mask_to_check = traj.getCylinderMask(frame, fit, distance, params.ADD_TO_SEARCH_RADIUS,
                                                             acceptance_angle, droplets_list[-1][1], droplets_list)
                        frame[mask_to_check == 255] = [0, 255, 255]
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
                            droplets_list.append([frame_number, (int(cx), int(cy))])
                        # several candidates (only add candidate closest to optimum point)
                        elif len(candidates) > 1:
                            best_point = traj.getBestPointInNextFrame(droplets_list, frame_number)
                            min_distance = math.sqrt(pow(frame.shape[0], 2) + pow(frame.shape[1], 2))
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
                        frame = cv2.circle(frame, (droplet[0], droplet[1]), 2, [0, 255, 0], -1)
                    cv2.imshow("Frame", frame)
                    cv2.imshow("Foreground", fgmask_opt)
                    cv2.waitKey()
                else:
                    for point in droplets_list:
                        droplet = point[1]
                        frame = cv2.circle(frame, (droplet[0], droplet[1]), 2, [0, 255, 0], -1)
                    cv2.imshow("Frame", frame)
                    cv2.imshow("Foreground", fgmask_opt)
                    cv2.waitKey(1)

    # show trajectory as frame overlay and save result
    video = cv2.VideoCapture(video_path)
    ret, frame = video.read()
    frame_result = frame.copy()
    for point in droplets_list:
        droplet = point[1]
        frame = cv2.circle(frame, (droplet[0], droplet[1]), 2, [0, 0, 0], -1)
    cv2.imwrite(saving_path + pat + "_" + sequence_number + "_Sampling_Points_Principal_Trajectory.png", frame)

    file.write("\n\nEVALUATION OF SAMPLING POINTS ON PRINCIPAL TRAJECTORY\n")
    file.write("List of sampling points on principal trajectory: \n")
    file.write(str(droplets_list))
    file.write("\n")

    tracking = True
    if len(droplets_list) > 3:
        while droplets_list[-1][0] - droplets_list[-2][0] > 10 or droplets_list[-2][0] - droplets_list[-3][0] > 10:
            droplets_list.remove(droplets_list[-1])
            droplets_list.remove(droplets_list[-1])

    if len(droplets_list) <= 7 + 7:
        file.write("Evaluation of principal trajectory not possible (not enough sampling points).\n")
        print("Tracking not possible!")
        tracking = False

    # IMPACT/REBOUND DISTINCTION PROCEDURE
    if tracking:
        points = list()
        frames = list()
        
        for i in range(6, len(droplets_list)):
            droplet = droplets_list[i]
            points.append(droplet[1])
            frames.append(droplet[0])
        # separate sampling points into two subsets
        angle, list_first, list_second, impact_number = traj.iterative_impact(points, frames)

        file.write("Angle of principal trajectory: " + str(angle) + "\n")
        if angle < 40:
            frame_impact = droplets_list[-1][0]
        else:
            frame_impact = impact_number
            droplets_list = traj.getMainTrajectoryUntilImpact(droplets_list, frame_impact)
        file.write("Frame index of droplet impact (principal trajectory): " + str(frame_impact) + "\n")

        video = cv2.VideoCapture(video_path)
        ret, frame = video.read()
        print(droplets_list)
        for point in droplets_list:
            droplet = point[1]
            frame = cv2.circle(frame, (droplet[0], droplet[1]), 2, [0, 255, 0], -1)
        cv2.imwrite(saving_path + pat + "_" + sequence_number +
                    "_Sampling_Points_on_Principal_Trajectory_After_Post-Processing.png", frame)
        file.write("List of sampling points on principal trajectory after post-processing: \n")
        file.write(str(droplets_list))
        file.write("\n")

        # IDENTIFY SAMPLING POINTS ON DROPLET REBOUND TRAJECTORY
        # initialize background subtraction
        background = cv2.createBackgroundSubtractorMOG2(history=params.BGSM_HISTORY, detectShadows=False)
        background.setBackgroundRatio(params.BGSM_BACKGROUND_RATIO)
        # Video laden
        video = cv2.VideoCapture(video_path)
        # initialize frame index
        frame_number = 0
        # create list for sampling points on rebound trajectory
        rebound_list = list()
        # iterate over frames of sequence
        while (1):
            # load frame
            ret, frame = video.read()
            # break loop if no frame available
            if not ret:
                print("break")
                break
            else:
                # increment and print frame index
                frame_number = frame_number + 1
                print(frame_number)

                frame_mean = prepro.getPreForBackgroundSubtraction(frame)

                # apply background subtraction to obtain moving foreground
                fgmask = background.apply(frame_mean)

                # optimize foreground
                # apply morphological closing to obtain homogeneous circular objects
                fgmask_opt = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, params.KERNEL_CIRCLE)

                if frame_number == droplets_list[-1][0]:
                    rebound_list.append([frame_number, droplets_list[-1][1]])

                # identify additional sampling points
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
                                                                                rebound_list[-1][1][1]],[cx, cy])
                                    if distance < min_distance:
                                        min_distance = distance
                                        candidate_to_add = candidate
                                cx, cy = traj.getCentroidOfContour(candidate_to_add)
                                rebound_list.append([frame_number, (int(cx), int(cy))])
                    # sampling points 2 to 5
                    elif len(rebound_list) > 1 and len(rebound_list) <= 4:
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
                    # Stützstellen ab Nr. 6
                    # sampling points no. 6 and onwards
                    elif len(rebound_list) > 4:
                        # check if temporal distance is below threshold value
                        if abs(rebound_list[-1][0] - frame_number) > params.TIME_BETWEEN_BLOBS:
                            pass
                        else:
                            distance = traj.getDistanceInNextFrame(rebound_list, frame_number)
                            fit = traj.getFitDropletList(rebound_list)
                            frame = cv2.line(frame, (0, int(0 * fit[0] + fit[1])),
                                             (frame.shape[1], int(frame.shape[1] * fit[0] + fit[1])), [255, 0, 0], 1)
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
                                    distance = math.sqrt(
                                        pow(abs(best_point[0] - cx), 2) + pow(abs(best_point[1] - cy), 2))
                                    if distance < min_distance:
                                        min_distance = distance
                                        candidate_to_add = candidate
                                cx, cy = traj.getCentroidOfContour(candidate_to_add)
                                rebound_list.append([frame_number, (int(cx), int(cy))])

                for point in droplets_list:
                    droplet = point[1]
                    frame = cv2.circle(frame, (droplet[0], droplet[1]), 2, [0, 255, 0], -1)
                for point in rebound_list:
                    droplet = point[1]
                    frame = cv2.circle(frame, (droplet[0], droplet[1]), 2, [255, 255, 0], -1)
                cv2.imshow("Frame", frame)
                cv2.imshow("Foreground", fgmask_opt)
                cv2.waitKey(1)

        video = cv2.VideoCapture(video_path)
        ret, frame = video.read()
        for point in rebound_list:
            droplet = point[1]
            frame = cv2.circle(frame, (droplet[0], droplet[1]), 2, [155, 88, 0], -1)
        for point in droplets_list:
            droplet = point[1]
            frame = cv2.circle(frame, (droplet[0], droplet[1]), 2, [155, 88, 0], -1)
        cv2.imwrite(saving_path + pat + "_" + sequence_number + "_Rebound_Trajectory.png", frame)
        cv2.imwrite("Deckblatt.png", frame)

        # EVALUATION OF DROPLET REBOUND TRAJECTORY
        file.write("\nEVALUATION OF SAMPLING POINTS ON REBOUND TRAJECTORY\n")
        file.write("List of sampling points on rebound trajectory: ")
        file.write(str(rebound_list))
        file.write("\n")
        tracking = True
        if len(rebound_list) <= 5:
            file.write("Evaluation of rebound trajectory not possible (not enough sampling points).\n")
            tracking = False
        else:
            points = list()
            frames = list()
            for i in range(6, len(droplets_list)):
                droplet = droplets_list[i]
                points.append(droplet[1])
                frames.append(droplet[0])
            for i in range(0, len(rebound_list)):
                droplet = rebound_list[i]
                points.append(droplet[1])
                frames.append(droplet[0])

        if not len(points) >= 7:
            tracking = False

        stimulation = "Impact"
        if tracking:
            angle, list_first, list_second, impact_number = traj.iterative_impact(points, frames)
            file.write("Angle between principal and rebound trajectories: " + str(angle) + "\n")

            stimulation = ""
            if angle < 15:
                stimulation = "Impact"
            else:
                distance_main = traj.getDistanceBetweenDroplets([droplets_list[0][1][0], droplets_list[0][1][1]],
                                                                [droplets_list[-1][1][0], droplets_list[-1][1][1]])
                distance_abprall = traj.getDistanceBetweenDroplets([rebound_list[0][1][0], rebound_list[0][1][1]],
                                                                   [rebound_list[-1][1][0], rebound_list[-1][1][1]])
                if distance_main/distance_abprall < (1/7) or distance_main/distance_abprall > 7:
                    stimulation = "Impact"
                else:
                    stimulation = "Rebound"

        file.write("\nFINAL RESULT\n")
        file.write("Stimulation type: " + stimulation + "\n")
        file.write("Frame index of droplet impact: " + str(frame_impact) + "\n")
        if stimulation == "Rebound":
            file.write("Rebound angle: " + str(angle) + "\n")


        # DETECTION OF ADDITIONAL DROPLETS
        main_list = list()
        for point in droplets_list:
            main_list.append(point)
        for point in rebound_list:
            main_list.append(point)

        # initialize background subtraction
        background = cv2.createBackgroundSubtractorMOG2(history=params.BGSM_HISTORY, detectShadows=False)
        background.setBackgroundRatio(params.BGSM_BACKGROUND_RATIO)

        # load frame sequence
        video = cv2.VideoCapture(video_path)
        # initialize frame index
        frame_number = 0
        further_droplets_list = list()
        while (1):
            # load frame
            ret, frame = video.read()
            # break loop if no frame available
            if not ret:
                print("break")
                break
            else:
                # increment and print frame index
                frame_number = frame_number + 1
                print(frame_number)

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

                if main_list[0][0] <= frame_number <= main_list[-1][0]:
                    if len(further_droplets_list) == 0:
                        droplets = []
                        droplets = detector.detect(cv2.bitwise_not(fgmask_opt))
                        # retain all candidates with given minimum radius
                        candidates = list()
                        for i in range(0, len(main_list)):
                            if main_list[i][0] == frame_number:
                                index = i
                        for droplet in droplets:
                            if droplet.size / 2.0 > 3:
                                distance = traj.getDistanceBetweenDroplets([droplet.pt[0],
                                                                            droplet.pt[1]],
                                                                           [main_list[index][1][0],
                                                                            main_list[index][1][1]])
                                if distance > 40:
                                    candidates.append(droplet)
                        if len(candidates) == 1:
                            further_droplets_list.append([frame_number, (int(candidates[0].pt[0]),
                                                                         int(candidates[0].pt[1]))])
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
                            further_droplets_list.append([frame_number,
                                                             (int(candidate_to_add.pt[0]),
                                                              int(candidate_to_add.pt[1]))])

                if main_list[0][0] <= frame_number <= main_list[-1][0]:
                    if len(further_droplets_list) > 0 and frame_number > further_droplets_list[-1][0]:
                        if len(further_droplets_list) == 1:
                            # check if temporal distance is below threshold value
                            if abs(further_droplets_list[-1][0] - frame_number) > params.TIME_BETWEEN_BLOBS:
                                pass
                            else:
                                droplets = detector.detect(cv2.bitwise_not(fgmask_opt))
                                # retain all candidates with given minimum radius
                                candidates = list()
                                for i in range(0, len(main_list)):
                                    if main_list[i][0] == frame_number:
                                        index = i
                                for droplet in droplets:
                                    if droplet.size / 2.0 > 3:
                                        last_droplet = further_droplets_list[-1][1]
                                        distance = traj.getDistanceBetweenDroplets(last_droplet,
                                                                                   [droplet.pt[0], droplet.pt[1]])
                                        # check if detected blob is within circle around first droplet position
                                        if distance < params.MAX_RADIUS_SECOND_DROPLET_MAIN:
                                            distance = traj.getDistanceBetweenDroplets([droplet.pt[0], droplet.pt[1]],
                                                                                       [main_list[index][1][0],
                                                                                        main_list[index][1][1]])
                                            if distance > 40:
                                                candidates.append(droplet)
                                if len(candidates) == 1:
                                    further_droplets_list.append([frame_number,
                                                                     (int(candidates[0].pt[0]),
                                                                      int(candidates[0].pt[1]))])
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
                                    further_droplets_list.append([frame_number,
                                                                     (int(candidate_to_add.pt[0]),
                                                                      int(candidate_to_add.pt[1]))])
                        elif 1 < len(further_droplets_list) <= 4:
                            # check if temporal distance is below threshold value
                            if abs(further_droplets_list[-1][0] - frame_number) > params.TIME_BETWEEN_BLOBS:
                                pass
                            else:
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
                                frame = cv2.line(frame, (0, int(0 * fit[0] + fit[1])),
                                                 (frame.shape[1], int(frame.shape[1] * fit[0] + fit[1])),
                                                 [255, 0, 0], 1)
                                acceptance_angle = traj.getAcceptanceAngle(further_droplets_list, frame_number)
                                mask_to_check = traj.getCylinderMask(frame, fit, distance, params.ADD_TO_SEARCH_RADIUS,
                                                                     acceptance_angle, further_droplets_list[-1][1],
                                                                     further_droplets_list)
                                frame[mask_to_check == 255] = [0, 255, 255]
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
                        cv2.imshow("Frame", frame)
                        cv2.imshow("Foreground", fgmask_opt)
                        cv2.waitKey()

        if len(further_droplets_list) > 10:
            video = cv2.VideoCapture(video_path)
            ret, frame = video.read()
            for point in further_droplets_list:
                frame = cv2.circle(frame, (int(point[1][0]), int(point[1][1])), 1, [0, 255, 255])
            cv2.imwrite(saving_path + pat + "_" + sequence_number + "_Further_Droplets.png", frame)
            file.write("Further droplets detected: yes\n")
            file.write("Sampling points of further droplets: ")
            file.write(str(further_droplets_list))
            file.write("\n")
        else:
            file.write("Further droplets detected: no\n")

    # close result file
    file.close()