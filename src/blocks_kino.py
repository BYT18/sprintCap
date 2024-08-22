#!pip install -q mediapipe
# !wget -O pose_landmarker.task -q https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2
import matplotlib.pyplot as plt
import statistics
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.signal import find_peaks


def draw_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected poses to visualize.
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]

        # Draw the pose landmarks.
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            solutions.pose.POSE_CONNECTIONS,
            solutions.drawing_styles.get_default_pose_landmarks_style())
    return annotated_image


# Function to calculate Euclidean distance
def euclidean_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def compute_slope(point1, point2):
    """
    Compute the slope of a line given two points.

    Parameters:
    point1 (tuple): The first point (x1, y1).
    point2 (tuple): The second point (x2, y2).

    Returns:
    float: The slope of the line.
    """
    x1, y1 = point1
    x2, y2 = point2
    if x2 - x1 == 0:
        return float('inf')  # Infinite slope (vertical line)
    return (y2 - y1) / (x2 - x1)


def are_parallel(line1, line2, tolerance):
    """
    Check if two lines are parallel within a given tolerance.

    Parameters:
    line1 (tuple): The first line defined by two points ((x1, y1), (x2, y2)).
    line2 (tuple): The second line defined by two points ((x3, y3), (x4, y4)).
    tolerance (float): The tolerance within which the slopes should be considered equal.

    Returns:
    bool: True if the lines are parallel within the given tolerance, False otherwise.
    """
    slope1 = compute_slope(line1[0], line1[1])
    slope2 = compute_slope(line2[0], line2[1])

    # Handle case where both slopes are infinity (vertical lines)
    if slope1 == float('inf') and slope2 == float('inf'):
        return True

    return abs(slope1 - slope2) < tolerance


# can do all the feedbakc checks in the seocnd interation fo the video using the key frames found already in the kinogram
def toe_off_feedback(plantAnk, plantKnee, plantHip, swingAnk, swingKnee, swingHip, swingToe, swingHeel, ):
    feedback = []

    # check knee bend
    if plantKnee[0] > plantAnk[0] and plantKnee[0] > plantHip[0]:
        feedback.append(
            "Noticeable knee bend in stance leg (excessive pushing = stunted swing leg recovery / exacerbating rear side mechanics)")

    # check 90 degree angle between thighs?
    # thigh_angle = compute_angle(plantKnee, (plantHip+swingHip)/2 , plantKnee)
    thigh_angle = compute_angle(plantKnee, plantHip, plantKnee)
    if thigh_angle > 100 or thigh_angle < 80:
        feedback.append("90-degree angle between thighs")

    # check toe of swing leg is vertical to knee cap
    if swingKnee[0] - swingToe[0] < 5:
        feedback.append("Toe of swing leg directly vertical to kneecap")

    # front shin parallel to rear thigh
    # compute slope between ankle and knee
    par = are_parallel((plantHip, plantKnee), (swingKnee, swingAnk), 2)
    if par:
        feedback.append("Front shin parallel to rear thigh")

    # dorsiflexion of swing foot
    # compute angle between knee, ankle, toe
    footAng = compute_angle(swingToe, swingAnk, swingKnee)
    if 75 < footAng < 105:
        feedback.append("Dorsiflexion of swing foot")

    # forearms perpindicular to each other

    # no arching back
    # check how much head x is in front of hips or shoulders?

    return feedback


def max_vert_feedback(frontAnk, frontKnee, frontHip, frontToe, backAnk, backKnee, backHip):
    # check knee bend
    feedback = []
    # check 90 degree angle between thighs?
    thigh_angle = compute_angle(frontKnee, frontHip, backKnee)
    if thigh_angle > 100 or thigh_angle < 80:
        feedback.append("90-degree angle between thighs")

    # knee angle front leg
    knee_angle = compute_angle(frontHip, frontKnee, frontAnk)
    if knee_angle > 100:
        feedback.append("Knees")

    # dorsiflexion of front foot
    # cant use heel vs toe y value since foot can be at an angle?
    footAng = compute_angle(frontToe, frontAnk, frontKnee)
    if 75 < footAng < 105:
        feedback.append("Dorsiflexion of swing foot")

    # peace?? fluidity

    # neutral head

    # straight line between rear leg knee and shoulder

    # slight cross body arm swing - neutral rotation of spine

    return feedback


def strike_feedback(frontAnk, frontKnee, frontHip, frontToe, backAnk, backKnee, backHip):
    feedback = []

    # rear femur perpendicular to ground
    if abs(backKnee[0] - backHip[0]) < 5:
        feedback.append("Femur perpendicular to ground")

    # check 20-40 degree angle between thighs
    thigh_angle = compute_angle(frontKnee, frontHip, backKnee)
    if 10 < thigh_angle < 50:
        feedback.append("20-40-degree gap between thighs")

    # front foot beginning to supinate
    footAng = compute_angle(frontKnee, frontAnk, frontToe)
    if footAng > 90:
        feedback.append("Front foot beginning to supinate ")

    # knee extension comes from reflexive action???

    # greater contraction in hammys???

    return feedback


def touch_down_feedback(plantAnk, plantKnee, plantHip, plantHeel, plantToe, swingAnk, swingKnee, swingHip, swingToe,
                        swingHeel):
    feedback = []

    # knee closeness
    knees = swingKnee[0] - plantKnee[0]
    if knees >= 0:
        feedback.append("Knees together - Faster atheletes may have swing leg slightly in front of stance leg")

    if knees < 0:
        feedback.append(
            "If knee is behind - indication of slower rotational velocities = over pushing = exacerbation of rear side mechanics")

    # perpendicular stance shin, heel underneath knee
    if (plantKnee[0] - plantAnk[0]) < 5 and (plantKnee[0] - plantHeel[0]) < 5:
        feedback.append("Shank of stance shin perpendicular to ground, heel directly underneath knee")

    # hands parallel to each other and ...

    # inital contact with outside of ball of foot / plantar flex
    if (plantToe[1] - plantHeel[1]) < 5:
        feedback.append("Plantar flex prior to touchdown")

    # allow elbow joints to open as hands pass hips

    return feedback


def full_supp_feedback(plantAnk, plantKnee, plantHip, plantToe, plantHeel, swingAnk, swingKnee, swingHip, swingToe,
                       swingHeel):
    feedback = []

    # swing foot tucked under glutes
    # check y value of heel and how close it is to hips
    if swingHeel[1] - swingHip[1] < 30:
        feedback.append("Swing foot tucked under glutes")

    # swing leg in front of hips (thigh 45 degrees from perpindicular)
    thigh_angle = compute_angle(swingAnk, swingKnee, swingHip)
    if thigh_angle < 55:
        feedback.append("Swing leg in front of hips (thigh 45 degrees from perpendicular)")

    # stance amortiation ? excess amortization
    if plantKnee[0] - plantAnk[0] > 15:
        feedback.append("Collapse at knee")

    # plantar flex prior to touchdown
    if abs(plantHeel[1] - plantToe[1]) < 5:
        feedback.append("Plantar flex prior to touchdown")

    return feedback


def compute_angle(point1, point2, point3):
    # Convert points to numpy arrays for easier calculations
    p1 = np.array(point1)
    p2 = np.array(point2)
    p3 = np.array(point3)

    # Calculate the vectors formed by the points
    v1 = p1 - p2
    v2 = p3 - p2

    # Calculate the dot product of the two vectors
    dot_product = np.dot(v1, v2)

    # Calculate the magnitudes of the vectors
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)

    # Calculate the angle between the vectors using the arccosine function
    angle = np.arccos(dot_product / (magnitude_v1 * magnitude_v2))

    # Convert angle from radians to degrees
    angle_degrees = np.degrees(angle)

    return angle_degrees


def contact_flight_analysis(frames, fps, true_fps_factor, duration):
    imp_frames = frames
    # tot_frames = dur*curr_fps(slow-mo)
    # Total frames in the original video = Frame rate × Duration = 240 fps × 3 s = 720 frames

    # tot_frames = 3*720

    # Frame rate of the GIF = Total frames of GIF / Duration = 93 frames / 3 s
    # fps = 93 / 3

    # Time per frame in GIF = Duration / Total frames of GIF = 3 s / 93 frames

    tbf = duration / (fps * true_fps_factor)

    gcontact_times = []
    flight_times = []
    counter = 0
    for i in range(len(imp_frames) - 1):
        if imp_frames[i] + 1 == imp_frames[i + 1]:
            counter += 1
            if i + 1 == len(imp_frames) - 1:
                gcontact_times.append(counter * tbf)
                flight_times.append(0)
        else:
            gcontact_times.append(counter * tbf)

            if (imp_frames[i + 1] - imp_frames[i] > counter * 1.3) and counter != 0:
                flight_times.append(
                    (tbf * (imp_frames[i + 1] - imp_frames[i])) / ((imp_frames[i + 1] - imp_frames[i]) // counter))
            else:
                flight_times.append((tbf * (imp_frames[i + 1] - imp_frames[i])))
            counter = 0

    """# Plotting the first set of bars
    bars1 = plt.bar(range(len(gcontact_times)), gcontact_times, color='red', label='Ground')

    # Plotting the second set of bars on top of the first set
    bars2 = plt.bar(range(len(flight_times)), flight_times, color='blue', label='Flight', alpha=0.5, bottom=gcontact_times)

    # Adding labels to the bars
    for bar1, bar2 in zip(bars1, bars2):
        height1 = bar1.get_height()
        height2 = bar2.get_height()
        plt.text(bar1.get_x() + bar1.get_width() / 2., height1 / 2, '{:.3f}'.format(height1), ha='center', va='center', color='white')
        plt.text(bar2.get_x() + bar2.get_width() / 2., height1 + height2 / 2, '{:.3f}'.format(height2), ha='center', va='center', color='black')


    # Adding labels and title
    plt.xlabel('Step')
    plt.ylabel('Time')
    plt.title('Stacked Bar Chart')

    # Adding legend
    plt.legend()

    # Display the plot
    plt.show()"""
    return [flight_times, gcontact_times]


# use leg length to calibrate distance to pole, then use new leg length to calibrate distance from pole in next step, this shoudl give more accurate measurments
def step_len_anal(ank_pos, frames, output_frames, leg_length_px, leg_length):
    imp_frames = frames
    print(imp_frames)
    cap_frames = []
    s_lens = []
    initial = 0
    for i in range(len(imp_frames) - 1):
        # if frames are consequtive then they are looking at the same ground contact
        if imp_frames[i] + 1 == imp_frames[i + 1] and initial == 0:
            # initial = left or right ankle position
            marker_pos = obj_detect(output_frames[imp_frames[i]], 0)
            # initial = abs(ank_pos[i][0] - marker_pos)
            cap_frames.append(imp_frames[i])
            initial = euclidean_distance(ank_pos[i], marker_pos)
            pixel_to_meter_ratio = leg_length / leg_length_px[imp_frames[i]]
            initial = initial * pixel_to_meter_ratio
            """ initial = ank_pos[i]
            init = i
            print("a")
            print(i)
            print(initial)
            print("a")"""

        elif imp_frames[i] + 1 == imp_frames[i + 1] and initial != 0:
            continue
        else:
            marker_pos = obj_detect(output_frames[imp_frames[i]], 0)
            # left_step_length_px1 = abs(abs(ank_pos[i+1][0]-marker_pos) - initial1)
            # s_lens.append(abs(abs(ank_pos[i+1][0]-marker_pos) - initial))
            cap_frames.append(imp_frames[i])
            # s_lens.append(ank_pos[i + 1][0] - initial)
            left_step_length_px = euclidean_distance(ank_pos[i + 1], marker_pos)
            """ left_step_length_px = euclidean_distance(ank_pos[i+1],initial)
            left_step_length_px = abs(ank_pos[i+1][0] - initial[0])
            fin = i
            print("b")
            print(i)
            print(ank_pos[i + 1])
            print(leg_length_px[imp_frames[i+1]])
            print(left_step_length_px)
            print("b")"""

            # idk what this is for
            if len(leg_length_px) >= imp_frames[i + 1]:
                pixel_to_meter_ratio = leg_length / leg_length_px[imp_frames[i + 1]]
                # left_step_length = (left_step_length_px / leg_length_px) * leg_length
                left_step_length = left_step_length_px * pixel_to_meter_ratio
                left_step_length = abs(left_step_length - initial)
                s_lens.append(left_step_length)

                initial = 0

    """initial = ank_pos[1]
    print("verif")
    print(imp_frames)
    print(ank_pos)
    print(initial)
    print(ank_pos[13])
    left_step_length_px = euclidean_distance(ank_pos[13],initial)
    print(left_step_length_px)
    pixel_to_meter_ratio = leg_length / 90
    # left_step_length = (left_step_length_px / leg_length_px) * leg_length
    left_step_length = left_step_length_px * pixel_to_meter_ratio
    s_lens.append(left_step_length)"""
    print(cap_frames)
    return s_lens


def step_length(height, height_pixels, distances, results, start_frame, end_frame, start_foot_coords, end_foot_coords):
    # height_pixels = abs(results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y -
    #                    results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].y)  # height in pixels
    # height_pixels = abs(475-280)
    # height_pixels = 207
    """
    meters_per_pixel = height / height_pixels
    print(meters_per_pixel)
    #pixel_distance = ((end_foot_coords[0] - start_foot_coords[0]) ** 2 + (
    #            end_foot_coords[1] - start_foot_coords[1]) ** 2) ** 0.5
    pixel_distance = (((723-581) - (538-671)) ** 2) ** 0.5
    step_length_meters = pixel_distance * meters_per_pixel
    return step_length_meters
    """
    lens = []
    for i in range(len(distances)):
        meters_per_pixel = height / height_pixels
        pixel_distance = ((distances[i]) ** 2) ** 0.5
        step_length_meters = pixel_distance * meters_per_pixel
        lens.append(step_length_meters)

    """# Plot step lengths as a bar chart
    plt.bar(range(len(lens)), lens, color='skyblue')
    plt.title('Step Lengths in Meters')
    plt.xlabel('Step Number')
    plt.ylabel('Step Length (meters)')
    plt.xticks(range(len(lens)), [f'Step {i + 1}' for i in range(len(lens))])
    plt.grid(axis='y')
    plt.show()"""

    return lens


"""
might not work if the pole is not always in frame? eg: since we are calling this whenever interested in initial position, 
there is no guarantee a pole is in frame.
"""


def obj_detect(img, temp):
    # img = cv2.imread('temptest2.png', cv2.IMREAD_GRAYSCALE)
    # Resize the image to make it bigger
    """scale_factor = 3  # Change this factor to scale the image (e.g., 2 for doubling the size)
    width = int(img.shape[1] * scale_factor)
    height = int(img.shape[0] * scale_factor)
    dim = (width, height)

    # Resize image
    img = cv2.resize(img, dim, interpolation=cv2.INTER_LINEAR)"""

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    assert img is not None, "file could not be read, check with os.path.exists()"
    img2 = img.copy()
    template = cv2.imread('./random/pole.png', cv2.IMREAD_GRAYSCALE)
    assert template is not None, "file could not be read, check with os.path.exists()"
    w, h = template.shape[::-1]

    # All the 6 methods for comparison in a list
    # methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR', 'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
    methods = ['cv2.TM_CCORR_NORMED']

    for meth in methods:
        img = img2.copy()
        method = eval(meth)

        # Apply template Matching
        res = cv2.matchTemplate(img, template, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)

        cv2.rectangle(img, top_left, bottom_right, 255, 2)
        mid = top_left[0] + (bottom_right[0] - top_left[0]) / 2
        print(mid)
        print(min_val)

        """plt.subplot(121), plt.imshow(res, cmap='gray')
        plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(img, cmap='gray')
        plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
        plt.suptitle(meth)

        plt.show()"""
        return (mid, bottom_right[1])
    # img_rgb = cv2.imread('mario.png')
    """img_rgb = img
    assert img_rgb is not None, "file could not be read, check with os.path.exists()"
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    template = cv2.imread('pole.png', cv2.IMREAD_GRAYSCALE)
    assert template is not None, "file could not be read, check with os.path.exists()"
    w, h = template.shape[::-1]

    res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.8
    loc = np.where(res >= threshold)
    mid = 0
    for pt in zip(*loc[::-1]):
        cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)
        mid = pt[0] + w / 2

    cv2.imwrite('res.png', img_rgb)
    return mid
"""


def calculate_velocity(positions, fps, ref_len_actual, ref_len_px, title):
    velocities = []
    for i in range(1, len(positions)):
        # velocity = (np.array(positions[i]) - np.array(positions[i - 1])) * fps
        # horizontal_velocity_px = (np.array(positions[i]) - np.array(positions[i - 1])) * fps
        horizontal_velocity_px = euclidean_distance(positions[i], positions[i - 1]) * fps
        velocity = (horizontal_velocity_px / ref_len_px[i]) * ref_len_actual
        velocities.append(velocity)
        # Example implementation, adjust based on your actual function
        # velocities = np.diff(positions, axis=0) * fps

    """frames_x = range(len(velocities))

    smoothed = lowess(velocities, frames_x, frac=0.3)

    # Create a scatter plot
    # plt.plot(frames_x, angles, color='blue', label='Angles')
    # Plot the results
    plt.scatter(frames_x, velocities, label='Data', color='gray')
    plt.plot(smoothed[:, 0], smoothed[:, 1], label='LOESS', color='red')
    #plt.figure(figsize=(10, 6))
    #plt.plot(frames_x, velocities, color='red', label='Velocity')
    plt.xlabel('Frame Index')
    plt.ylabel('Velocity (m/frame)')
    plt.title('Velocity of ' + title)
    plt.legend()
    plt.show()"""

    return velocities


def calculate_acceleration(velocities):
    accels = []
    for i in range(1, len(velocities)):
        acc = (velocities[i] - velocities[i - 1])
        accels.append(acc)

    return accels


# Function to calculate relative velocity
def compute_relative_velocity(previous_frame, current_frame, pixel_to_meter_ratio, fps):
    prev_keypoints = previous_frame
    curr_keypoints = current_frame

    # Calculate horizontal displacement in pixels
    displacement_pixels = abs(curr_keypoints['ankle'][0] - prev_keypoints['ankle'][0])

    # Convert displacement to meters
    displacement_meters = displacement_pixels * pixel_to_meter_ratio

    # Compute velocity (meters per second)
    velocity = displacement_meters * fps

    return velocity


# Detect thigh scissoring
def detect_scissor(values, threshold):
    """zero_crossings = []
    for i in range(1, len(y_values)):
        if y_values[i-1] >= threshold > y_values[i] or y_values[i-1] < threshold <= y_values[i]:
            zero_crossings.append(i)
    return zero_crossings"""
    local_maxima = []
    for i in range(1, len(values) - 1):
        if values[i - 1] < values[i] > values[i + 1]:
            local_maxima.append(i)
    return local_maxima


def moving_average(values, window_size):
    cumsum = np.cumsum(np.insert(values, 0, 0))
    return (cumsum[window_size:] - cumsum[:-window_size]) / float(window_size)


def find_median_of_consecutive_keys(data):
    # Sort the dictionary by keys
    sorted_keys = sorted(data.keys())
    print(data)
    print(sorted_keys)

    # Initialize variables
    median_values = []
    current_group = [sorted_keys[0]]

    # Group consecutive keys
    for i in range(1, len(sorted_keys)):
        if sorted_keys[i] == sorted_keys[i - 1] + 1:
            current_group.append(sorted_keys[i])
        else:
            # Find the median value in the current group
            median_valueY = statistics.median([data[key][1] for key in current_group])
            median_valueX = statistics.median([data[key][0] for key in current_group])
            median_values.append((median_valueX, median_valueY))
            current_group = [sorted_keys[i]]

    # Don't forget to add the last group
    if current_group:
        median_valueY = statistics.median([data[key][1] for key in current_group])
        median_valueX = statistics.median([data[key][0] for key in current_group])
        median_values.append((median_valueX, median_valueY))

    return median_values


def torso_angles(start_points, end_points, fixed_line_points):
    angles = []
    start = 0
    end = 0

    # Iterate over the fixed line points to create segments
    for i in range(len(fixed_line_points) - 1):
        # Define the fixed line vector
        vector_fixed = np.array([frame_width,
                                 fixed_line_points[i + 1][1] - fixed_line_points[i][1]])

        # for start, end in zip(start_points, end_points):
        while start_points[start][0] < fixed_line_points[i + 1][0] and end_points[end][0] < fixed_line_points[i + 1][0]:
            # Define the vector for the current segment in the list
            vector_segment = np.array(
                [end_points[end][0] - start_points[start][0], end_points[end][1] - start_points[start][1]])

            # Calculate the angle between the fixed line and the current segment
            dot_product = np.dot(vector_segment, vector_fixed)

            # Calculate the magnitudes of the vectors
            magnitude1 = np.linalg.norm(vector_segment)
            magnitude2 = np.linalg.norm(vector_fixed)

            # Calculate the cosine of the angle
            cos_theta = dot_product / (magnitude1 * magnitude2)

            # Ensure the value is within the valid range for arccos due to floating-point errors
            cos_theta = np.clip(cos_theta, -1.0, 1.0)

            # Calculate the angle in radians
            angle_radians = np.arccos(cos_theta)

            # Convert the angle to degrees
            angle_degrees = np.degrees(angle_radians)

            angles.append(angle_degrees)
            start += 1
            end += 1

    return angles


def angular_velocity(angles, title):
    # Calculate angular velocity
    # angular_velocity = [(angles[i+1] - angles[i]) for i in range(len(angles) - 1)]
    angular_velocity = angles

    # Generate x-values for angular velocity (indices start from 0 to len(angular_velocity) - 1)
    x = range(len(angular_velocity))

    # x = [(frame_index / 30) * 1000 for frame_index in frames_x]

    return angular_velocity


def standard_error(data):
    return np.std(data, ddof=1) / np.sqrt(len(data))


# Define a function to calculate the coefficient of variation
def coefficient_of_variation(data):
    mean = np.mean(data)
    if mean == 0:
        return np.nan  # Avoid division by zero
    return np.std(data, ddof=1) / mean


# Define a function to calculate the mean lag
def mean_lag(data):
    data = data.dropna()  # Drop NaN values to avoid issues
    if len(data) < 2:
        return np.nan
    lags = [j - i for i, j in zip(data[:-1], data[1:])]
    return np.mean(lags)


def get_lag(data):
    # Identify peaks in the angular velocity signal
    data = np.array(data)
    peaks, _ = find_peaks(data)

    # Compute lag (time difference) between consecutive peaks
    lags = np.diff(peaks)


def extract_consecutive_values(data):
    result = []

    # Initialize variables to store the first and last values of consecutive keys
    first_key = None
    last_key = None
    first_value = None
    last_value = None

    # Iterate through the sorted keys of the dictionary
    sorted_keys = sorted(data.keys())
    for i, key in enumerate(sorted_keys):
        value = data[key]

        # If it's the first key or the current key is not consecutive to the previous one
        if i == 0 or key != sorted_keys[i - 1] + 1:
            # If we have previously stored values, add them to the result
            if first_key is not None:
                result.append((first_key, first_value, last_key, last_value))

            # Reset for the new group of consecutive keys
            first_key = key
            first_value = value

        # Update the last_key and last_value for each iteration
        last_key = key
        last_value = value

    # Append the last group of consecutive keys after the loop
    if first_key is not None:
        result.append((first_key, first_value, last_key, last_value))

    return result


def detect_foot_contact(foot_positions, stability_threshold, min_frames):
    """
    Detects when the foot hits the ground based on position stability.

    Parameters:
        foot_positions (list of tuples): List of (x, y) tuples representing the foot position over time.
        stability_threshold (float): Maximum allowable deviation in position to consider the foot stable.
        min_frames (int): Minimum number of consecutive frames for which the foot must be stable to consider it as contact.

    Returns:
        list of tuples: List of (start_frame, end_frame) tuples representing periods of foot contact.
    """
    foot_positions = np.array(foot_positions)
    n_frames = len(foot_positions)

    contact_periods = []
    i = 0

    while i < n_frames:
        # Find the start of a potential contact period
        start_frame = i

        while i < n_frames and np.all(np.abs(foot_positions[start_frame] - foot_positions[i]) <= stability_threshold):
            i += 1

        end_frame = i - 1

        # Check if the period is long enough to be considered contact
        if end_frame - start_frame + 1 >= min_frames:
            contact_periods.append((start_frame, end_frame))

    return contact_periods


def combine_tuples_based_on_gaps(intervals):
    """
    Combine intervals if the gap between consecutive intervals is less than the average gap.

    Parameters:
        intervals (list of tuples): List of (start, end) tuples representing intervals.

    Returns:
        list of tuples: List of combined intervals.
    """
    if not intervals or len(intervals) == 2:
        return intervals

    # Sort intervals by their start time
    intervals = sorted(intervals)

    # Calculate gaps between consecutive intervals
    gaps = [intervals[i + 1][0] - intervals[i][1] for i in range(len(intervals) - 1)]

    # Calculate the average gap
    if gaps:
        average_gap = np.mean(gaps)
    else:
        average_gap = float('inf')  # If there's only one interval, there are no gaps

    combined_intervals = []
    current_start, current_end = intervals[0]

    for i in range(1, len(intervals)):
        start, end = intervals[i]

        # If the gap is less than the average, combine intervals
        if start <= current_end + average_gap:
            current_end = max(current_end, end)
        else:
            combined_intervals.append((current_start, current_end))
            current_start, current_end = start, end

    # Add the last interval
    combined_intervals.append((current_start, current_end))

    return combined_intervals

def filter_tuples_by_threshold(intervals, threshold):
    """
    Filters out tuples that start before a certain threshold.

    Parameters:
        intervals (list of tuples): List of (start, end) tuples representing intervals.
        threshold (int): The minimum start value to keep tuples.

    Returns:
        list of tuples: List of filtered intervals.
    """
    # Filter intervals based on the threshold
    filtered_intervals = [interval for interval in intervals if interval[0] >= threshold]

    return filtered_intervals

"""
Main
"""
# STEP 1: Import the necessary modules.
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Upload Video
video_path = './blocks2.mp4'
# video_path = './working_vids/timo.MOV'
output_path = 'output_video.mp4'
cap = cv2.VideoCapture(video_path)

# Get the frame rate and frame size of the video
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
median_frame = total_frames // 2

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4 files
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# variables for frame analysis
kinogram = [0]
not_found = True
max_knee_seperation = 0
output = []
max_y = 0
ground = 1000
height = 100
knee_hip_alignment_support = 100
knee_hip_alignment_strike = 100
knee_ank_alignment_support = 100

# variables for ground contacts
ground = 0
ground_contacts = []

# vars for step len
height_in_pixels = 0
ank_pos = []

# smoothness and rom
elbL_pos = []
elbR_pos = []
ankL_pos = []
ankR_pos = []
hipL_pos = []
hipR_pos = []
kneeL_pos = []
kneeL_velocities = []
kneeR_pos = []
kneeR_velocities = []
thigh_angles = []
kneeR_angles = []
mid_pelvis_pos = []
nose_pos = []
wrL_pos = []
wrR_pos = []
shouL_pos = []
shouR_pos = []
toeR_pos = []
toeL_pos = []

norm_elbR = []

prev_ankle_pos = None
prev_time = None
leg_length = 0.5  # in meters
leg_length_px = []

prev_landmarks = None
with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Get the current frame number
        frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1

        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame to get pose landmarks
        results = pose.process(rgb_frame)

        # Draw the pose annotation on the frame
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # get_kinogram_frames(frame,results,kinogram,max_knee_seperation)
            left_knee = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE]
            right_knee = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE]

            left_ankle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
            left_foot = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX]
            left_heel = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HEEL]
            right_ankle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]
            right_foot = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX]
            right_heel = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HEEL]

            left_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
            right_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]

            left_elbow = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
            # right_elbow = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
            right_elbow = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
            # print(right_elbow)

            left_wr = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
            right_wr = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]

            left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]

            noseOrg = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]

            # Convert normalized coordinates to pixel values
            kneeL = (int(left_knee.x * frame_width), int(left_knee.y * frame_height))
            kneeR = (int(right_knee.x * frame_width), int(right_knee.y * frame_height))

            ankL = (int(left_ankle.x * frame_width), int(left_ankle.y * frame_height))
            ankR = (int(right_ankle.x * frame_width), int(right_ankle.y * frame_height))
            footL = (int(left_foot.x * frame_width), int(left_foot.y * frame_height))
            footR = (int(right_foot.x * frame_width), int(right_foot.y * frame_height))
            heelL = (int(left_heel.x * frame_width), int(left_heel.y * frame_height))
            heelR = (int(right_heel.x * frame_width), int(right_heel.y * frame_height))

            hipL = (int(left_hip.x * frame_width), int(left_hip.y * frame_height))
            hipR = (int(right_hip.x * frame_width), int(right_hip.y * frame_height))
            midPelvis = (
            int(((right_hip.x + left_hip.x) / 2) * frame_width), int(((right_hip.y + left_hip.y) / 2) * frame_height))

            elbL = (int(left_elbow.x * frame_width), int(left_elbow.y * frame_height))
            elbR = (int(right_elbow.x * frame_width), int(right_elbow.y * frame_height))

            wrL = (int(left_wr.x * frame_width), int(left_wr.y * frame_height))
            wrR = (int(right_wr.x * frame_width), int(right_wr.y * frame_height))

            shouL = (int(left_shoulder.x * frame_width), int(left_shoulder.y * frame_height))
            shouR = (int(right_shoulder.x * frame_width), int(right_shoulder.y * frame_height))

            nose = (int(noseOrg.x * frame_width), int(noseOrg.y * frame_height))

            kneeL_pos.append(kneeL)
            kneeR_pos.append(kneeR)
            mid_pelvis_pos.append(midPelvis)
            nose_pos.append(nose)
            hipL_pos.append(hipL)
            hipR_pos.append(hipR)
            ankL_pos.append(ankL)
            ankR_pos.append(ankR)
            elbL_pos.append(elbL)
            elbR_pos.append(elbR)
            wrL_pos.append(wrL)
            wrR_pos.append(wrR)
            shouL_pos.append(shouL)
            shouR_pos.append(shouR)
            toeR_pos.append(footR)
            toeL_pos.append(footL)

            ground = max(footR[1], footL[1], heelR[1], heelL[1], ground)

            #max sep
            if ((kneeR[0] - kneeL[0]) ** 2 + (kneeR[1] - kneeL[1]) ** 2) ** 0.5 > max_knee_seperation:
                max_knee_seperation = ((kneeR[0] - kneeL[0]) ** 2 + (kneeR[1] - kneeL[1]) ** 2) ** 0.5
                kinogram[0] = frame_idx

            #ankle crossing knee
            """left_leg_ang = compute_angle(ankL,kneeL,hipL)
            print(frame_idx)
            print(ankR[0]-kneeL[0])
            print(left_leg_ang)
            print("")
            if abs(ankR[0]-kneeL[0]) <= 15 and left_leg_ang > 135:
                kinogram.append(frame_idx)
                #not_found = False"""


            # ground contact
            #thigh_angle = compute_angle(kneeR,midPelvis,kneeL)
            #and thigh_angle < 100

            # back foot the knee is less than some threshold behind the corresponding hip
            #if (abs(footL[1] - ground) < 15 or abs(footR[1] - ground) < 15) and (euclidean_distance(wrR,hipR)<25 or euclidean_distance(wrL,hipL)<25):  #add additional factor that accoutns for thigh angles or knee closness
            #    ground_contacts.append(frame_idx)  # take lowest point of the ground points in the group

            #first identify the touch downs, then inbetween each touch down identify the cross and full extension which is independnet of the maxs between the cycles

        # Write the frame to the output video
        out.write(frame)
        output.append(frame)

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

front_leg_off_block = 0
for i in range(len(ankR_pos)):
    if ankR_pos[i][0] - toeR_pos[i][0] > 0:
        kinogram.append(i)
        print(i)
        break

for i in range(len(ankL_pos)):
    if ankL_pos[i][0] - toeL_pos[i][0] > 0 and ankR_pos[i][0] - ankL_pos[i][0] > 30:
        kinogram.append(i)
        print(i)
        front_leg_off_block = i
        break

"""imp_frames = kinogram
num_frames = len(imp_frames)
fig, axs = plt.subplots(1, num_frames, figsize=(num_frames * 5, 5))

for i in range(num_frames):
    axs[i].imshow(output[imp_frames[i]])
    axs[i].axis('off')
    axs[i].set_title(f"Contact {i + 1}")

plt.tight_layout()
plt.show()"""

# Parameters
stability_threshold = 7  # Allowable deviation in position (x or y)
min_frames = 10  # Minimum number of consecutive frames

# Detect foot contact
contact_periods_R = detect_foot_contact(toeR_pos, stability_threshold, min_frames)
contact_periods_L = detect_foot_contact(toeL_pos, stability_threshold, min_frames)

# Print contact periods
print("Right Foot contact periods (start_frame, end_frame):", contact_periods_R)
print("Left Foot contact periods (start_frame, end_frame):", contact_periods_L)

contact_periods_R = filter_tuples_by_threshold(contact_periods_R, front_leg_off_block)
contact_periods_L = filter_tuples_by_threshold(contact_periods_L, front_leg_off_block)
print(contact_periods_R)
print(contact_periods_L)

contact_periods_R = combine_tuples_based_on_gaps(contact_periods_R)
contact_periods_L = combine_tuples_based_on_gaps(contact_periods_L)

print("Combined intervals:", contact_periods_R)
print("Combined intervals:", contact_periods_L)

"""imp_frames = [182, 192, 201, 235, 298, 326]
num_frames = len(imp_frames)
fig, axs = plt.subplots(1, num_frames, figsize=(num_frames * 5, 5))

for i in range(num_frames):
    axs[i].imshow(output[imp_frames[i]])
    axs[i].axis('off')
    axs[i].set_title(f"Contact {i + 1}")

plt.tight_layout()
plt.show()"""

for i in contact_periods_R:
    print(i)
    for j in range(i[0], i[1]):
        right_leg_ang = compute_angle(ankR_pos[j],kneeR_pos[j],hipR_pos[j])
        if ankL_pos[j][0]-kneeR_pos[j][0] >= 0 and right_leg_ang > 120:
            kinogram.append(j)
            break

for i in contact_periods_L:
    for j in range(i[0], i[1]):
        left_leg_ang = compute_angle(ankL_pos[j], kneeL_pos[j], hipL_pos[j])
        if ankR_pos[j][0] - kneeL_pos[j][0] >= 0 and left_leg_ang > 120:
            kinogram.append(j)
            break

#imp_frames = [136, 186, 243, 273, 353, 376]
imp_frames = kinogram
#imp_frames = [136, 186, 243, 280, 343, 376, 198, 233, 298, 326]
num_frames = len(imp_frames)

# Increase figure size for better clarity
fig, axs = plt.subplots(1, num_frames, figsize=(num_frames * 8, 8))  # Adjusted figsize

for i in range(num_frames):
    axs[i].imshow(output[imp_frames[i]])
    axs[i].axis('off')
    axs[i].set_title(f"Contact {i + 1}")

plt.tight_layout()

# Save the figure with higher resolution
plt.savefig('imp_frames.png', dpi=300)  # Adjust the dpi for higher resolution
plt.show()