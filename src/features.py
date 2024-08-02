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
import numpy as np
import math
from typing import Tuple
import os

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


def tricubic(x):
    y = np.zeros_like(x)
    idx = (x >= -1) & (x <= 1)
    y[idx] = np.power(1.0 - np.power(np.abs(x[idx]), 3), 3)
    return y

class Loess(object):

    @staticmethod
    def normalize_array(array: np.ndarray) -> Tuple[np.ndarray, float, float]:
        min_val = np.min(array)
        max_val = np.max(array)
        return (array - min_val) / (max_val - min_val), min_val, max_val

    def __init__(self,
                 xx: np.ndarray,
                 yy: np.ndarray,
                 degree: int = 1):
        self.n_xx, self.min_xx, self.max_xx = self.normalize_array(xx)
        self.n_yy, self.min_yy, self.max_yy = self.normalize_array(yy)
        self.degree = degree

    @staticmethod
    def get_min_range(distances: np.ndarray,
                      window: int) -> np.ndarray:
        min_idx = np.argmin(distances)
        n = len(distances)
        if min_idx == 0:
            return np.arange(0, window)
        if min_idx == n - 1:
            return np.arange(n - window, n)

        min_range = [min_idx]
        while len(min_range) < window:
            i0 = min_range[0]
            i1 = min_range[-1]
            if i0 == 0:
                min_range.append(i1 + 1)
            elif i1 == n - 1:
                min_range.insert(0, i0 - 1)
            elif distances[i0 - 1] < distances[i1 + 1]:
                min_range.insert(0, i0 - 1)
            else:
                min_range.append(i1 + 1)
        return np.array(min_range)

    @staticmethod
    def get_weights(distances: np.ndarray,
                    min_range: np.ndarray) -> np.ndarray:
        max_distance = np.max(distances[min_range])
        weights = tricubic(distances[min_range] / max_distance)
        return weights

    def normalize_x(self, value: float) -> float:
        return (value - self.min_xx) / (self.max_xx - self.min_xx)

    def denormalize_y(self, value: float) -> float:
        return value * (self.max_yy - self.min_yy) + self.min_yy

    def estimate(self,
                 x: float,
                 window: int,
                 use_matrix: bool = False,
                 degree: int = 1) -> float:
        n_x = self.normalize_x(x)
        distances = np.abs(self.n_xx - n_x)
        min_range = self.get_min_range(distances, window)
        weights = self.get_weights(distances, min_range)

        if use_matrix or degree > 1:
            wm = np.multiply(np.eye(window), weights)
            xm = np.ones((window, degree + 1))

            xp = np.array([[math.pow(n_x, p)] for p in range(degree + 1)])
            for i in range(1, degree + 1):
                xm[:, i] = np.power(self.n_xx[min_range], i)

            ym = self.n_yy[min_range]
            xmt_wm = np.transpose(xm) @ wm
            beta = np.linalg.pinv(xmt_wm @ xm) @ xmt_wm @ ym
            y = (beta @ xp)[0]
        else:
            xx = self.n_xx[min_range]
            yy = self.n_yy[min_range]
            sum_weight = np.sum(weights)
            sum_weight_x = np.dot(xx, weights)
            sum_weight_y = np.dot(yy, weights)
            sum_weight_x2 = np.dot(np.multiply(xx, xx), weights)
            sum_weight_xy = np.dot(np.multiply(xx, yy), weights)

            mean_x = sum_weight_x / sum_weight
            mean_y = sum_weight_y / sum_weight

            b = (sum_weight_xy - mean_x * mean_y * sum_weight) / \
                (sum_weight_x2 - mean_x * mean_x * sum_weight)
            a = mean_y - b * mean_x
            y = a + b * n_x
        return self.denormalize_y(y)

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
            if (imp_frames[i + 1] - imp_frames[i] > counter * 1.3):
                flight_times.append(
                    (tbf * (imp_frames[i + 1] - imp_frames[i])) / ((imp_frames[i + 1] - imp_frames[i]) // counter))
            else:
                flight_times.append((tbf * (imp_frames[i + 1] - imp_frames[i])))
            counter = 0

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

        elif imp_frames[i] + 1 == imp_frames[i + 1] and initial != 0:
            continue
        else:
            marker_pos = obj_detect(output_frames[imp_frames[i]], 0)
            # left_step_length_px1 = abs(abs(ank_pos[i+1][0]-marker_pos) - initial1)
            # s_lens.append(abs(abs(ank_pos[i+1][0]-marker_pos) - initial))
            cap_frames.append(imp_frames[i])
            # s_lens.append(ank_pos[i + 1][0] - initial)
            left_step_length_px = euclidean_distance(ank_pos[i + 1], marker_pos)

            # idk what this is for
            if len(leg_length_px) >= imp_frames[i + 1]:
                pixel_to_meter_ratio = leg_length / leg_length_px[imp_frames[i + 1]]
                # left_step_length = (left_step_length_px / leg_length_px) * leg_length
                left_step_length = left_step_length_px * pixel_to_meter_ratio
                left_step_length = abs(left_step_length - initial)
                s_lens.append(left_step_length)

                initial = 0

    print(cap_frames)
    return s_lens


def step_length(height, height_pixels, distances, results, start_frame, end_frame, start_foot_coords, end_foot_coords):

    lens = []
    for i in range(len(distances)):
        meters_per_pixel = height / height_pixels
        pixel_distance = ((distances[i]) ** 2) ** 0.5
        step_length_meters = pixel_distance * meters_per_pixel
        lens.append(step_length_meters)

    return lens


"""
might not work if the pole is not always in frame? eg: since we are calling this whenever interested in initial position, 
there is no guarantee a pole is in frame.
"""

def obj_detect(img, temp):

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    assert img is not None, "file could not be read, check with os.path.exists()"
    img2 = img.copy()
    template = cv2.imread('pole.png', cv2.IMREAD_GRAYSCALE)
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

        return (mid, bottom_right[1])

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

    local_maxima = []
    for i in range(1, len(values) - 1):
        if values[i - 1] < values[i] > values[i + 1]:
            local_maxima.append(i)
    return local_maxima[1::2]


def moving_average(values, window_size):
    cumsum = np.cumsum(np.insert(values, 0, 0))
    return (cumsum[window_size:] - cumsum[:-window_size]) / float(window_size)


def find_median_of_consecutive_keys(data):
    # Sort the dictionary by keys
    sorted_keys = sorted(data.keys())

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


def torso_angles(start_points, end_points, fixed_line_points, frame_width):
    angles = []
    start = 0
    end = 0

    # Iterate over the fixed line points to create segments for the ground
    for i in range(len(fixed_line_points) - 1):
        # Define the fixed line vector
        vector_fixed = np.array([frame_width,
                                 fixed_line_points[i + 1][1] - fixed_line_points[i][1]])
        # while the body is in between the x-coord of the ground line, compute the angle
        # for start, end in zip(start_points, end_points):
        while start_points[start][0] < fixed_line_points[i + 1][0] and end_points[end][0] < fixed_line_points[i + 1][0] and start < len(start_points)-1 and end < len(end_points)-1:
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


def angular_velocity(angles,fps):
    # Calculate angular velocity
    angular_velocity = [(angles[i + 1] - angles[i]) * fps for i in range(len(angles) - 1)]
    #angular_velocity = [(angles[i+1] - angles[i]) for i in range(len(angles) - 1)]
    #angular_velocity = angles

    # Generate x-values for angular velocity (indices start from 0 to len(angular_velocity) - 1)
    frames_x = range(len(angular_velocity))

    x = [(frame_index / fps) * 1000 for frame_index in frames_x]

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


def get_lag(data1, data2):
    # Identify peaks in the angular velocity signal
    #data = np.array(data)
    #peaks, _ = find_peaks(data)

    # Compute lag (time difference) between consecutive peaks
    #lags = np.diff(peaks)

    # Identifying peaks (indices of maximum values)
    peak_index1 = np.argmax(data1)
    peak_index2 = np.argmax(data2)

    # Difference in indices (time difference in terms of data points)
    peak_difference = peak_index1 - peak_index2

    # peak of movement 1 - peak of movement 2 divided by cycle length
    # if negative, 1- (aka if first peak is smaller than seconds, then take 1 - it
    if peak_difference < 0:
        peak_difference = 1 - peak_difference

    # Assuming cycle_length is known or calculated, for example:
    cycle_length = len(data1)  # Total length of the data

    # Calculating lag
    lag = peak_difference / cycle_length

    return lag


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

def spectral_arclength(movement, fs, padlevel=4, fc=10.0, amp_th=0.05):
    """
    Calcualtes the smoothness of the given speed profile using the modified spectral
    arc length metric.

    Parameters
    ----------
    movement : np.array
               The array containing the movement speed profile.
    fs       : float
               The sampling frequency of the data.
    padlevel : integer, optional
               Indicates the amount of zero padding to be done to the movement
               data for estimating the spectral arc length. [default = 4]
    fc       : float, optional
               The max. cut off frequency for calculating the spectral arc
               length metric. [default = 10.]
    amp_th   : float, optional
               The amplitude threshold to used for determing the cut off
               frequency upto which the spectral arc length is to be estimated.
               [default = 0.05]

    Returns
    -------
    sal      : float
               The spectral arc length estimate of the given movement's
               smoothness.
    (f, Mf)  : tuple of two np.arrays
               This is the frequency(f) and the magntiude spectrum(Mf) of the
               given movement data. This spectral is from 0. to fs/2.
    (f_sel, Mf_sel) : tuple of two np.arrays
                      This is the portion of the spectrum that is selected for
                      calculating the spectral arc length.

    Notes
    -----
    This is the modfieid spectral arc length metric, which has been tested only
    for discrete movements.
    It is suitable for movements that are a few seconds long, but for long
    movements it might be slow and results might not make sense (like any other
    smoothness metric).

    Examples
    --------
    >>> t = np.arange(-1, 1, 0.01)
    >>> move = np.exp(-5*pow(t, 2))
    >>> sal, _, _ = spectral_arclength(move, fs=100.)
    >>> '%.5f' % sal
    '-1.41403'

    """
    if np.isnan(movement).any():
        # Log or handle the presence of NaN values
        print("Skipping computation due to NaN values in the data.")
        return None, None, None  # or other default/placeholder values

    # Number of zeros to be padded.
    nfft = int(pow(2, np.ceil(np.log2(len(movement))) + padlevel))

    # Frequency
    f = np.arange(0, fs, fs/nfft)
    # Normalized magnitude spectrum
    Mf = abs(np.fft.fft(movement, nfft))
    Mf = Mf/max(Mf)

    # Indices to choose only the spectrum within the given cut off frequency Fc.
    # NOTE: This is a low pass filtering operation to get rid of high frequency
    # noise from affecting the next step (amplitude threshold based cut off for
    # arc length calculation).
    fc_inx = ((f <= fc)*1).nonzero()
    f_sel = f[fc_inx]
    Mf_sel = Mf[fc_inx]

    # Choose the amplitude threshold based cut off frequency.
    # Index of the last point on the magnitude spectrum that is greater than
    # or equal to the amplitude threshold.
    inx = ((Mf_sel >= amp_th)*1).nonzero()[0]
    fc_inx = range(inx[0], inx[-1]+1)
    f_sel = f_sel[fc_inx]
    Mf_sel = Mf_sel[fc_inx]

    # Calculate arc length
    new_sal = -sum(np.sqrt(pow(np.diff(f_sel)/(f_sel[-1] - f_sel[0]), 2) +
                           pow(np.diff(Mf_sel), 2)))
    return new_sal, (f, Mf), (f_sel, Mf_sel)

def se_loop(ran,title):
    se_min={}
    for col, mins in ran:
        n = len(mins)  # number of observations
        if n > 1:  # Ensure there is more than one value to calculate standard deviation and SE
            mean_min = np.mean(mins)
            std_dev = np.std(mins, ddof=1)  # Sample standard deviation
            se = std_dev / np.sqrt(n)  # Standard Error
            se_min[f'se_{title}_{col}'] = se
        else:
            se_min[f'se_{title}_{col}'] = 0  # or handle the case as needed (e.g., 0, None)
    return se_min

def process_videos_in_folder(folder_path):
    # Get a list of video files in the folder
    video_files = [f for f in os.listdir(folder_path) if f.endswith(('.mp4', '.avi', '.mov','.MOV'))]
    i = 0
    for video_file in video_files:
        video_path = os.path.join(folder_path, video_file)
        print(f"Processing {video_path}")
        extract(video_path, str(i))
        i+=1
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
# video_path = 'adam.mov'
def extract(path,title_count):
    #video_path = './working_vids/marie.mp4'
    video_path = path
    pb = 0
    level_of_expertise = "novice"
    sex = 'F'

    output_path = 'output_'+title_count+'.mp4'
    cap = cv2.VideoCapture(video_path)

    # Get the frame rate and frame size of the video
    fps = cap.get(cv2.CAP_PROP_FPS)
    # Get total number of frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Calculate duration in seconds
    duration_seconds = total_frames / fps

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4 files
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # variables for frame analysis
    kinogram = [0, 0, 0, 0, 0]
    max_knee_seperation = 0
    output = []
    max_y = 0
    ground = 1000
    height = 100
    knee_hip_alignment_support = 100
    knee_hip_alignment_strike = 100
    knee_ank_alignment_support = 100

    # variables for ground contacts
    ground_points = {}
    ground_frames = []
    ground_contacts = 0
    # threshold = 5  # Threshold for considering significant movement (in pixels)
    # prev_left_foot = None
    # prev_right_foot = None

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
    backside = []
    frontside = []
    toeR_pos = []
    toeL_pos = []

    norm_elbR = []

    prev_ankle_pos = None
    prev_time = None
    leg_length = 0.5  # in meters
    leg_length_px = []

    # Parameters for lucas kanade optical flow
    # lk_params = dict(winSize=(15, 15), maxLevel=2,criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    prev_landmarks = None
    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
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
                toeL_pos.append(footL)
                toeR_pos.append(footR)


                if (ankR[0] > midPelvis[0]):
                    frontside.append(ankR)
                else:
                    backside.append(ankR)

                # Calculate leg length in pixels
                # if abs(hipL[0] - ankL[0])<10 and abs(hipL[0] - kneeL[0])<10:
                leg_length_px.append(euclidean_distance(hipR, kneeR))
                # print(leg_length_px)
                # print(results.pose_world_landmarks.landmark)

                prev_ankle_pos = ankL
                prev_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

                tAng = compute_angle(hipR, kneeR, ankR)
                kneeR_angles.append(tAng)

                tAng = compute_angle(kneeL, midPelvis, kneeR)
                thigh_angles.append(tAng)

                # toeoff
                # max thigh seperation
                # works for toe off, could also do max distance between ankle and opposite knee
                # to get different feet/sides, just check which foot is in front of pelvis
                point1 = ankL
                point2 = kneeL
                point3 = hipL
                angle = compute_angle(point1, point2, point3)

                # if ((kneeR[0] - kneeL[0]) ** 2 + (kneeR[1] - kneeL[1]) ** 2) ** 0.5 > max_knee_seperation and footL[1]>ground:
                if ((kneeR[0] - kneeL[0]) ** 2 + (kneeR[1] - kneeL[1]) ** 2) ** 0.5 > max_knee_seperation and footL[
                    1] + 5 > ground:
                    max_knee_seperation = ((kneeR[0] - kneeL[0]) ** 2 + (kneeR[1] - kneeL[1]) ** 2) ** 0.5
                    # key_frames = frame_idx
                    kinogram[0] = frame_idx

                # max proj / vert
                # if abs(kneeL[0] - kneeR[0]) > 30 and abs(footL[1] - footR[1]) < height and abs(kneeL[1] - ankL[1]) < 50 and abs(kneeR[0] - ankR[0]) < 50:
                if abs(kneeL[0] - kneeR[0]) > 30 and abs(footL[1] - footR[1]) < height:
                    height = abs(footL[1] - footR[1])
                    kinogram[1] = frame_idx

                # full support left
                # should also check foot is on ground
                # possible check ankle score too
                if abs(ankL[0] - midPelvis[0]) < knee_hip_alignment_support and ankR[1] < ankL[1]:
                    knee_hip_alignment_support = abs(ankL[0] - midPelvis[0])
                    knee_ank_alignment_support = abs(kneeL[0] - ankL[0])
                    pix = footL[1] - nose[1]
                    if pix > height_in_pixels:
                        height_in_pixels = pix
                    kinogram[4] = frame_idx

                # strike R
                # opposite leg is close to underneath COM and strike foot is infront of body and opposite foot higher than strike knee
                if abs(kneeL[0] - hipL[0]) < knee_hip_alignment_strike and ankL[1] + 15 < ground and ankR[0] > hipR[0]:
                    # and ankL[1]<kneeR[1]
                    knee_hip_alignment_strike = abs(kneeL[0] - hipL[0])
                    kinogram[2] = frame_idx

                # touch down
                if ((ankR[1] - ankL[1]) ** 2) ** 0.5 > max_y:
                    max_y = ((ankR[1] - ankL[1]) ** 2) ** 0.5
                    kinogram[3] = frame_idx

                # if abs(kneeL[0] - kneeR[0]) < 10 or (abs(elbL[0] - elbR[0]) < 10 and abs(kneeL[0] - kneeR[0]) < 25):
                if abs(kneeL[0] - kneeR[0]) < 25 and (abs(footL[1] - heelL[1]) < 10 or abs(footR[1] - heelR[1]) < 10):
                    ground_contacts += 1
                    if footL[1] > footR[1]:
                        ground = footL[1]
                        g1 = footL
                    else:
                        ground = footR[1]
                        g1 = footR
                    # ground = max(footL[1], footR[1])
                    ground_frames.append(frame_idx)  # take lowest point of the ground points in the group
                    ground_points[frame_idx] = g1

            prev_frame = frame.copy()

            # Write the frame to the output video
            #out.write(frame)
            #output.append(frame)

        # Release resources
        #cap.release()
        #out.release()
        cv2.destroyAllWindows()

    # looking at all knee positions, extract only the ones that are about to cross
    left_knee_crossings = [i for i in range(1, len(kneeL_pos)) if
                           (kneeL_pos[i - 1] < mid_pelvis_pos[i] and kneeL_pos[i] > mid_pelvis_pos[i]) or (
                                       kneeL_pos[i - 1] > mid_pelvis_pos[i] and kneeL_pos[i] < mid_pelvis_pos[i])]
    right_knee_crossings = [i for i in range(1, len(kneeR_pos)) if
                            (kneeR_pos[i - 1] < mid_pelvis_pos[i] and kneeR_pos[i] > mid_pelvis_pos[i]) or (
                                        kneeR_pos[i - 1] > mid_pelvis_pos[i] and kneeR_pos[i] < mid_pelvis_pos[i])]

    # variables for kinogram feedback
    feedback = []

    # variables for ground point contact frames
    imp_frames = []
    contact = False
    threshold = 100000
    #print("ground")
    #print(ground_points)
    # Sort the dictionary by keys
    ground_points_smooth = find_median_of_consecutive_keys(ground_points)
    #print(ground_points_smooth)

    cap = cv2.VideoCapture(video_path)
    # Get the frame rate and frame size of the video
    fps = cap.get(cv2.CAP_PROP_FPS)
    #print("fps " + str(fps))
    #print(ground_frames)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4 files
    # out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # Parameters for lucas kanade optical flow
    # lk_params = dict(winSize=(15, 15), maxLevel=2,criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    """
    Determine Ground Contacts
    """
    prev_landmarks = None
    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.3, min_tracking_confidence=0.3) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Get the current frame number
            frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1

            if frame_idx in ground_frames:
                contact = True
                threshold = ground_points[frame_idx][1]
                # print("thresh")

            # Convert the BGR image to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the frame to get pose landmarks
            results = pose.process(rgb_frame)

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

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
                right_elbow = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]

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
                    int(((right_hip.x + left_hip.x) / 2) * frame_width),
                    int(((right_hip.y + left_hip.y) / 2) * frame_height))

                elbL = (int(left_elbow.x * frame_width), int(left_elbow.y * frame_height))
                elbR = (int(right_elbow.x * frame_width), int(right_elbow.y * frame_height))

                if frame_idx in kinogram:
                    ind = kinogram.index(frame_idx)
                    if ind == 0:
                        if ankL[1] > ankR[1]:
                            f = toe_off_feedback(ankL, kneeL, hipL, ankR, kneeR, hipR, footR, heelR)
                        else:
                            f = toe_off_feedback(ankR, kneeR, hipR, ankL, kneeL, hipL, footL, heelL)
                        # print("Toe off")
                        # print(f)
                    elif ind == 1:
                        if ankL[0] > ankR[0]:
                            f = max_vert_feedback(ankL, kneeL, hipL, footL, ankR, kneeR, hipR)
                        else:
                            f = max_vert_feedback(ankR, kneeR, hipR, footR, ankL, kneeL, hipL)
                        # print("MV")
                        # print(f)
                    elif ind == 2:
                        if ankL[0] > ankR[0]:
                            f = strike_feedback(ankL, kneeL, hipL, footL, ankR, kneeR, hipR)
                        else:
                            f = strike_feedback(ankR, kneeR, hipR, footR, ankL, kneeL, hipL)
                        # print("Strike")
                        # print(f)
                    elif ind == 3:
                        if ankL[0] > ankR[0]:
                            f = touch_down_feedback(ankL, kneeL, hipL, heelL, footL, ankR, kneeR, hipR, footR, heelR)
                        else:
                            f = touch_down_feedback(ankR, kneeR, hipR, heelR, footR, ankL, kneeL, hipL, footL, heelL)
                        # print("TD")
                        # print(f)
                    else:
                        if ankL[0] > ankR[0]:
                            f = full_supp_feedback(ankL, kneeL, hipL, footL, heelL, ankR, kneeR, hipR, footR, heelR)
                        else:
                            f = full_supp_feedback(ankR, kneeR, hipR, ankL, footR, heelR, kneeL, hipL, footL, heelL)
                        # print("FS")
                        # print(f)

            # Draw the pose annotation on the frame
            if results.pose_landmarks and contact == True:
                # print(frame_idx)
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
                right_elbow = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]

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
                midPelvis = (
                int(((right_hip.x + left_hip.x) / 2) * frame_width), int(((right_hip.y + left_hip.y) / 2) * frame_height))

                elbL = (int(left_elbow.x * frame_width), int(left_elbow.y * frame_height))
                elbR = (int(right_elbow.x * frame_width), int(right_elbow.y * frame_height))

                if frame_idx in imp_frames:
                    # print('false1')
                    contact = False
                else:

                    # ensure hand position is sufficiently far (this will ensure that no stoppages occur when knees are close
                    # left foot is on the ground
                    if (ankR[1] - ankL[1] <= 0):
                        # checking if the distance between knees does not change enough then that means we are at toe off frame
                        if abs(footL[1] - threshold) < 10:
                            imp_frames.append(frame_idx)
                            ank_pos.append(ankL)
                        else:
                            contact = False
                    else:
                        if abs(footR[1] - threshold) < 10:
                            imp_frames.append(frame_idx)
                            ank_pos.append(ankR)
                        else:
                            contact = False

        # Release resources
        cap.release()
        # out.release()
        cv2.destroyAllWindows()

    """
    Velocity and Smoothness Analysis 
    """
    # Calculate velocities
    """velocitiesL = calculate_velocity(kneeL_pos, fps*8, leg_length,leg_length_px,"Left Knee")
    velocitiesR = calculate_velocity(kneeR_pos, fps*8, leg_length,leg_length_px,"Right Knee")
    v_elb_R = calculate_velocity(elbR_pos, fps*8, leg_length,leg_length_px,"Right Elbow")
    v_ank_R = calculate_velocity(ankR_pos, fps*8, leg_length,leg_length_px,"Right Elbow")
    v_hip_R = calculate_velocity(hipR_pos, fps*8, leg_length,leg_length_px,"Right Elbow")
    
    frames_x = range(len(velocitiesR))
    
    smoothed = lowess(v_hip_R, frames_x, frac=0.3)
    smoothed2 = lowess(velocitiesR, frames_x, frac=0.3)
    smoothed3 = lowess(v_ank_R, frames_x, frac=0.3)
    
    # Create a scatter plot
    # plt.plot(frames_x, angles, color='blue', label='Angles')
    # Plot the results
    #plt.scatter(frames_x, velocitiesL, label='Data', color='gray')
    plt.plot(smoothed[:, 0], smoothed[:, 1], label='LOESS Hip', color='red')
    plt.plot(smoothed2[:, 0], smoothed2[:, 1], label='LOESS Knee', color='blue')
    plt.plot(smoothed3[:, 0], smoothed3[:, 1], label='LOESS Ankle', color='green')
    #plt.figure(figsize=(10, 6))
    #plt.plot(frames_x, velocities, color='red', label='Velocity')
    plt.xlabel('Frame Index')
    plt.ylabel('Velocity (m/frame)')
    plt.title('Velocity of Leg Components')
    plt.legend()
    plt.show()"""

    """ratios = []
    for v1, v2 in zip(velocitiesR, velocitiesL):
        if v2 == 0:
            ratios.append(float('inf'))  # Handle division by zero
        else:
            ratios.append(v1 / v2)
    
    plt.plot(range((len(ratios))), ratios, label='Knee Velocity Ratio', color='orange')
    plt.xlabel('Frame Index')
    plt.ylabel('Velocity (units/s)')
    plt.title('Velocity Ratio Comparison of Left and Right Knees')
    
    plt.show()"""

    # Compute velocity magnitudes
    # since velocities are in x and y direction, need to reduce to scalar that does not have direction
    # velocity_magnitudeL = np.linalg.norm(velocitiesL, axis=1)
    # velocity_magnitudeR = np.linalg.norm(velocitiesR, axis=1)

    # Create time axis
    # time_velocity = np.arange(len(velocity_magnitudeL)) / fps

    # Plot velocities on the same axis
    """plt.figure(figsize=(10, 6))
    plt.plot(time_velocity, velocity_magnitudeL, label='Left Knee Velocity')
    plt.plot(time_velocity, velocity_magnitudeR, label='Right Knee Velocity', color='orange')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (units/s)')
    plt.title('Velocity Comparison of Left and Right Knees')
    plt.legend()"""
    # plt.show()
    pos_data = {
        'P_R_knee': kneeR_pos,
        'P_L_knee': kneeL_pos,
        'P_R_hip': hipR_pos,
        'P_L_hip': hipL_pos,
        'P_pelv': mid_pelvis_pos,
        'P_R_elbow': elbR_pos,
        # 'P_L_elbow': elbL_pos,
        'P_R_shou': shouR_pos,
        # 'P_L_shou': shouL_pos,
        'P_R_ankle': ankR_pos,
        'P_R_wr': wrR_pos,
        # 'P_L_ankle': ankL_pos,
        'P_R_toe':toeR_pos,
        'P_head': nose_pos,
    }

    smoothed_data = {}

    for key, value in pos_data.items():
        # Split the tuples into two lists: one for x coordinates, one for y coordinates
        x_values, y_values = zip(*value)

        # Create Loess objects for x and y values
        loess_x = Loess(range(len(x_values)), list(x_values))
        loess_y = Loess(range(len(y_values)), list(y_values))

        # Smooth x and y values separately
        smoothed_x = [loess_x.estimate(i, window=12, use_matrix=False, degree=3) for i in range(len(x_values))]
        smoothed_y = [loess_y.estimate(i, window=12, use_matrix=False, degree=3) for i in range(len(y_values))]

        # Combine smoothed x and y values back into tuples
        smoothed_positions = list(zip(smoothed_x, smoothed_y))

        # Store the smoothed data in the dictionary
        smoothed_data[key] = smoothed_positions


    # Angular velocities
    thigh_angles = []
    ta_rough = []
    for i in range(len(smoothed_data["P_R_knee"])):
        a = compute_angle(smoothed_data["P_R_knee"][i], smoothed_data["P_pelv"][i], smoothed_data["P_L_knee"][i])
        thigh_angles.append(a)
        a2 = compute_angle(kneeR_pos[i], mid_pelvis_pos[i], kneeL_pos[i])
        ta_rough.append(a2)
    AV_thigh = angular_velocity(thigh_angles, fps)
    #AV_thigh = angular_velocity(thigh_angles,fps)

    kneeL_angles = []
    for i in range(len(kneeL_pos)):
        a = compute_angle(hipL_pos[i], kneeL_pos[i], ankL_pos[i])
        kneeL_angles.append(a)
    AV_L_knee = angular_velocity(kneeL_angles, fps)

    kneeR_angles = []
    for i in range(len(smoothed_data['P_R_knee'])):
        a = compute_angle(smoothed_data['P_R_hip'][i], smoothed_data['P_R_knee'][i], smoothed_data['P_R_ankle'][i])
        kneeR_angles.append(a)
    AV_R_knee = angular_velocity(kneeR_angles, fps)

    elbL_angles = []
    for i in range(len(elbL_pos)):
        a = compute_angle(shouL_pos[i], elbL_pos[i], wrL_pos[i])
        elbL_angles.append(a)
    AV_L_elb = angular_velocity(elbL_angles, fps)

    elbR_angles = []
    for i in range(len(smoothed_data['P_R_elbow'])):
        a = compute_angle(smoothed_data['P_R_shou'][i], smoothed_data['P_R_elbow'][i], smoothed_data['P_R_wr'][i])
        elbR_angles.append(a)
    AV_R_elb = angular_velocity(elbR_angles, fps)

    shouL_angles = []
    for i in range(len(shouL_pos)):
        a = compute_angle(elbL_pos[i], shouL_pos[i], hipL_pos[i])
        shouL_angles.append(a)
    AV_L_shou = angular_velocity(shouL_angles, fps)

    shouR_angles = []
    for i in range(len(smoothed_data['P_R_shou'])):
        a = compute_angle(smoothed_data['P_R_elbow'][i], smoothed_data['P_R_shou'][i], smoothed_data['P_R_hip'][i])
        shouR_angles.append(a)
    AV_R_shou = angular_velocity(shouR_angles, fps)

    hipR_angles = []
    for i in range(len(smoothed_data['P_R_hip'])):
        a = compute_angle(smoothed_data['P_R_shou'][i], smoothed_data['P_R_hip'][i], smoothed_data['P_R_knee'][i])
        hipR_angles.append(a)
    AV_R_hip = angular_velocity(hipR_angles, fps)

    hipL_angles = []
    for i in range(len(hipL_pos)):
        a = compute_angle(shouL_pos[i], hipL_pos[i], kneeL_pos[i])
        hipL_angles.append(a)
    AV_L_hip = angular_velocity(hipL_angles, fps)

    A_R_ank = []
    for i in range(len(smoothed_data['P_R_ankle'])):
        a = compute_angle(smoothed_data['P_R_knee'][i], smoothed_data['P_R_ankle'][i], smoothed_data['P_R_toe'][i])
        A_R_ank.append(a)
    AV_R_ank = angular_velocity(A_R_ank, fps)

    A_L_ank = []
    for i in range(len(ankL_pos)):
        a = compute_angle(kneeL_pos[i], ankL_pos[i], toeL_pos[i])
        A_L_ank.append(a)
    AV_L_ank = angular_velocity(A_L_ank, fps)

    tor_angles = torso_angles(smoothed_data['P_head'], smoothed_data['P_pelv'], ground_points_smooth, frame_width)
    AV_torso = angular_velocity(tor_angles, fps)

    #smoothed = lowess(elbR_ang_vel, range(len(elbR_ang_vel)), frac=0.3)
    #get_lag(elbR_ang_vel)
    #get_lag(smoothed[:, 1])

    """
    Data Extraction
    """
    file_path = 'sprint_data.csv'

    # Create a dictionary with all data
    data = {
        'A_R_knee': kneeR_angles,
        #'A_L_knee': kneeL_angles,
        'A_R_hip': hipR_angles,
        #'A_L_hip': hipL_angles,
        'A_R_elbow': elbR_angles,
        #'A_L_elbow': elbL_angles,
        'A_R_shou': shouR_angles,
        #'A_L_shou': shouL_angles,
        'A_R_ankle': A_R_ank,
        #'A_L_ankle': A_L_ank,
        'A_thigh': thigh_angles,
        'A_torso': tor_angles,
        'AV_R_knee': AV_R_knee,
        #'AV_L_knee': AV_L_knee,
        'AV_R_hip': AV_R_hip,
        #'AV_L_hip': AV_L_hip,
        'AV_R_elbow': AV_R_elb,
        #'AV_L_elbow': AV_L_elb,
        'AV_R_shou': AV_R_shou,
        #'AV_L_shou': AV_L_shou,
        'AV_R_ankle': AV_R_ank,
        #'AV_L_ankle': AV_L_ank,
        'AV_torso': AV_torso
    }

    # Check lengths of each column
    lengths = {key: len(value) for key, value in data.items()}

    """smoothed_data = {}
    for key, value in data.items():
        #if key.startswith("P"):
        #    continue
        xx = range(len(value))
        loess = Loess(xx, value)
        ys = []
        for x in xx:
            y = loess.estimate(x, window=12, use_matrix=False, degree=3)
            ys.append(y)
        smoothed_data[key] = ys"""

    # Find columns with lengths that differ from the others
    first_length = next(iter(lengths.values()))  # get the length of the first item
    unequal_lengths = {key: value for key, value in lengths.items() if value != first_length}

    # Output the results
    #print("Column lengths:", lengths)
    #print("Columns with unequal lengths:", unequal_lengths)

    # Determine the maximum length
    max_length = max(len(data[key]) for key in data)

    # Pad shorter columns with NaN
    for key in data:
        if key == 'A_thigh':
        #key = 'A_thigh'
            #plt.figure(figsize=(10, 6))
            plt.plot(range((len(ta_rough))), ta_rough, label='not', color='orange')
            plt.plot(range((len(thigh_angles))), thigh_angles, label='smooth')
            plt.xlabel('Time (s)')
            plt.ylabel('Velocity (units/s)')
            plt.title('Smooth vs nonsmooth vel')
            plt.legend()
            plt.show()

        data[key] = data[key] + [np.nan] * (max_length - len(data[key]))

    # Generate true signal

    # Simulate low capture rate
    y = np.array(ta_rough)
    b_indices = [i for i in range(len(y)) if (i + 2) % 3 == 0]
    b_indices = np.array(b_indices)  # Convert to numpy array
    b = y[b_indices]

    # Create gaps for interpolation
    c = np.full(len(y), np.nan)
    c[b_indices] = b
    print(c)

    # Perform LOESS smoothing
    xx = np.arange(len(y))
    loess_result = lowess(c, xx, frac=0.05, it=0, is_sorted=True)

    # Fill the smoothed values back into c
    # loess_result[:, 0] should be xx, but let's ensure it for safety
    # Also, align loess_result to the length of c
    if len(loess_result[:, 0]) == len(xx):
        c = np.where(np.isnan(c), loess_result[:, 1], c)
    else:
        # Handle the case where the lengths do not match
        # Use interpolation or adjust as needed
        interpolator = np.interp(xx, loess_result[:, 0], loess_result[:, 1])
        c = np.where(np.isnan(c), interpolator, c)

    # Plotting the results
    plt.plot(xx, y, color=[0, 1, 0, 0.3], linewidth=2.0, label='True Signal')
    plt.scatter(b_indices, b, color='blue', label='Captured Points')
    plt.plot(xx, c, color='red', label='LOESS Smoothed Signal')
    plt.legend()
    plt.show()
    # Create the DataFrame
    df = pd.DataFrame(data)

    #df.to_csv('checking.csv', index=False)

    """# Calculate mean and median for each column
    means = df.mean()
    medians = df.median()
    
    # Prepare data for a single row output
    data = {"Video": video_path}
    for col in df.columns:
        data[f'mean_{col}'] = [means[col]]
        data[f'median_{col}'] = [medians[col]]

    #n = df.count()

    # Calculate the standard deviation for each column
    #std_dev = df.std()

    # Compute the Standard Error for each column
    #se = std_dev / np.sqrt(n)

    # Adding "SE_" prefix to the index labels (if they are strings)
    #se.index = se.index.to_series().apply(lambda x: f'SE_{x}')
    #print(se)"""

    # Calculate means for each segment defined by indices
    lags_to_analyze = {"AV_R_knee":["AV_R_hip","AV_R_elbow","AV_R_ankle", "AV_R_shou"]}
    segment_means = {col: [] for col in df.columns}
    seg_max = {col: [] for col in df.columns if col.startswith("AV") or col.startswith("A_")}
    seg_min = {col: [] for col  in df.columns if col.startswith("A_")}
    roms = {col: [] for col in df.columns if col.startswith("A_")}
    sparc = {col: [] for col in df.columns if col.startswith("AV")}
    #perc_cycle_time = {col+: [] for col in df.columns if col.startswith("AV")}
    # New dictionary to store the combined keys
    perc_cycle_time = {}

    # Loop through the original dictionary
    for key, values in lags_to_analyze.items():
        for value in values:
            # Create a new key by combining the original key and the value
            new_key = f"{key}_to_{value}"
            # Store in the new dictionary with an example placeholder value (could be anything)
            perc_cycle_time[new_key] = []  # Replace None with actual data or computation if needed


    indices = detect_scissor(df['A_thigh'],10)
    #plt.figure(figsize=(10, 6))
    #plt.plot(range(len(thigh_angles)), df['A_thigh'], label='smooth')
    #plt.plot(range(len(thigh_angles)), thigh_angles, label='Left Knee Velocity')
    #plt.xlabel('Time (s)')
    #plt.ylabel('Velocity (units/s)')
    #plt.title('Velocity Comparison of Left and Right Knees')
    #plt.legend()
    #plt.show()
    #print('ranges')
    #print(indices)
    for col in df.columns:
        for i in range(len(indices)):
            # Calculate mean up to the given index (exclusive, assuming 0-based index)
            if i == 0:
                #mean_value = np.mean(df[col][:indices[i]])
                max_value = max(df[col][:indices[i]])
                if col.startswith("A_"):
                    min_value = min(df[col][:indices[i]])
                    seg_min[col].append(min_value)

                    rom = max_value - min_value
                    roms[col].append(rom)
                if col.startswith("AV"): # why is this always nan?
                    if col in lags_to_analyze.keys():
                        for x in lags_to_analyze[col]:
                            lag = get_lag(df[col][0:indices[i]],df[x][0:indices[i]])
                            #print("lag")
                            #print(lag)
                            #lag = lag / indices[i]
                            #lag = np.mean(lag)
                            perc_cycle_time[col+"_to_"+x].append(lag)

                    spec, smth1, smth2 = spectral_arclength(df[col][:indices[i]],fps)
                    if spec != None:
                        sparc[col].append(spec)
            else:
                # mean_value = np.mean(df[col][indices[i-1]:indices[i]])
                max_value = max(df[col][indices[i-1]:indices[i]])
                if col.startswith("A_"):
                    min_value = min(df[col][indices[i-1]:indices[i]])
                    rom = max_value - min_value

                    roms[col].append(rom)
                    seg_min[col].append(min_value)
                if col.startswith("AV"):
                    data = df[col][indices[i-1]:indices[i]]
                    if col in lags_to_analyze.keys():
                        for x in lags_to_analyze[col]:
                            lag = get_lag(df[col][indices[i-1]:indices[i]], df[x][indices[i-1]:indices[i]])
                        # print("lag")
                        # print(lag)
                        #lag = lag / indices[i]
                        #lag = np.mean(lag)
                            perc_cycle_time[col+"_to_"+x].append(lag)

                    #print(col)
                    #print(df[col][indices[i-1]:indices[i]])
                    spec, smth1, smth2 = spectral_arclength(df[col][indices[i-1]:indices[i]],fps)
                    if spec != None:
                        sparc[col].append(spec)
            #segment_means[col].append(mean_value)
            seg_max[col].append(max_value)


    # Calculate the mean of these segment means for each column
    #overall_means = {f'mean+SE_{col}': np.mean(means) + se[col] for col, means in segment_means.items()}
    # Compute mean+SE for each column and create two dictionaries
    #print(perc_cycle_time)
    #print(seg_min.items())
    #print(seg_min)
    #print(seg_max.items())
    overall_mins = {f'min_{col}': np.mean(means) for col, means in seg_min.items()}
    overall_maxs = {f'max_{col}': np.mean(means) for col, means in seg_max.items()}
    overall_roms = {f'rom_{col}': np.mean(means) for col, means in roms.items()}
    overall_perc_cycles = {f'%_cycle_{col}': np.mean(means) for col, means in perc_cycle_time.items()}
    #overall_cons = {f'lag_cons_{col}': se[col] for col, means in perc_cycle_time.items()}
    overall_sparcs = {f'sparc_{col}': np.mean(means) for col, means in sparc.items()}

    # Calculate the standard error of the minimum values for each column
    se_min = se_loop(seg_min.items(),'min')
    se_max = se_loop(seg_max.items(),'max')
    se_rom = se_loop(roms.items(),'rom')
    se_sparc = se_loop(sparc.items(),'sparc')
    se_lag = se_loop(perc_cycle_time.items(),'lag')

    # Combine both dictionaries into a single dictionary
    combined_means = {'Video':video_path, 'FPS':fps, 'Dur': duration_seconds, 'Sex': sex, 'PB':pb, "Level": level_of_expertise, "Cycles": len(indices), **overall_mins, **overall_maxs, **overall_roms, **overall_sparcs, **overall_perc_cycles, **se_lag, **se_min, **se_max, **se_rom, **se_sparc}

    # Create a new DataFrame from the prepared data
    #result_df = pd.DataFrame(data)
    result_df = pd.DataFrame(combined_means,index=[0])

    try:
        existing_df = pd.read_csv(file_path)

        # Append new data
        existing_df = pd.concat([existing_df, result_df], ignore_index=True)

        # Save the updated DataFrame back to the CSV
        existing_df.to_csv(file_path, index=False)

    except FileNotFoundError as e:
        print("")
        print(f"Error: {e}")
        print("Creating from scratch")
        # Save the DataFrame to a CSV file
        result_df.to_csv('sprint_data.csv', index=False)





    #need to standardize all data _______________________________________________________--
    from sklearn.preprocessing import StandardScaler
    """from sklearn.decomposition import PCA
    
    pca = PCA(n_components=2)
    
    principalComponents = pca.fit_transform(x)
    
    principalDf = pd.DataFrame(data = principalComponents
                 , columns = ['principal component 1', 'principal component 2'])"""

    # Save the DataFrame to a CSV file
    #df.to_csv('features.csv', index=False)


# Specify the folder containing videos
folder_path = './new'

# Process all videos in the folder
process_videos_in_folder(folder_path)
#extract('nov3.MOV')


"""import pandas as pd
from xgboost import XGBClassifier
from xgboost import plot_importance
from matplotlib import pyplot

# Load data using pandas
dataset = pd.read_csv('sprint_data.csv')

# Assuming columns are labeled and we want to exclude the first 5 columns
# Modify 'X' and 'y' extraction based on your specific needs
feature_names = dataset.columns[4:]  # Adjust index based on your data structure
X = dataset.iloc[:, 4:].values  # Features: all columns from index 6 onwards
y = dataset.iloc[:, 3].values  # Target: column at index 5 (Sex Cycles)
print(y)

# Plot feature importance
# fit model no training data
model = XGBClassifier()
model.fit(X, y)

# Feature importance
feature_importances = model.feature_importances_

# Pair feature names with their importance
features_with_importance = sorted(zip(feature_names, feature_importances), key=lambda x: x[1], reverse=True)

# Print feature importances
for feature, importance in features_with_importance:
    print(f'Feature: {feature}, Importance: {importance}')

# Plot feature importance
plt.bar(range(len(feature_importances)), feature_importances)
plt.xticks(range(len(feature_importances)), feature_names, rotation=90)
plt.xlabel('Feature')
plt.ylabel('Importance Score')
plt.title('Feature Importances in XGBoost Model')
plt.show()"""