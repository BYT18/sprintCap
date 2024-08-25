#!pip install -q mediapipe
#!wget -O pose_landmarker.task -q https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task
import heapq
import statistics

from mediapipe import solutions
import subprocess
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2
import matplotlib.pyplot as plt
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

#can do all the feedbakc checks in the seocnd interation fo the video using the key frames found already in the kinogram
def toe_off_feedback(plantAnk, plantKnee, plantHip, swingAnk, swingKnee, swingHip, swingToe, swingHeel):
    feedback = []

    # check knee bend
    if plantKnee[0] > plantAnk[0] and plantKnee[0] > plantHip[0]:
        feedback.append("Noticeable knee bend in stance leg (excessive pushing = stunted swing leg recovery / exacerbating rear side mechanics)")

    #check 90 degree angle between thighs?
    #thigh_angle = compute_angle(plantKnee, (plantHip+swingHip)/2 , plantKnee)
    thigh_angle = compute_angle(plantKnee, plantHip, plantKnee)
    if thigh_angle > 100 or thigh_angle < 80:
        feedback.append("90-degree angle between thighs")

    #check toe of swing leg is vertical to knee cap
    if swingKnee[0] - swingToe[0] < 5:
        feedback.append("Toe of swing leg directly vertical to kneecap")

    # front shin parallel to rear thigh
    # compute slope between ankle and knee
    par = are_parallel((plantHip,plantKnee), (swingKnee, swingAnk), 2)
    if par:
        feedback.append("Front shin parallel to rear thigh")

    # dorsiflexion of swing foot
    # compute angle between knee, ankle, toe
    footAng = compute_angle(swingToe,swingAnk,swingKnee)
    if 75<footAng<105:
        feedback.append("Dorsiflexion of swing foot")

    #forearms perpindicular to each other

    # no arching back
    # check how much head x is in front of hips or shoulders?
    #feedback.append("Torso angle: "+str(torsoAng))

    return feedback

def max_vert_feedback(frontAnk, frontKnee, frontHip, frontToe, backAnk, backKnee, backHip,backShou):
    # check knee bend
    feedback = []
    #check 90 degree angle between thighs?
    thigh_angle = compute_angle(frontKnee, frontHip, backKnee)
    if thigh_angle > 100 or thigh_angle < 80:
        feedback.append("90-degree angle between thighs")

    #knee angle front leg
    knee_angle = compute_angle(frontHip,frontKnee,frontAnk)
    if knee_angle > 100:
        feedback.append(">110 degrees knee angle in front leg")

    #dorsiflexion of front foot
    # cant use heel vs toe y value since foot can be at an angle?
    footAng = compute_angle(frontToe,frontAnk,frontKnee)
    if 70<footAng<115:
        feedback.append("Dorsiflexion of swing foot")

    #peace?? fluidity
    #look at sparc

    #neutral head
    #torso angle?

    #straight line between rear leg knee and shoulder
    if abs(backKnee[0]-backHip[0]) < 15 and abs(backHip[0]-backShou[0])< 15:
        feedback.append("Straight line between rear knee, hips, and shoulders")


    #slight cross body arm swing - neutral rotation of spine

    return feedback


def strike_feedback(frontAnk, frontKnee, frontHip, frontToe, backAnk, backKnee, backHip):
    feedback = []

    # rear femur perpendicular to ground
    if abs(backKnee[0] - backHip[0]) < 5:
        feedback.append("Femur perpendicular to ground")

    # check 20-40 degree angle between thighs
    thigh_angle = compute_angle(frontKnee, frontHip , backKnee)
    if 10 < thigh_angle < 50:
        feedback.append("20-40-degree gap between thighs")

    # front foot beginning to supinate
    footAng = compute_angle(frontKnee,frontAnk,frontToe)
    if footAng > 90:
        feedback.append("Front foot beginning to supinate ")

    # knee extension comes from reflexive action???
    feedback.append("Knee extension should come from strong reflexive action of posterior chain")

    # greater contraction in hammys???

    return feedback

def touch_down_feedback(plantAnk, plantKnee, plantHip, plantHeel, plantToe, swingAnk, swingKnee, swingHip, swingToe, swingHeel, elbR_ang, elbL_ang):
    feedback = []

    # knee closeness
    knees = swingKnee[0] - plantKnee[0]
    if knees >= 0:
        feedback.append("Knees together - Faster atheletes may have swing leg slightly in front of stance leg")

    if knees < 0:
        feedback.append("If knee is behind - indication of slower rotational velocities = over pushing = exacerbation of rear side mechanics")

    #perpendicular stance shin, heel underneath knee
    if (plantKnee[0] - plantAnk[0])<5 and (plantKnee[0] - plantHeel[0])<5:
        feedback.append("Shank of stance shin perpendicular to ground, heel directly underneath knee")

    #hands parallel to each other and ...
    if abs(elbR_ang - elbL_ang)<20:
        feedback.append("Hands paralllel to each other / similar elbow angles")

    #inital contact with outside of ball of foot / plantar flex
    if (plantToe[1] - plantHeel[1])<5:
        feedback.append("Plantar flex prior to touchdown")

    #allow elbow joints to open as hands pass hips

    return feedback


def full_supp_feedback(plantAnk, plantKnee, plantHip, plantToe, plantHeel, swingAnk, swingKnee, swingHip, swingToe, swingHeel):
    feedback = []

    # swing foot tucked under glutes
    # check y value of heel and how close it is to hips
    if swingHeel[1] - swingHip[1] < 30:
        feedback.append("Swing foot tucked under glutes")
    else:
        feedback.append("Heel not cycling high enough")
    # swing leg in front of hips (thigh 45 degrees from perpindicular)
    thigh_angle = compute_angle(swingAnk, swingKnee, swingHip)
    if thigh_angle < 55:
        feedback.append("Swing leg in front of hips (thigh 45 degrees from perpendicular)")

    # stance amortiation ? excess amortization
    if plantKnee[0] - plantAnk[0] > 25:
        feedback.append("Collapse at knee. Strength related or touch down too far in front of COM")

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
    tbf = duration / (fps*true_fps_factor)
    gcontact_times = []
    flight_times = []
    counter = 0
    for i in range(len(imp_frames)-1):
        if imp_frames[i] + 1 == imp_frames[i+1]:
            counter += 1
            if i+1 == len(imp_frames)-1:
                gcontact_times.append(counter*tbf)
                flight_times.append(0)
        else:
            #gcontact_times.append(counter*tbf)
            #counter = 0
            #flight_times.append(tbf*(imp_frames[i+1]-imp_frames[i]))
            gcontact_times.append(counter * tbf)
            if (imp_frames[i + 1] - imp_frames[i] > counter * 1.3) and counter != 0:
                flight_times.append(
                    (tbf * (imp_frames[i + 1] - imp_frames[i])) / ((imp_frames[i + 1] - imp_frames[i]) // counter))
            else:
                flight_times.append((tbf * (imp_frames[i + 1] - imp_frames[i])))
            counter = 0

    return [flight_times,gcontact_times]


def step_len_anal(ank_pos, frames, output_frames, pic, leg_length_px, leg_length):
    imp_frames = frames
    cap_frames = []
    s_lens = []
    initial = 0
    for i in range(len(imp_frames) - 1):
        #if frames are consequtive then they are looking at the same ground contact
        if imp_frames[i] + 1 == imp_frames[i + 1] and initial == 0:
            #initial = left or right ankle position
            marker_pos = obj_detect(output_frames[imp_frames[i]],pic)
            #initial = abs(ank_pos[i][0] - marker_pos)
            cap_frames.append(imp_frames[i])
            initial = euclidean_distance(ank_pos[i],marker_pos)
            pixel_to_meter_ratio = leg_length / leg_length_px[imp_frames[i]]
            initial = initial * pixel_to_meter_ratio

        elif imp_frames[i] + 1 == imp_frames[i + 1] and initial != 0:
            continue
        else:
            marker_pos = obj_detect(output_frames[imp_frames[i]], pic)
            #left_step_length_px1 = abs(abs(ank_pos[i+1][0]-marker_pos) - initial1)
            #s_lens.append(abs(abs(ank_pos[i+1][0]-marker_pos) - initial))
            cap_frames.append(imp_frames[i])
            #s_lens.append(ank_pos[i + 1][0] - initial)
            left_step_length_px = euclidean_distance(ank_pos[i + 1], marker_pos)

            #idk what this is for
            if len(leg_length_px)>=imp_frames[i+1]:
                pixel_to_meter_ratio = leg_length / leg_length_px[imp_frames[i+1]]
                #left_step_length = (left_step_length_px / leg_length_px) * leg_length
                left_step_length = left_step_length_px * pixel_to_meter_ratio
                left_step_length = abs(left_step_length - initial)
                s_lens.append(left_step_length)
                initial = 0
    return s_lens

def step_length(height, height_pixels, distances, results, start_frame, end_frame, start_foot_coords, end_foot_coords):
    lens = []

    for i in range(len(distances)):
        meters_per_pixel = height / height_pixels
        pixel_distance = ((distances[i]) ** 2) ** 0.5
        step_length_meters = pixel_distance * meters_per_pixel
        lens.append(step_length_meters)

    return lens

# Function to calculate velocity
def calculate_velocity(positions, fps):
    velocities = []
    for i in range(1, len(positions)):
        velocity = (np.array(positions[i]) - np.array(positions[i - 1])) * fps
        velocities.append(velocity)
        # Example implementation, adjust based on your actual function
        # velocities = np.diff(positions, axis=0) * fps
    return velocities


"""
might not work if the pole is not always in frame? eg: since we are calling this whenever interested in initial position, 
there is no guarantee a pole is in frame.
"""
def obj_detect(img, temp):

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    assert img is not None, "file could not be read, check with os.path.exists()"
    img2 = img.copy()
    #template = cv2.imread('pole.png', cv2.IMREAD_GRAYSCALE)
    #template = cv2.imread(temp, cv2.IMREAD_GRAYSCALE)
    temp_file_path = 'media/pics/image.png'
    with open(temp_file_path, 'wb') as f:
        for chunk in temp.chunks():
            f.write(chunk)
    template = cv2.imread(temp_file_path, cv2.IMREAD_GRAYSCALE)
    #template = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
    assert template is not None, "file could not be read, check with os.path.exists()"
    w, h = template.shape[::-1]

    # All the 6 methods for comparison in a list
    #methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR', 'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
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

        cv2.rectangle(img, top_left, bottom_right, 255, 20)
        #cv2.imshow("Image with Rectangle", img)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        mid = top_left[0] + (bottom_right[0] - top_left[0]) / 2

        return (mid, bottom_right[1])

# Function to calculate Euclidean distance
def euclidean_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

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
            median_values.append((median_valueX,median_valueY))
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

            angles.append(180-angle_degrees)
            start += 1
            end += 1

    return angles

import math
from typing import Tuple

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
        if min_idx == n-1:
            return np.arange(n - window, n)

        min_range = [min_idx]
        while len(min_range) < window:
            i0 = min_range[0]
            i1 = min_range[-1]
            if i0 == 0:
                min_range.append(i1 + 1)
            elif i1 == n-1:
                min_range.insert(0, i0 - 1)
            elif distances[i0-1] < distances[i1+1]:
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

def pos_smooth(pos, win):
    xx = range(len(pos))
    x_values, y_values = zip(*pos)
    # Create Loess objects for x and y values
    loess_x = Loess(range(len(x_values)), list(x_values))
    loess_y = Loess(range(len(y_values)), list(y_values))
    ys1 = []
    for x in xx:
        wind = win
        # Smooth x and y values separately
        smoothed_x = loess_x.estimate(x, window=wind, use_matrix=False, degree=3)
        smoothed_y = loess_y.estimate(x, window=wind, use_matrix=False, degree=3)
        ys1.append((smoothed_x, smoothed_y))
    return ys1


"""
Main
"""
def get_gif(vid, pic, ath_height,slowmo, step):
    # STEP 1: Import the necessary modules.
    import mediapipe as mp
    import tempfile

    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    #video_path = 'C:\Users\hocke\Desktop\UofT\Third Year\CSC309\moveNet\src\fly.mov'
    video_path = vid
    #output_path = '/pics/output_video.mp4'
    output_path = os.path.join('media/pics', 'output_video.mov')
    #output_path = os.path.join('/pics', 'output_video.mp4')
    # Save the uploaded file to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mov') as temp_file:
        for chunk in vid.chunks():
            temp_file.write(chunk)
        temp_file_path = temp_file.name

    video_path = temp_file_path
    cap = cv2.VideoCapture(video_path)

    # Get the frame rate and frame size of the video
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # Codec for mp4 files
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # variables for frame analysis
    toe_offs = []
    max_verts = []
    strikes = []
    tdowns = []
    full_supps = []
    kinogram = [0,0,0,0,0]
    max_knee_seperation = 0
    output = []
    max_y = 0
    ground = 1000
    height = 100
    knee_hip_alignment_support = 100
    knee_hip_alignment_strike = 100
    knee_ank_alignment_support = 100

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

    # variables for ground contacts
    ground_points={}
    ground_frames = []
    ground_contacts = 0

    leg_length = 0.5  # in meters
    leg_length_px = []

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
                midPelvis = (int(((right_hip.x + left_hip.x) / 2) * frame_width),
                             int(((right_hip.y + left_hip.y) / 2) * frame_height))

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

                if (ankR[0] > midPelvis[0]):
                    frontside.append(ankR)
                else:
                    backside.append(ankR)

                leg_length_px.append(euclidean_distance(hipR, kneeR))

                prev_ankle_pos = ankL
                prev_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

                tAng = compute_angle(hipR, kneeR, ankR)
                kneeR_angles.append(tAng)

                tAng = compute_angle(kneeL, midPelvis, kneeR)
                thigh_angles.append(tAng)

                # toeoff
                if footL[1] + 5 > ground:
                    max_knee_seperation = ((kneeR[0] - kneeL[0]) ** 2 + (kneeR[1] - kneeL[1]) ** 2) ** 0.5
                    kinogram[0] = frame_idx
                    if toe_offs and any(abs(frame_idx - fn) < 2 for _, fn in toe_offs):
                        pass
                    if len(toe_offs) < 5:
                        heapq.heappush(toe_offs, (max_knee_seperation, frame_idx))
                    else:
                        # Check if the new value is larger than the smallest value in the heap
                        if max_knee_seperation > toe_offs[0][0]:  # Compare based on the value
                            # Replace the smallest value with the new tuple
                            heapq.heappushpop(toe_offs, (max_knee_seperation, frame_idx))

                # max proj /
                if abs(kneeL[0] - kneeR[0]) > 20:
                    height = abs(footL[1] - footR[1])
                    kinogram[1] = frame_idx

                    if max_verts and any(abs(frame_idx - fn) < 2 for _, fn in max_verts):
                        pass
                    else:
                        if len(max_verts) < 5:
                            heapq.heappush(max_verts, (-height, frame_idx))
                        else:
                            if -height > max_verts[0][0]:
                                heapq.heappushpop(max_verts, (-height, frame_idx))

                # full support left
                if ankR[1] < ankL[1]:
                    knee_hip_alignment_support = abs(ankL[0] - midPelvis[0])
                    knee_ank_alignment_support = abs(kneeL[0] - ankL[0])
                    pix = footL[1] - nose[1]
                    if pix > height_in_pixels:
                        height_in_pixels = pix
                    kinogram[4] = frame_idx
                    if full_supps and any(abs(frame_idx - fn) < 2 for _, fn in full_supps):
                        pass
                    if len(full_supps) < 5:
                        heapq.heappush(full_supps, (-knee_hip_alignment_support, frame_idx))
                    else:
                        if -knee_hip_alignment_support > full_supps[0][0]:
                            heapq.heappushpop(full_supps, (-knee_hip_alignment_support, frame_idx))

                # strike R
                if ankL[1] + 15 < ground and ankR[0] > hipR[0]:
                    # and ankL[1]<kneeR[1]
                    knee_hip_alignment_strike = abs(kneeL[0] - hipL[0])
                    kinogram[2] = frame_idx

                    if strikes and any(abs(frame_idx - fn) < 2 for _, fn in strikes):
                        pass  # Skip this frame if it's too close to an existing one

                    if len(strikes) < 5:
                        # Add the new value to the heap if we have fewer than 5 elements
                        heapq.heappush(strikes, (-knee_hip_alignment_strike, frame_idx))
                    else:
                        # Check if the new value is larger than the smallest value in the heap
                        if -knee_hip_alignment_strike > strikes[0][0]:  # Compare based on the value
                            # Replace the smallest value with the new tuple
                            heapq.heappushpop(strikes, (-knee_hip_alignment_strike, frame_idx))

                # touch down
                max_y = ((ankR[1] - ankL[1]) ** 2) ** 0.5
                kinogram[3] = frame_idx

                if tdowns and any(abs(frame_idx - fn) < 2 for _, fn in tdowns):
                    pass  # Skip this frame if it's too close to an existing one

                if len(tdowns) < 5:
                    # Add the new value to the heap if we have fewer than 5 elements
                    heapq.heappush(tdowns, (max_y, frame_idx))
                else:
                    # Check if the new value is larger than the smallest value in the heap
                    if max_y > tdowns[0][0]:  # Compare based on the value
                        # Replace the smallest value with the new tuple
                        heapq.heappushpop(tdowns, (max_y, frame_idx))

                # if abs(kneeL[0] - kneeR[0]) < 10 or (abs(elbL[0] - elbR[0]) < 10 and abs(kneeL[0] - kneeR[0]) < 25):
                if abs(kneeL[0] - kneeR[0]) < 25 and (abs(footL[1] - heelL[1]) < 10 or abs(footR[1] - heelR[1]) < 10):
                    ground_contacts += 1
                    if footL[1] > footR[1]:
                        ground = footL[1]
                        g1 = footL
                    else:
                        ground = footR[1]
                        g1 = footR
                    ground_frames.append(frame_idx)  # take lowest point of the ground points in the group
                    ground_points[frame_idx] = g1

            # Write the frame to the output video
            #out.write(frame)
            output.append(frame)

        # Release resources
        #out.release()
        cv2.destroyAllWindows()

    # variables for kinogram feedback
    feedback = {}

    imp_frames = []
    contact = False
    threshold = 100000

    #ground_points_smooth = find_median_of_consecutive_keys(ground_points)
    #tor_angles = torso_angles(nose_pos,mid_pelvis_pos, ground_points_smooth, frame_width)

    """cap = cv2.VideoCapture(video_path)
    # Get the frame rate and frame size of the video
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4 files
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))"""
    cap = cv2.VideoCapture(video_path)
    #fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4 files
    #out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Get the current frame number
            frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1

            if frame_idx in ground_frames:
                contact = True
                threshold = ground_points[frame_idx][1]

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
                            #f = toe_off_feedback(ankR, kneeR, hipR, ankL, kneeL, hipL, footL, heelL, tor_angles[ind])
                            f = toe_off_feedback(ankR, kneeR, hipR, ankL, kneeL, hipL, footL, heelL)
                        feedback["TO"]=f
                    elif ind == 1:
                        if ankL[0] > ankR[0]:
                            f = max_vert_feedback(ankL, kneeL, hipL, footL, ankR, kneeR, hipR, shouR)
                        else:
                            f = max_vert_feedback(ankR, kneeR, hipR, footR, ankL, kneeL, hipL, shouL)
                        feedback["MV"] = f
                    elif ind == 2:
                        if ankL[0] > ankR[0]:
                            f = strike_feedback(ankL, kneeL, hipL, footL, ankR, kneeR, hipR)
                        else:
                            f = strike_feedback(ankR, kneeR, hipR, footR, ankL, kneeL, hipL)
                        feedback["S"]=f
                    elif ind == 3:
                        elbR_ang = compute_angle(wrR, elbR, shouR)
                        elbL_ang = compute_angle(wrL, elbL, shouL)
                        if ankL[0] > ankR[0]:
                            f = touch_down_feedback(ankL, kneeL, hipL, heelL, footL, ankR, kneeR, hipR, footR, heelR, elbR_ang, elbL_ang)
                        else:
                            f = touch_down_feedback(ankR, kneeR, hipR, heelR, footR, ankL, kneeL, hipL, footL, heelL, elbR_ang, elbL_ang)
                        feedback["TD"]=f
                    else:
                        if ankL[0] > ankR[0]:
                            f = full_supp_feedback(ankL, kneeL, hipL, footL, heelL, ankR, kneeR, hipR, footR, heelR)
                        else:
                            f = full_supp_feedback(ankR, kneeR, hipR, ankL, footR, heelR, kneeL, hipL, footL, heelL)
                        feedback["FS"]=f

            # Draw the pose annotation on the frame
            if results.pose_landmarks and contact == True:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                #get_kinogram_frames(frame,results,kinogram,max_knee_seperation)
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

                hipL =  (int(left_hip.x * frame_width), int(left_hip.y * frame_height))
                midPelvis = (int(((right_hip.x + left_hip.x)/2) * frame_width), int(((right_hip.y + left_hip.y)/2) * frame_height))

                elbL = (int(left_elbow.x * frame_width), int(left_elbow.y * frame_height))
                elbR = (int(right_elbow.x * frame_width), int(right_elbow.y * frame_height))

                if frame_idx in imp_frames:
                    contact = False
                else:

                    # ensure hand position is sufficiently far (this will ensure that no stoppages occur when knees are close
                    # left foot is on the ground
                    if (ankR[1] - ankL[1] <= 0):
                            #checking if the distance between knees does not change enough then that means we are at toe off frame
                        if abs(footL[1] - threshold)<10:
                            imp_frames.append(frame_idx)
                            ank_pos.append(ankL)
                        else:
                            contact = False
                    else:
                        if abs(footR[1] - threshold)<10:
                            imp_frames.append(frame_idx)
                            ank_pos.append(ankR)
                        else:
                            contact = False

            # Write the frame to the output video
            #out.write(frame)
            #output.append(frame)

        # Release resources
        cap.release()
        #out.release()
        cv2.destroyAllWindows()

    if slowmo == 0:
        wind = 5
    else:
        wind = 20

    kneeR_pos = pos_smooth(kneeR_pos,wind)
    ankR_pos = pos_smooth(ankR_pos,wind)
    hipR_pos = pos_smooth(hipR_pos,wind)

    kneeR_angles = []
    for i in range(len(kneeR_pos)):
        a = compute_angle(hipR_pos[i], kneeR_pos[i], ankR_pos[i])
        kneeR_angles.append(a)

    signal = np.array(kneeR_angles)
    # Compute Z-scores
    mean = np.mean(signal)
    std_dev = np.std(signal)
    z_scores = (signal - mean) / std_dev

    # Define a threshold for anomaly
    threshold = 2
    anomalies = np.where(np.abs(z_scores) > threshold)[0]

    padding = math.ceil(len(output) * 0.05)
    # Write the trimmed frames to the output video
    if len(anomalies)>0:
        for i in range(anomalies[len(anomalies) - 1] + padding, len(output)):
            out.write(output[i])
    else:
        for i in range(len(output)):
            out.write(output[i])
    out.release()

        # Now, convert the output video to MP4 using ffmpeg
    try:
            v_path = os.path.join('media/pics', 'output_video.mp4')
            subprocess.run(['ffmpeg', '-y', '-i', output_path, '-vcodec', 'h264', '-acodec', 'aac', v_path], check=True)

            print("Conversion to MP4 successful.")
    except subprocess.CalledProcessError as e:
            print(f"Conversion to MP4 failed: {e}")

    kinogram[0] = toe_offs[4][1]
    kinogram[1] = max_verts[4][1]
    kinogram[2] = strikes[4][1]
    kinogram[3] = tdowns[4][1]
    kinogram[4] = full_supps[4][1]

    from django.conf import settings  # To use MEDIA_ROOT and MEDIA_URL

    # Ensure the directory exists
    output_dir = os.path.join(settings.MEDIA_ROOT, 'pics')
    os.makedirs(output_dir, exist_ok=True)

    image_urls = []
    counter = 1

    # First set of images
    for i in range(len(kinogram)):
        rgb_frame = cv2.cvtColor(output[kinogram[i]], cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(10, 6))
        plt.imshow(rgb_frame)
        plt.axis('off')

        plot_path = os.path.join(output_dir, f'key_frame_{i + 1}.png')
        plt.savefig(plot_path, bbox_inches='tight', pad_inches=0)
        plt.close()  # Close the figure

    # Second set of images (from heaps)
    images_heaps = [toe_offs, max_verts, strikes, tdowns, full_supps]

    """for x in images_heaps:
        for y in x:
            i = y[1]
            rgb_frame = cv2.cvtColor(output[i], cv2.COLOR_BGR2RGB)
            plt.figure(figsize=(10, 6))
            plt.imshow(rgb_frame)
            plt.axis('off')

            plot_path = os.path.join(output_dir, f'out_{counter}.png')
            counter += 1
            plt.savefig(plot_path, bbox_inches='tight', pad_inches=0)
            plt.close()  # Close the figure

            # Generate the URL to be returned to the frontend
            image_url = os.path.join(settings.MEDIA_URL, 'pics', os.path.basename(plot_path))
            image_urls.append(image_url)"""

    """
    Velocity and Smoothness Analysis 
    """
    # Calculate velocities
    velocitiesL = calculate_velocity(kneeL_pos, fps)
    velocitiesR = calculate_velocity(kneeR_pos, fps)

    # Compute velocity magnitudes
    # since velocities are in x and y direction, need to reduce to scalar that does not have direction
    velocity_magnitudeL = np.linalg.norm(velocitiesL, axis=1)
    velocity_magnitudeR = np.linalg.norm(velocitiesR, axis=1)

    # Create time axis
    time_velocity = np.arange(len(velocity_magnitudeL)) / fps

    # Convert numpy arrays to lists for serialization
    velocity_magnitudeL_list = velocity_magnitudeL.tolist()
    velocity_magnitudeR_list = velocity_magnitudeR.tolist()
    time_velocity_list = time_velocity.tolist()

    """num_frames = len(imp_frames)
    fig, axs = plt.subplots(1, num_frames, figsize=(num_frames * 5, 5))

    for i in range(num_frames):
        axs[i].imshow(output[imp_frames[i]])
        axs[i].axis('off')
        axs[i].set_title(f"Contact {i + 1}")

    plt.tight_layout()
    #plt.show()"""

    """
    Step Length Analysis 
    """
    if step == 1:
        sLength = step_len_anal(ank_pos, imp_frames, output, pic, leg_length_px,leg_length)
    else:
        sLength = []
    print("steps " + str(sLength))
    """
    Flight and ground contact time analysis 
    """
    if slowmo == 1:
        f_g_times = contact_flight_analysis(imp_frames,fps,8,len(output)/fps)
    else:
        f_g_times = contact_flight_analysis(imp_frames,fps,1,len(output)/fps)

    ground_times = f_g_times[1]
    flight_times = f_g_times[0]

    # watch out for last flight time is always 0
    if len(sLength) != 0:
        max_step_len = sum(sLength)/len(sLength)
        if len(ground_times) > 0 and len(flight_times) > 1:
            avg_ground_time = sum(ground_times) / len(ground_times)
            avg_flight_time = sum(flight_times) / (len(flight_times) - 1)
            time_btw_steps = 0
            print(avg_ground_time)
        else:
            avg_ground_time = 0
            avg_flight_time = 0
    else:
        max_step_len = 0
        avg_ground_time = 0
        avg_flight_time = 0

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
        #'P_R_toe': toeR_pos,
        'P_head': nose_pos,
    }

    """# Example x-coordinate positions (from pose estimation)
    x_coords = [x for x, y in nose_pos]
    # Example list of reference object sizes in pixels for each frame
    reference_sizes_pixels = leg_length_px  # This changes per frame
    reference_size_meters = leg_length # The real-world size of the object (e.g., 1 meter)

    # Calculate displacements in pixels
    displacements_pixels = np.diff(x_coords)

    # Frame rate
    fps = 30
    time_interval = 1 / fps

    # Initialize an empty list to store velocities
    velocities_mps = []

    # Calculate scale factor and velocities for each frame
    for i in range(len(displacements_pixels)):
        scale_factor_frame = reference_sizes_pixels[i] / reference_size_meters
        displacement_meters = displacements_pixels[i] / scale_factor_frame
        velocity_mps = displacement_meters / time_interval
        velocities_mps.append(velocity_mps)

    print("X Coordinates (pixels):", x_coords)
    print("Displacements (meters):", displacements_pixels)
    print("Horizontal Velocities (m/s):", velocities_mps)"""

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


    return [{"ground":ground_times,"flight":flight_times,"kneePos":kneeL_pos,"feedback":feedback,
            "avg_f":avg_flight_time, "avg_g":avg_ground_time, "maxSL":max_step_len, "sLen" : sLength, "ang":thigh_angles,
            "vL": velocity_magnitudeL_list, "vR": velocity_magnitudeR_list, "vT": time_velocity_list}, image_urls]
