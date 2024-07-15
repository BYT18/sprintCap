#!pip install -q mediapipe
#!wget -O pose_landmarker.task -q https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task

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
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

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
def toe_off_feedback(plantAnk, plantKnee, plantHip, swingAnk, swingKnee, swingHip, swingToe, swingHeel, ):
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

    return feedback

def max_vert_feedback(frontAnk, frontKnee, frontHip, frontToe, backAnk, backKnee, backHip):
    # check knee bend
    feedback = []
    #check 90 degree angle between thighs?
    thigh_angle = compute_angle(frontKnee, frontHip, backKnee)
    if thigh_angle > 100 or thigh_angle < 80:
        feedback.append("90-degree angle between thighs")

    #knee angle front leg
    knee_angle = compute_angle(frontHip,frontKnee,frontAnk)
    if knee_angle > 100:
        feedback.append("Knees")

    #dorsiflexion of front foot
    # cant use heel vs toe y value since foot can be at an angle?
    footAng = compute_angle(frontToe,frontAnk,frontKnee)
    if 75<footAng<105:
        feedback.append("Dorsiflexion of swing foot")

    #peace?? fluidity

    #neutral head

    #straight line between rear leg knee and shoulder

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

    # greater contraction in hammys???

    return feedback

def touch_down_feedback(plantAnk, plantKnee, plantHip, plantHeel, plantToe, swingAnk, swingKnee, swingHip, swingToe, swingHeel):
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

def contact_flight_analysis(frames, fps, duration):

    imp_frames = frames
    #tot_frames = dur*curr_fps(slow-mo)
    #Total frames in the original video = Frame rate × Duration = 240 fps × 3 s = 720 frames

    #tot_frames = 3*720

    #Frame rate of the GIF = Total frames of GIF / Duration = 93 frames / 3 s
    #fps = 93 / 3

    #Time per frame in GIF = Duration / Total frames of GIF = 3 s / 93 frames

    tbf = duration / (fps*8)

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
            gcontact_times.append(counter*tbf)
            counter = 0
            flight_times.append(tbf*(imp_frames[i+1]-imp_frames[i]))

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
    return [flight_times,gcontact_times]

# use leg length to calibrate distance to pole, then use new leg length to calibrate distance from pole in next step, this shoudl give more accurate measurments
def step_len_anal(ank_pos, frames, output_frames, leg_length_px, leg_length):
    imp_frames = frames
    print("anal")
    print(imp_frames)
    cap_frames = []
    s_lens = []
    initial = 0
    for i in range(len(imp_frames) - 1):
        #if frames are consequtive then they are looking at the same ground contact
        if imp_frames[i] + 1 == imp_frames[i + 1] and initial == 0:
            #initial = left or right ankle position
            marker_pos = obj_detect(output_frames[imp_frames[i]],0)
            #initial = abs(ank_pos[i][0] - marker_pos)
            cap_frames.append(imp_frames[i])
            initial = euclidean_distance(ank_pos[i],marker_pos)
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
            #left_step_length_px1 = abs(abs(ank_pos[i+1][0]-marker_pos) - initial1)
            #s_lens.append(abs(abs(ank_pos[i+1][0]-marker_pos) - initial))
            cap_frames.append(imp_frames[i])
            #s_lens.append(ank_pos[i + 1][0] - initial)
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

            pixel_to_meter_ratio = leg_length / leg_length_px[imp_frames[i+1]]
            #left_step_length = (left_step_length_px / leg_length_px) * leg_length
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
    #height_pixels = abs(results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y -
    #                    results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].y)  # height in pixels
    #height_pixels = abs(475-280)
    #height_pixels = 207
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
    #img = cv2.imread('temptest2.png', cv2.IMREAD_GRAYSCALE)
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
    template = cv2.imread('pole.png', cv2.IMREAD_GRAYSCALE)
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

        cv2.rectangle(img, top_left, bottom_right, 255, 2)
        mid = top_left[0]+ (bottom_right[0]-top_left[0])/2
        print(mid)
        print(min_val)

        """plt.subplot(121), plt.imshow(res, cmap='gray')
        plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(img, cmap='gray')
        plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
        plt.suptitle(meth)

        plt.show()"""
        return (mid,bottom_right[1])
    #img_rgb = cv2.imread('mario.png')
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

def calculate_velocity(positions, fps, ref_len_actual, ref_len_px,title):
    velocities = []
    for i in range(1, len(positions)):
        #velocity = (np.array(positions[i]) - np.array(positions[i - 1])) * fps
        #horizontal_velocity_px = (np.array(positions[i]) - np.array(positions[i - 1])) * fps
        horizontal_velocity_px = euclidean_distance(positions[i],positions[i - 1]) * fps
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

def torso_angles(start_points, end_points, fixed_line_points):
    angles = []
    start = 0
    end = 0

    # Iterate over the fixed line points to create segments
    for i in range(len(fixed_line_points) - 1):
        # Define the fixed line vector
        vector_fixed = np.array([frame_width,
                                 fixed_line_points[i + 1][1] - fixed_line_points[i][1]])

        #for start, end in zip(start_points, end_points):
        while start_points[start][0] < fixed_line_points[i + 1][0] and end_points[end][0]<fixed_line_points[i + 1][0]:
            # Define the vector for the current segment in the list
            vector_segment = np.array([end_points[end][0] - start_points[start][0], end_points[end][1] - start_points[start][1]])

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

    """frames_x = range(len(angles))

    # Perform LOESS smoothing
    smoothed = lowess(angles, frames_x, frac=0.3)

    # Create a scatter plot
    #plt.plot(frames_x, angles, color='blue', label='Angles')
    # Plot the results
    plt.scatter(frames_x, angles, label='Data', color='gray')
    plt.plot(smoothed[:, 0], smoothed[:, 1], label='LOESS', color='red')
    # Add labels and title
    plt.xlabel('Frame')
    plt.ylabel('Angles')
    plt.title('Torso Angles over Frames')
    # Add a legend
    plt.legend()
    plt.show()"""

    return angles

def angluar_velocity(angles, title):
    # Calculate angular velocity
    angular_velocity = [(angles[i+1] - angles[i]) for i in range(len(angles) - 1)]

    # Generate x-values for angular velocity (indices start from 0 to len(angular_velocity) - 1)
    frames_x = range(len(angular_velocity))

    x = [(frame_index / 30) * 1000 for frame_index in frames_x]

    # Plot angular velocity
    """smoothed = lowess(angular_velocity, x, frac=0.3, delta=150)

    # Plot the results
    plt.scatter(x, angular_velocity, label='Data', color='gray')
    plt.plot(smoothed[:, 0], smoothed[:, 1], label='LOESS', color='red')

    #plt.plot(frames_x, angular_velocity, color='red', label='Angular Velocity')
    plt.xlabel('Frame Index')
    plt.ylabel('Angular Velocity (degrees/frame)')
    plt.title('Angular Velocity of '+title)
    plt.legend()
    plt.show()"""

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

    """# Plot angular velocity with peaks
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(len(data)), data, label='Angular Velocity')
    plt.plot(peaks, data[peaks], 'x', label='Peaks')
    plt.xlabel('Time (s)')
    plt.ylabel('Angular Velocity')
    plt.legend()
    plt.title('Angular Velocity with Peaks')

    # Plot lags between peaks
    plt.subplot(1, 2, 2)
    plt.bar(range(len(lags)), lags, color='skyblue')
    plt.xlabel('Peak Number')
    plt.ylabel('Lag between Peaks (s)')
    plt.title('Lag between Peaks')
    plt.grid(True)

    plt.tight_layout()
    plt.show()"""


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
video_path = 'adam.mov'
output_path = 'output_video.mp4'
cap = cv2.VideoCapture(video_path)

# Get the frame rate and frame size of the video
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4 files
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

"""###
img = cv2.imread("fullbody.jpg")
rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.3, min_tracking_confidence=0.3)
# Process the frame to get pose landmarks
results = pose.process(rgb_frame)
mp_drawing.draw_landmarks(rgb_frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

left_ankle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
left_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
ankL = (int(left_ankle.x * frame_width), int(left_ankle.y * frame_height))
hipL =  (int(left_hip.x * frame_width), int(left_hip.y * frame_height))
left_toe = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX]
left_eye = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EYE]
toeL = (int(left_toe.x * frame_width), int(left_toe.y * frame_height))
eyeL =  (int(left_eye.x * frame_width), int(left_eye.y * frame_height))
print(euclidean_distance(ankL,hipL)*(183/euclidean_distance(ankL,eyeL)))
# Plot the image
plt.imshow(rgb_frame)
# Draw the points
plt.show()
###"""


# variables for frame analysis
kinogram = [0,0,0,0,0]
max_knee_seperation = 0
output = []
max_y = 0
ground = 1000
height = 100
knee_hip_alignment_support = 100
knee_hip_alignment_strike = 100
knee_ank_alignment_support = 100

# variables for ground contacts
ground_points={}
ground_frames = []
ground_contacts = 0
#threshold = 5  # Threshold for considering significant movement (in pixels)
#prev_left_foot = None
#prev_right_foot = None

#vars for step len
height_in_pixels = 0
ank_pos = []

#smoothness and rom
elbL_pos = []
elbR_pos = []
ankL_pos = []
ankR_pos = []
hipL_pos = []
hipR_pos = []
kneeL_pos = []
kneeL_velocities=[]
kneeR_pos = []
kneeR_velocities=[]
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

prev_ankle_pos = None
prev_time = None
leg_length = 0.5  # in meters
leg_length_px = []

# Parameters for lucas kanade optical flow
# lk_params = dict(winSize=(15, 15), maxLevel=2,criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

prev_landmarks = None
with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.3, min_tracking_confidence=0.3) as pose:
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
            #right_elbow = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
            right_elbow = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
            #print(right_elbow)

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

            hipL =  (int(left_hip.x * frame_width), int(left_hip.y * frame_height))
            hipR =  (int(right_hip.x * frame_width), int(right_hip.y * frame_height))
            midPelvis = (int(((right_hip.x + left_hip.x)/2) * frame_width), int(((right_hip.y + left_hip.y)/2) * frame_height))

            elbL = (int(left_elbow.x * frame_width), int(left_elbow.y * frame_height))
            elbR = (int(right_elbow.x * frame_width), int(right_elbow.y * frame_height))

            wrL = (int(left_wr.x * frame_width), int(left_wr.y * frame_height))
            wrR = (int(right_wr.x * frame_width), int(right_wr.y * frame_height))

            shouL = (int(left_shoulder.x * frame_width), int(left_shoulder.y * frame_height))
            shouR = (int(right_shoulder.x * frame_width), int(right_shoulder.y * frame_height))

            nose = (int(noseOrg.x * frame_width), int(noseOrg.y * frame_height))

            #print(right_shoulder.z)
            """# Update landmarks
            current_landmarks = np.array([[left_ankle.x * frame.shape[1], left_ankle.y * frame.shape[0]],
                                          [right_ankle.x * frame.shape[1], right_ankle.y * frame.shape[0]]],
                                         dtype=np.float32)

            if prev_landmarks is not None:
                # Calculate optical flow (i.e., camera movement)
                new_landmarks, status, error = cv2.calcOpticalFlowPyrLK(prev_frame, frame, prev_landmarks, None,
                                                                        **lk_params)

                # Displacement due to camera movement
                displacement = new_landmarks - prev_landmarks

                # Adjust foot positions by removing the effect of camera movement
                adjusted_left_ankle = current_landmarks[0] - displacement[0]
                adjusted_right_ankle = current_landmarks[1] - displacement[1]

                # Use these adjusted positions for step length calculation
                # ...

            prev_landmarks = current_landmarks"""
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

            # Calculate leg length in pixels
            #if abs(hipL[0] - ankL[0])<10 and abs(hipL[0] - kneeL[0])<10:
            leg_length_px.append(euclidean_distance(hipR, kneeR))
            #print(leg_length_px)
            #print(results.pose_world_landmarks.landmark)

            """# Calculate speed
            if prev_ankle_pos is not None and prev_time is not None:
                current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # Get current time in seconds
                time_diff = current_time - prev_time

                # Horizontal velocity in pixels/frame
                horizontal_velocity_px = (ankL[0] - prev_ankle_pos[0]) / time_diff

                # Convert to real-world speed (m/s)
                speed = (horizontal_velocity_px / leg_length_px) * leg_length
                #print(f'ank: {ankL[0]:.2f}')
                #print(nose[0])
                print(f'Speed: {speed:.2f} m/s')
            """
            prev_ankle_pos = ankL
            prev_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

            tAng = compute_angle(hipR, kneeR, ankR)
            kneeR_angles.append(tAng)

            tAng = compute_angle(kneeL,midPelvis,kneeR)
            thigh_angles.append(tAng)

            #toeoff
            # max thigh seperation
            # works for toe off, could also do max distance between ankle and opposite knee
            # to get different feet/sides, just check which foot is in front of pelvis
            point1 = ankL
            point2 = kneeL
            point3 = hipL
            angle = compute_angle(point1, point2, point3)
            """print(frame_idx)
            print(ground)
            print(footL)
            print(footR)
            print(angle)
            print("")"""
            #if ((kneeR[0] - kneeL[0]) ** 2 + (kneeR[1] - kneeL[1]) ** 2) ** 0.5 > max_knee_seperation and footL[1]>ground:
            if ((kneeR[0] - kneeL[0]) ** 2 + (kneeR[1] - kneeL[1]) ** 2) ** 0.5 > max_knee_seperation and footL[1] + 5 >ground:
                max_knee_seperation = ((kneeR[0] - kneeL[0]) ** 2 + (kneeR[1] - kneeL[1]) ** 2) ** 0.5
                # key_frames = frame_idx
                kinogram[0] = frame_idx

            # max proj / vert
            #if abs(kneeL[0] - kneeR[0]) > 30 and abs(footL[1] - footR[1]) < height and abs(kneeL[1] - ankL[1]) < 50 and abs(kneeR[0] - ankR[0]) < 50:
            if abs(kneeL[0] - kneeR[0]) > 30 and abs(footL[1] - footR[1]) < height:
                height = abs(footL[1] - footR[1])
                kinogram[1] = frame_idx

            # full support left
            # should also check foot is on ground
            # possible check ankle score too
            if abs(ankL[0] - midPelvis[0]) < knee_hip_alignment_support and ankR[1] < ankL[1]:
                knee_hip_alignment_support = abs(ankL[0] - midPelvis[0])
                knee_ank_alignment_support = abs(kneeL[0] - ankL[0])
                pix = footL[1]-nose[1]
                if pix > height_in_pixels:
                    height_in_pixels = pix
                kinogram[4] = frame_idx


            # strike R
            if abs(kneeL[0] - hipL[0]) < knee_hip_alignment_strike and ankL[1] + 15 < ground:
                knee_hip_alignment_strike = abs(kneeL[0] - hipL[0])
                kinogram[2] = frame_idx

            # touch down
            if ((ankR[1] - ankL[1]) ** 2) ** 0.5 > max_y:
                max_y = ((ankR[1] - ankL[1]) ** 2) ** 0.5
                kinogram[3] = frame_idx

            #if abs(kneeL[0] - kneeR[0]) < 10 or (abs(elbL[0] - elbR[0]) < 10 and abs(kneeL[0] - kneeR[0]) < 25):
            if abs(kneeL[0] - kneeR[0]) < 25 and (abs(footL[1]-heelL[1])<10 or abs(footR[1]-heelR[1])<10):
                ground_contacts += 1
                if footL[1] > footR[1]:
                    ground = footL[1]
                    g1 = footL
                else:
                    ground = footR[1]
                    g1 = footR
                #ground = max(footL[1], footR[1])
                ground_frames.append(frame_idx) #take lowest point of the ground points in the group
                ground_points[frame_idx] = g1

            """if prev_left_foot and prev_right_foot:
                # Calculate the distance moved
                left_foot_movement = ((prev_left_foot[0] - footL[0]) ** 2 + (prev_left_foot[1] - footL[1]) ** 2) ** 0.5
                right_foot_movement = ((prev_right_foot[0] - footR[0]) ** 2 + (prev_right_foot[1] - footR[1]) ** 2) ** 0.5

                #print(f"Left Foot Movement: {left_foot_movement}, Right Foot Movement: {right_foot_movement}")

                # Check if the foot positions have changed significantly
                if (left_foot_movement <= threshold or right_foot_movement <= threshold) and abs(kneeL[0] - kneeR[0]) < 10:
                    ground_contacts += 1
                    ground = max(footL[1], footR[1])
                    print(f"Frame {int(cap.get(cv2.CAP_PROP_POS_FRAMES))}: Foot position changed")

                # Update the previous foot coordinates
            prev_left_foot = footL
            prev_right_foot = footR"""

        prev_frame = frame.copy()

        # Write the frame to the output video
        out.write(frame)
        output.append(frame)

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# looking at all knee positions, extract only the ones that are about to cross
left_knee_crossings = [i for i in range(1, len(kneeL_pos)) if (kneeL_pos[i-1] < mid_pelvis_pos[i] and kneeL_pos[i] >  mid_pelvis_pos[i] ) or (kneeL_pos[i-1] > mid_pelvis_pos[i]  and kneeL_pos[i] <  mid_pelvis_pos[i] )]
right_knee_crossings = [i for i in range(1, len(kneeR_pos)) if (kneeR_pos[i-1] < mid_pelvis_pos[i] and kneeR_pos[i] >  mid_pelvis_pos[i] ) or (kneeR_pos[i-1] > mid_pelvis_pos[i]  and kneeR_pos[i] <  mid_pelvis_pos[i] )]

# variables for kinogram feedback
feedback = []

# variables for ground point contact frames
imp_frames = []
contact = False
threshold = 100000
print("ground")
print(ground_points)
# Sort the dictionary by keys
ground_points_smooth = find_median_of_consecutive_keys(ground_points)
print(ground_points_smooth)
print(len(nose_pos))
print(len(mid_pelvis_pos))
tor_angles = torso_angles(nose_pos,mid_pelvis_pos, ground_points_smooth)
print(tor_angles)
"""contact = True
    ind = i
    prev_ind = i
    prev_dis = 0
    print(imp_frames)
    max_knee_seperation = 0
    key = 0
    threshold = 100000"""

cap = cv2.VideoCapture(video_path)
# Get the frame rate and frame size of the video
fps = cap.get(cv2.CAP_PROP_FPS)
print("fps " + str(fps))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4 files
#out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

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
            #print("thresh")

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
            hipR =  (int(right_hip.x * frame_width), int(right_hip.y * frame_height))
            midPelvis = (
            int(((right_hip.x + left_hip.x) / 2) * frame_width), int(((right_hip.y + left_hip.y) / 2) * frame_height))

            elbL = (int(left_elbow.x * frame_width), int(left_elbow.y * frame_height))
            elbR = (int(right_elbow.x * frame_width), int(right_elbow.y * frame_height))

            if frame_idx in kinogram:
                ind = kinogram.index(frame_idx)
                if ind == 0:
                    if ankL[1] > ankR[1]:
                        f = toe_off_feedback(ankL, kneeL, hipL, ankR, kneeR, hipR, footR, heelR)
                    else:
                        f = toe_off_feedback(ankR, kneeR, hipR, ankL, kneeL, hipL, footL, heelL)
                    #print("Toe off")
                    #print(f)
                elif ind == 1:
                    if ankL[0] > ankR[0]:
                        f = max_vert_feedback(ankL, kneeL, hipL, footL, ankR, kneeR, hipR)
                    else:
                        f = max_vert_feedback(ankR, kneeR, hipR, footR, ankL, kneeL, hipL)
                    #print("MV")
                    #print(f)
                elif ind == 2:
                    if ankL[0] > ankR[0]:
                        f = strike_feedback(ankL, kneeL, hipL, footL, ankR, kneeR, hipR)
                    else:
                        f = strike_feedback(ankR, kneeR, hipR, footR, ankL, kneeL, hipL)
                    #print("Strike")
                    #print(f)
                elif ind == 3:
                    if ankL[0] > ankR[0]:
                        f = touch_down_feedback(ankL, kneeL, hipL, heelL, footL, ankR, kneeR, hipR, footR, heelR)
                    else:
                        f = touch_down_feedback(ankR, kneeR, hipR, heelR, footR, ankL, kneeL, hipL, footL, heelL)
                    #print("TD")
                    #print(f)
                else:
                    if ankL[0] > ankR[0]:
                        f = full_supp_feedback(ankL, kneeL, hipL, footL, heelL, ankR, kneeR, hipR, footR, heelR)
                    else:
                        f = full_supp_feedback(ankR, kneeR, hipR, ankL, footR, heelR, kneeL, hipL, footL, heelL)
                    #print("FS")
                    #print(f)

        # Draw the pose annotation on the frame
        if results.pose_landmarks and contact == True:
            #print(frame_idx)
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
                #print('false1')
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

"""
Cooperation Analysis
"""
"""first_elements = [t[0] for t in kneeL_pos]
first_elements2 = [t[0] for t in kneeR_pos]
frames = range(len(thigh_angles))
plt.figure(figsize=(12, 6))
plt.plot(frames, first_elements, label='Left Knee')
plt.plot(frames, first_elements2, label='Right Knee')
#plt.axhline(y=1, color='k', linestyle='--', label='Midline')

# Mark crossings
for crossing in left_knee_crossings:
    plt.axvline(x=frames[crossing], color='b', linestyle='--', alpha=0.5)
for crossing in right_knee_crossings:
    plt.axvline(x=frames[crossing], color='r', linestyle='--', alpha=0.5)

plt.xlabel('Frame')
plt.ylabel('Horizontal Position (pixels)')
plt.title('Knee Positions and Midline Crossings')
plt.legend()
plt.show()

left_to_right_intervals = []
right_to_left_intervals = []

# Assuming frames are in chronological order
for crossing in left_knee_crossings:
    subsequent_right_crossings = [frames[i] for i in right_knee_crossings if frames[i] > frames[crossing]]
    if subsequent_right_crossings:
        left_to_right_intervals.append(subsequent_right_crossings[0] - frames[crossing])

for crossing in right_knee_crossings:
    subsequent_left_crossings = [frames[i] for i in left_knee_crossings if frames[i] > frames[crossing]]
    if subsequent_left_crossings:
        right_to_left_intervals.append(subsequent_left_crossings[0] - frames[crossing])

print('Left to Right Knee Intervals (frames):', left_to_right_intervals)
print('Right to Left Knee Intervals (frames):', right_to_left_intervals)
"""


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
#velocity_magnitudeL = np.linalg.norm(velocitiesL, axis=1)
#velocity_magnitudeR = np.linalg.norm(velocitiesR, axis=1)

# Create time axis
#time_velocity = np.arange(len(velocity_magnitudeL)) / fps

# Plot velocities on the same axis
"""plt.figure(figsize=(10, 6))
plt.plot(time_velocity, velocity_magnitudeL, label='Left Knee Velocity')
plt.plot(time_velocity, velocity_magnitudeR, label='Right Knee Velocity', color='orange')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (units/s)')
plt.title('Velocity Comparison of Left and Right Knees')
plt.legend()"""
#plt.show()

"""
Positional analysis
knee lift analysis
"""
elbL_coords = np.array(elbL_pos)
hipL_coords = np.array(hipL_pos)
kneeL_coords = np.array(kneeL_pos)
ankL_coords = np.array(ankL_pos)

elbR_coords = np.array(elbR_pos)
hipR_coords = np.array(hipR_pos)
kneeR_coords = np.array(kneeR_pos)
ankR_coords = np.array(ankR_pos)

# Extract x and y coordinates
elbL_x = elbL_coords[:, 0]
elbL_y = elbL_coords[:, 1]
elbR_x = elbR_coords[:, 0]
elbR_y = elbR_coords[:, 1]

hipL_x = hipL_coords[:, 0]
hipL_y = hipL_coords[:, 1]
hipR_x = hipR_coords[:, 0]
hipR_y = hipR_coords[:, 1]

kneeL_x = kneeL_coords[:, 0]
kneeL_y = kneeL_coords[:, 1]
kneeR_x = kneeR_coords[:, 0]
kneeR_y = kneeR_coords[:, 1]

ankL_x = ankL_coords[:, 0]
ankL_y = ankL_coords[:, 1]
ankR_x = ankR_coords[:, 0]
ankR_y = ankR_coords[:, 1]
"""
# Plot hip and knee positions
plt.figure(figsize=(10, 6))
#plt.plot(elbL_x, elbL_y, color='purple', label='Knee Left Position', marker='o', linestyle='-', alpha=0.7)
#plt.plot(elbR_x, elbR_y, color='red', label='Knee Right Position', marker='o', linestyle='-', alpha=0.7)
#plt.plot(hipL_x, hipL_y, color='pink', label='Hip Left Position', marker='o', linestyle='-', alpha=0.7)
#plt.plot(hipR_x, hipR_y, color='red', label='Hip Right Position', marker='o', linestyle='-', alpha=0.7)
#plt.plot(kneeL_x, kneeL_y, color='blue', label='Knee Left Position', marker='o', linestyle='-', alpha=0.7)
plt.plot(range(len(kneeR_x)), kneeR_x, color='orange', label='Knee Right Position', marker='o', linestyle='-', alpha=0.7)
#plt.plot(ankL_x, ankL_y, color='green', label='Ank Left Position',marker='o', linestyle='-', alpha=0.7)
plt.plot(range(len(ankR_x)), ankR_x, color='purple', label='Ank Right Position', marker='o', linestyle='-', alpha=0.7)

g = extract_consecutive_values(ground_points)
for interval in g:
    plt.axvline(x=interval[0], color='b', linestyle='--', alpha=0.5)
    plt.axvline(x=interval[2], color='b', linestyle='--', alpha=0.5)

plt.xlabel('Frame Index')
plt.ylabel('X Coordinate')
plt.title('Knee and Ankle X Positions')
plt.legend()
plt.grid(True)
plt.show()"""

"""# Plot thigh angles
# Generate x-values as indices
frames_x = range(len(thigh_angles))
# Create a scatter plot
plt.plot(frames_x, thigh_angles, color='blue', label='Data points')
# Add labels and title
plt.xlabel('Frame Index')
plt.ylabel('Angle (Degrees)')
plt.title('Thigh Angles over Frames')
# Add a legend
plt.legend()
plt.show()"""

# Angular velocities
thigh_ang_vel = angluar_velocity(thigh_angles, "Thigh")

kneeL_angles = []
for i in range(len(kneeL_pos)):
    a = compute_angle(hipL_pos[i],kneeL_pos[i],ankL_pos[i])
    kneeL_angles.append(a)
kneeL_ang_vel = angluar_velocity(kneeL_angles, "Knee Left")

kneeR_angles = []
for i in range(len(kneeR_pos)):
    a = compute_angle(hipR_pos[i],kneeR_pos[i],ankR_pos[i])
    kneeR_angles.append(a)
kneeR_ang_vel = angluar_velocity(kneeR_angles, "Knee Right")

elbL_angles = []
for i in range(len(elbL_pos)):
    a = compute_angle(shouL_pos[i],elbL_pos[i],wrL_pos[i])
    elbL_angles.append(a)
elbL_ang_vel = angluar_velocity(elbL_angles, "Elbow Left")

elbR_angles = []
for i in range(len(elbR_pos)):
    a = compute_angle(shouR_pos[i],elbR_pos[i],wrR_pos[i])
    elbR_angles.append(a)
elbR_ang_vel = angluar_velocity(elbR_angles, "Elbow Right")

hipR_angles = []
for i in range(len(hipR_pos)):
    a = compute_angle(shouR_pos[i],hipR_pos[i],kneeR_pos[i])
    hipR_angles.append(a)
hipR_ang_vel = angluar_velocity(hipR_angles, "Hip Right")

smoothed = lowess(elbR_ang_vel, range(len(elbR_ang_vel)), frac=0.3)
get_lag(elbR_ang_vel)
get_lag(smoothed[:,1])
"""
Summary Table
"""
# Create a dictionary to store the lists
key_data = {'AV_Thigh': thigh_ang_vel, 'AV_Knee_L': kneeL_ang_vel, 'AV_Knee_R': kneeR_ang_vel, 'AV_Elb_L': elbL_ang_vel, 'AV_Elb_R': elbR_ang_vel, 'AV_Hip_R': hipR_ang_vel}

# Create a DataFrame from the dictionary
df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in key_data.items()]))

# Calculate summary statistics
#summary = df.agg(['median', 'mean', 'min', 'max', standard_error, coefficient_of_variation, mean_lag]).transpose()
#summary.columns = ['Median', 'Mean', 'Min', 'Max', 'Standard Error', 'Coefficient of Variation', 'Mean Lag']

# Calculate built-in summary statistics
summary = df.agg(['median', 'mean', 'min', 'max']).transpose()
# Calculate custom statistics
summary['Standard Error'] = df.apply(standard_error)
summary['Coefficient of Variation'] = df.apply(coefficient_of_variation)
summary['Mean Lag'] = df.apply(mean_lag)

# Print the summary table
print(summary)

"""# Apply moving average with a window size of 3
window_size = 3
smoothed_values = moving_average(thigh_angles, window_size)
zero_crossings = detect_scissor(smoothed_values,10)

# Calculate time between zero crossings
time_between_crossings = [(frames[zero_crossings[i]] - frames[zero_crossings[i-1]])*(1/fps) for i in range(1, len(zero_crossings))]
print("cross")
print(time_between_crossings)
# Plotting
plt.figure(figsize=(14, 6))

# Scatter plot with zero crossings marked
plt.subplot(2, 1, 1)
plt.scatter(frames, thigh_angles, color='blue', label='Data points')
for zc in zero_crossings:
    plt.axvline(zc, color='r', linestyle='--', label='Crossing' if zc == zero_crossings[0] else "")
plt.xlabel('Frame Index')
plt.ylabel('Angle (Degrees)')
plt.title('Thigh Scissoring Cycle')
plt.legend()

# Plot time between zero crossings
plt.subplot(2, 1, 2)
plt.plot(range(1, len(time_between_crossings) + 1), time_between_crossings, marker='o', linestyle='-', color='green')
plt.xlabel('Crossing Index')
plt.ylabel('Time Between Crossings (indices)')
plt.title('Time Between Crossings')

plt.tight_layout()
plt.show()"""

"""
Step Length Analysis 
"""
#david 30-56
pix_distances = step_len_anal(ank_pos, imp_frames, output, leg_length_px,leg_length)
print(pix_distances)

print(height_in_pixels)
#sLength = step_length(1.77,height_in_pixels, pix_distances, results, 0, 0, (581, 460),(678, 460))
#print('lenBelow')
#print(sLength)

#print(imp_frames)
#print(ground_frames)
#print(kinogram)

"""
Show important frames
"""
"""# fly issues from 47 ends too early at 53
# david a bit too late 3 - 16 (should be 14ish)
# adam ends early 16 - 18
imp_frames = [17, 29, 45, 58, 66]
#imp_frames = kinogram
num_frames = len(imp_frames)
fig, axs = plt.subplots(1, num_frames, figsize=(num_frames * 5, 5))

for i in range(num_frames):
    axs[i].imshow(output[imp_frames[i]])
    axs[i].axis('off')
    axs[i].set_title(f"Contact {i + 1}")

plt.tight_layout()
plt.show()"""
"""
# Coordinates to draw
points = [(0, 459.0), (frame_width, 465.5),(0, 465.5), (frame_width, 439.0)]
# Plot the image
plt.imshow(output[45])
# Draw the points
x_coords, y_coords = zip(*points)
plt.scatter(x_coords, y_coords, color='red', s=100)  # s is the size of the points
# Show the plot
plt.show()"""

"""
Slope analysis
"""
"""forearm_slopes = []
for i in range(len(elbR_pos)):
    point1 = elbR_pos[i]
    point2 = wrR_pos[i]
    forearm_slopes.append(compute_slope(point1, point2))

thigh_slopes = []
for i in range(len(kneeR_pos)):
    point1 = hipR_pos[i]
    point2 = kneeR_pos[i]
    thigh_slopes.append(compute_slope(point1, point2))

frames_x = range(len(thigh_slopes))
# Create a scatter plot
plt.plot(frames_x, thigh_slopes, color='blue', label='Thigh')
plt.plot(frames_x, forearm_slopes, color='red', label='Forearm')
# Add labels and title
plt.xlabel('Frame Index')
plt.ylabel('Slope')
plt.title('Slope over Frames')
# Add a legend
plt.legend()
plt.show()"""

"""
Displacement Analysis
"""

dis = []
for i in range(len(ankR_pos)):
    point1 = kneeR_pos[i]
    point2 = kneeL_pos[i]
    dis.append(euclidean_distance(point1, point2))

frames_x = range(len(dis))
# Create a scatter plot
plt.plot(frames_x, dis, color='red', label='KneeR - KneeL')
g = extract_consecutive_values(ground_points)
for interval in g:
    plt.axvline(x=interval[0], color='b', linestyle='--', alpha=0.5)
    plt.axvline(x=interval[2], color='b', linestyle='--', alpha=0.5)
# Add labels and title
plt.xlabel('Frame Index')
plt.ylabel('Distance (Pixels')
plt.title('Distance between Knees over Frames')
# Add a legend
plt.legend()
plt.show()

"""
Flight and ground contact time analysis 
"""
f_g_times = contact_flight_analysis(imp_frames,fps,len(output)/fps)

ground_times = f_g_times[1]
flight_times = f_g_times[0]

# watch out for last flight time is always 0
max_step_len = max(pix_distances)
avg_ground_time = sum(ground_times)/len(ground_times)
avg_flight_time = sum(flight_times)/(len(flight_times)-1)
time_btw_steps = 0
print(avg_ground_time)

"""import numpy as np
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.optimize import curve_fit

# Generate noisy sine wave
x = np.arange(1, 10.1, 0.1)
a = 4 * np.sin(x)
a = a + np.random.randn(len(a)) + np.random.randn(len(a)) + np.random.randn(len(a))

# Define span and weights
span = 30
weight = [(1 - abs((j - (span + 1) / 2) / (span / 2 + 1))**3)**3 for j in range(1, span + 1)]

# Plot the weights
plt.figure(figsize=(10, 6))
plt.plot(range(1, span + 1), weight, marker='o', linestyle='-', color='b')
plt.title('Weights for LOESS Smoothing')
plt.xlabel('Index')
plt.ylabel('Weight')
plt.grid(True)
plt.show()

# Apply LOESS
b = lowess(a, x, frac=span / len(a), return_sorted=False)

# Function for cubic polynomial
def cubic_poly(x, a, b, c, d):
    return a * x**3 + b * x**2 + c * x + d

# Weighted cubic polynomial fit
d = []
e = []
print("lenA" + str(len(a)))
for i in range(len(a)):
    start = int(i - span / 2)
    end = int(i + span / 2)

    #x_range = x[start:end]
    #a_range = a[start:end]

    k=0
    o=30
    if (start <= 0):
        k = 0 - start
        start = 0
    if (end >= len(a)):
        end = len(a)-1
        o = end - start

    weights = weight[k:o]
    print(start)
    print(end)
    print(len(weights))
    # Weighted fit
    popt, _ = curve_fit(cubic_poly,  x[start:end], a[start:end], sigma=weights)
    d.append(cubic_poly(x[i], *popt))

    # Unweighted fit
    popt_unweighted, _ = curve_fit(cubic_poly,  x[start:end],  a[start:end])
    e.append(cubic_poly(x[i], *popt_unweighted))

# Plot results
plt.plot(x, a, 'o', label='Noisy Data')
plt.plot(x, b, '--', linewidth=3, label='LOESS')
plt.plot(x, d, 'k', linewidth=1.5, label='Weighted Cubic Fit')
plt.plot(x, e, 'r', linewidth=1.5, label='Unweighted Cubic Fit')
plt.legend()
plt.show()"""

import numpy as np
import time
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

"""
xx = np.arange(1, 101, 1)
a = 4 * np.sin(0.1 * xx)
a = a + np.random.randn(len(a)) + np.random.randn(len(a)) + np.random.randn(len(a))

loess = Loess(xx, a)

ys=[]
for x in xx:
    y = loess.estimate(x, window=30, use_matrix=False, degree=3)
    ys.append(y)
    # print(x, y)"""

xx = [(frame_index / 30) * 1000 for frame_index in range(len(elbR_angles))]
print(xx)
loess = Loess(xx, hipR_angles)
ys=[]
for x in xx:
    y = loess.estimate(x, window=15, use_matrix=False, degree=3)
    ys.append(y)

plt.figure(figsize=(12, 8))
plt.plot(xx, hipR_angles, 'o', label='Raw Data')
plt.plot(xx, ys, '--', linewidth=3, label='LOESS')
plt.legend()
plt.title('Right Hip Angles Over Time')
plt.xlabel('Time (ms)')
plt.ylabel('Angle (Degrees)')
plt.grid(True)
plt.show()