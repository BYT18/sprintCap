#!pip install -q mediapipe
#!wget -O pose_landmarker.task -q https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task

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

def contact_flight_analysis(frames, fps, tot_frames):

    imp_frames = frames
    #tot_frames = dur*curr_fps(slow-mo)
    #Total frames in the original video = Frame rate × Duration = 240 fps × 3 s = 720 frames
    tot_frames = 3*720
    #Frame rate of the GIF = Total frames of GIF / Duration = 93 frames / 3 s
    fps = 93 / 3
    #Time per frame in GIF = Duration / Total frames of GIF = 3 s / 93 frames
    tbf = 1 / fps
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

    # Plotting the first set of bars
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
    #plt.show()

    plot_path = os.path.join('media/pics', 'contact_time.png')
    plt.savefig(plot_path)
    return [flight_times,gcontact_times]


def step_len_anal(ank_pos, frames, output_frames):
    imp_frames = frames
    s_lens = []

    initial = 0
    for i in range(len(imp_frames) - 1):
        if imp_frames[i] + 1 == imp_frames[i + 1] and initial == 0:
            #initial = left or right ankle position
            marker_pos = obj_detect(output_frames[imp_frames[i]],0)
            initial = abs(ank_pos[i][0] - marker_pos)
            #initial = ank_pos[i][0]

        elif imp_frames[i] + 1 == imp_frames[i + 1] and initial != 0:
            continue
        else:
            marker_pos = obj_detect(output_frames[imp_frames[i]], 0)
            s_lens.append(abs(abs(ank_pos[i+1][0]-marker_pos) - initial))
            #s_lens.append(ank_pos[i + 1][0] - initial)
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

        return mid

"""
Main
"""
def get_gif(vid):
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
    hipL_pos = []
    hipR_pos = []
    kneeL_pos = []
    kneeL_velocities = []
    kneeR_pos = []
    kneeR_velocities = []
    thigh_angles = []

    # variables for ground contacts
    ground_points={}
    ground_frames = []
    ground_contacts = 0


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
                noseOrg = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
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
                hipR = (int(right_hip.x * frame_width), int(right_hip.y * frame_height))
                midPelvis = (int(((right_hip.x + left_hip.x)/2) * frame_width), int(((right_hip.y + left_hip.y)/2) * frame_height))

                elbL = (int(left_elbow.x * frame_width), int(left_elbow.y * frame_height))
                elbR = (int(right_elbow.x * frame_width), int(right_elbow.y * frame_height))

                nose = (int(noseOrg.x * frame_width), int(noseOrg.y * frame_height))

                kneeL_pos.append(kneeL)
                kneeR_pos.append(kneeR)
                hipL_pos.append(hipL)
                hipR_pos.append(hipR)

                tAng = compute_angle(kneeL, midPelvis, kneeR)
                thigh_angles.append(tAng)

                #toeoff
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
                    ground = max(footL[1], footR[1])
                    ground_frames.append(frame_idx) #take lowest point of the ground points in the group
                    ground_points[frame_idx] = ground

            # Write the frame to the output video
            out.write(frame)
            output.append(frame)

        # Release resources
        #cap.release()
        #out.release()
        cv2.destroyAllWindows()

    # variables for kinogram feedback
    feedback = {}

    imp_frames = []
    contact = False
    threshold = 100000

    cap = cv2.VideoCapture(video_path)
    # Get the frame rate and frame size of the video
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4 files
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.3, min_tracking_confidence=0.3) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Get the current frame number
            frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1

            if frame_idx in ground_frames:
                contact = True
                threshold = ground_points[frame_idx]

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
                        print("Toe off")
                        print(f)
                        feedback["TO"]=f
                    elif ind == 1:
                        if ankL[0] > ankR[0]:
                            f = max_vert_feedback(ankL, kneeL, hipL, footL, ankR, kneeR, hipR)
                        else:
                            f = max_vert_feedback(ankR, kneeR, hipR, footR, ankL, kneeL, hipL)
                        print("MV")
                        print(f)
                        feedback["MV"] = f
                    elif ind == 2:
                        if ankL[0] > ankR[0]:
                            f = strike_feedback(ankL, kneeL, hipL, footL, ankR, kneeR, hipR)
                        else:
                            f = strike_feedback(ankR, kneeR, hipR, footR, ankL, kneeL, hipL)
                        print("Strike")
                        print(f)
                        feedback["S"]=f
                    elif ind == 3:
                        if ankL[0] > ankR[0]:
                            f = touch_down_feedback(ankL, kneeL, hipL, heelL, footL, ankR, kneeR, hipR, footR, heelR)
                        else:
                            f = touch_down_feedback(ankR, kneeR, hipR, heelR, footR, ankL, kneeL, hipL, footL, heelL)
                        print("TD")
                        print(f)
                        feedback["TD"]=f
                    else:
                        if ankL[0] > ankR[0]:
                            f = full_supp_feedback(ankL, kneeL, hipL, footL, heelL, ankR, kneeR, hipR, footR, heelR)
                        else:
                            f = full_supp_feedback(ankR, kneeR, hipR, ankL, footR, heelR, kneeL, hipL, footL, heelL)
                        print("FS")
                        print(f)
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
                        else:
                            contact = False
                    else:
                        if abs(footR[1] - threshold)<10:
                            imp_frames.append(frame_idx)
                        else:
                            contact = False

            # Write the frame to the output video
            out.write(frame)
            output.append(frame)

        # Release resources
        cap.release()
        out.release()
        cv2.destroyAllWindows()

        # Now, convert the output video to MP4 using ffmpeg
    try:
            v_path = os.path.join('media/pics', 'output_video.mp4')
            subprocess.run(['ffmpeg', '-y', '-i', output_path, '-vcodec', 'h264', '-acodec', 'aac', v_path], check=True)

            print("Conversion to MP4 successful.")
    except subprocess.CalledProcessError as e:
            print(f"Conversion to MP4 failed: {e}")

    for i in range(len(kinogram)):
        # Convert the frame from BGR to RGB
        rgb_frame = cv2.cvtColor(output[kinogram[i]], cv2.COLOR_BGR2RGB)

        # Create a plot
        plt.figure(figsize=(10, 6))
        plt.imshow(rgb_frame)
        plt.axis('off')

        # Save the plot
        plot_path = os.path.join('media/pics', f'key_frame_{i + 1}.png')
        plt.savefig(plot_path, bbox_inches='tight', pad_inches=0)
        #plt.savefig(f'key_frame_{i + 1}.png')
        #plt.close()

    """
    Velocity and Smoothness Analysis 
    """
    # Calculate velocities
    velocitiesL = calculate_velocity(kneeL_pos, fps)
    velocitiesR = calculate_velocity(kneeR_pos, fps)
    print(velocitiesL)

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

    num_frames = len(imp_frames)
    fig, axs = plt.subplots(1, num_frames, figsize=(num_frames * 5, 5))

    for i in range(num_frames):
        axs[i].imshow(output[imp_frames[i]])
        axs[i].axis('off')
        axs[i].set_title(f"Contact {i + 1}")

    plt.tight_layout()
    #plt.show()

    """
    Step Length Analysis 
    """
    # david 30-56
    #pix_distances = step_len_anal(ank_pos, imp_frames, output)
    #sLength = step_length(1.77, height_in_pixels, pix_distances, results, 0, 0, (581, 460), (678, 460))

    """
    Flight and ground contact time analysis 
    """
    f_g_times = contact_flight_analysis(imp_frames, 1, 1)

    ground_times = f_g_times[1]
    flight_times = f_g_times[0]

    # watch out for last flight time is always 0
    #max_step_len = max(sLength)
    avg_ground_time = sum(ground_times) / len(ground_times)
    avg_flight_time = sum(flight_times) / (len(flight_times) - 1)
    time_btw_steps = 0
    #print(avg_ground_time)

    #return output_path
    #return [1, 2, 3, 5, 8, 10]
    # could return dictionary whcih would be easier to read
    #"vL": velocity_magnitudeL, "vR": velocity_magnitudeR, "vT": time_velocity
    return {"ground":ground_times,"flight":flight_times,"kneePos":kneeL_pos,"feedback":feedback,
            "avg_f":avg_flight_time, "avg_g":avg_ground_time, "ang":thigh_angles,
            "vL": velocity_magnitudeL_list, "vR": velocity_magnitudeR_list, "vT": time_velocity_list}
    #return [ground_times,flight_times]
