#!pip install -q mediapipe
#!wget -O pose_landmarker.task -q https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2
import matplotlib.pyplot as plt


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
    plt.show()

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
        return mid
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
Main
"""

# STEP 1: Import the necessary modules.
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Upload Video
video_path = 'david.mov'
output_path = 'output_video.mp4'
cap = cv2.VideoCapture(video_path)

# Get the frame rate and frame size of the video
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4 files
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

#smoothness
kneeL_pos = []
kneeL_velocities=[]
kneeR_pos = []
kneeR_velocities=[]


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
            right_elbow = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]

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
            midPelvis = (int(((right_hip.x + left_hip.x)/2) * frame_width), int(((right_hip.y + left_hip.y)/2) * frame_height))

            elbL = (int(left_elbow.x * frame_width), int(left_elbow.y * frame_height))
            elbR = (int(right_elbow.x * frame_width), int(right_elbow.y * frame_height))

            nose = (int(noseOrg.x * frame_width), int(noseOrg.y * frame_height))

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
                ground = max(footL[1], footR[1])
                ground_frames.append(frame_idx) #take lowest point of the ground points in the group
                ground_points[frame_idx] = ground

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

imp_frames = []
contact = False
threshold = 100000
print(ground_points)
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
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4 files
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

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
            threshold = ground_points[frame_idx]
            #print("thresh")

        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame to get pose landmarks
        results = pose.process(rgb_frame)

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
        out.write(frame)
        output.append(frame)

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()


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

# Plot velocities on the same axis
plt.figure(figsize=(10, 6))
plt.plot(time_velocity, velocity_magnitudeL, label='Left Knee Velocity')
plt.plot(time_velocity, velocity_magnitudeR, label='Right Knee Velocity', color='orange')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (units/s)')
plt.title('Velocity Comparison of Left and Right Knees')
plt.legend()
plt.show()

"""
Step Length Analysis 
"""
#david 30-56
pix_distances = step_len_anal(ank_pos, imp_frames, output)
print(pix_distances)

print(height_in_pixels)
sLength = step_length(1.77,height_in_pixels, pix_distances, results, 0, 0, (581, 460),(678, 460))
print('lenBelow')
print(sLength)

print(imp_frames)
print(ground_frames)
print(kinogram)

"""
Show important frames
"""
# fly issues from 47 ends too early at 53
# david a bit too late 3 - 16 (should be 14ish)
# adam ends early 16 - 18
#imp_frames = [28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,51,52,53, 54, 55, 56, 57]
#imp_frames = kinogram
num_frames = len(imp_frames)
fig, axs = plt.subplots(1, num_frames, figsize=(num_frames * 5, 5))

for i in range(num_frames):
    axs[i].imshow(output[imp_frames[i]])
    axs[i].axis('off')
    axs[i].set_title(f"Contact {i + 1}")

plt.tight_layout()
plt.show()

"""
Flight and ground contact time analysis 
"""
contact_flight_analysis(imp_frames,1,1)
