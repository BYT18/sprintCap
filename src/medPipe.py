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

"""
Main
"""

# STEP 1: Import the necessary modules.
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

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
            if ((kneeR[0] - kneeL[0]) ** 2 + (kneeR[1] - kneeL[1]) ** 2) ** 0.5 > max_knee_seperation and footL[1]>ground:
                max_knee_seperation = ((kneeR[0] - kneeL[0]) ** 2 + (kneeR[1] - kneeL[1]) ** 2) ** 0.5
                # key_frames = frame_idx
                kinogram[0] = frame_idx

            # max proj / vert
            if abs(kneeL[0] - kneeR[0]) > 30 and abs(ankL[1] - ankR[1]) < height and abs(kneeL[1] - ankL[1]) < 50 and abs(kneeR[0] - ankR[0]) < 50:
                height = abs(ankL[1] - ankR[1])
                kinogram[1] = frame_idx

            # full support left
            # should also check foot is on ground
            # possible check ankle score too
            if abs(ankL[0] - midPelvis[0]) < knee_hip_alignment_support and ankR[1] < ankL[1]:
                knee_hip_alignment_support = abs(ankL[0] - midPelvis[0])
                knee_ank_alignment_support = abs(kneeL[0] - ankL[0])
                kinogram[4] = frame_idx


            # strike R
            if abs(kneeL[0] - hipL[0]) < knee_hip_alignment_strike and ankL[1] + 15 < ground:
                knee_hip_alignment_strike = abs(kneeL[0] - hipL[0])
                kinogram[2] = frame_idx

            # touch down
            if ((ankR[1] - ankL[1]) ** 2) ** 0.5 > max_y:
                max_y = ((ankR[1] - ankL[1]) ** 2) ** 0.5
                kinogram[3] = frame_idx

            print(frame_idx)
            print(kneeL[0] - kneeR[0])
            print(footL[1]-heelL[1])
            print(footR[1]-heelR[1])
            print("")
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
            print("thresh")

        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame to get pose landmarks
        results = pose.process(rgb_frame)

        # Draw the pose annotation on the frame
        if results.pose_landmarks and contact == True:
            print(frame_idx)
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
                print('false1')
                contact = False
            else:

                # ensure hand position is sufficiently far (this will ensure that no stoppages occur when knees are close
                # left foot is on the ground
                if (ankR[1] - ankL[1] <= 0):
                        #checking if the distance between knees does not change enough then that means we are at toe off frame
                    if abs(footL[1] - threshold)<10:
                        imp_frames.append(frame_idx)
                    else:
                        print(frame_idx)
                        print(footL[1] - threshold)
                        print('false2')
                        contact = False
                else:
                    if abs(footR[1] - threshold)<10:
                        imp_frames.append(frame_idx)
                    else:
                        print(frame_idx)
                        print(footR[1] - threshold)
                        print('false3')
                        contact = False
                print("")

        # Write the frame to the output video
        out.write(frame)
        output.append(frame)

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

print(imp_frames)
print(ground_frames)
print(kinogram)
# fly issues from 47 ends too early at 53
# david a bit too late 3 - 16 (should be 14ish)
# adam ends early 16 - 18
#imp_frames = [53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69]
num_frames = len(imp_frames)
fig, axs = plt.subplots(1, num_frames, figsize=(num_frames * 5, 5))

for i in range(num_frames):
    axs[i].imshow(output[imp_frames[i]])
    axs[i].axis('off')
    axs[i].set_title(f"Contact {i + 1}")

plt.tight_layout()
plt.show()

contact_flight_analysis(imp_frames,1,1)

"""plt.imshow(output[49])
plt.axis('off')
plt.show()"""