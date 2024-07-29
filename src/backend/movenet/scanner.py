from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2
import matplotlib.pyplot as plt

# STEP 1: Import the necessary modules.
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Function to calculate Euclidean distance
def euclidean_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def get_dim(img_path, leg_length):
    # convert from inch to cm
    print(leg_length)
    leg_length = float(leg_length)*2.54

    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    # Load the image
    #img = cv2.imread(img_path)
    image_data = img_path.read()
    np_image = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

    # Check if the image is loaded correctly
    if img is None:
        raise ValueError("Image not found or unable to load.")

    # Get the dimensions of the image
    frame_height, frame_width = img.shape[:2]

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
    print(euclidean_distance(ankL,hipL)*(leg_length/euclidean_distance(ankL,eyeL)))
    return (euclidean_distance(ankL,hipL)*(leg_length/euclidean_distance(ankL,eyeL)))
    # Plot the image
    #plt.imshow(rgb_frame)
    # Draw the points
    #plt.show()