import tensorflow as tf
import tensorflow_hub as hub
from tensorflow_docs.vis import embed
import numpy as np
import cv2
from matplotlib.collections import LineCollection
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import imageio
from IPython.display import HTML, display
from tqdm import tqdm
import math

#https://analyticsindiamag.com/how-to-do-pose-estimation-with-movenet/
angles = []
ankle_positions = []
threshold = 3
switch = 0
switches = []
key_frames = 0

# Dictionary to map joints of body part
KEYPOINT_DICT = {
 'nose':0,
 'left_eye':1,
 'right_eye':2,
 'left_ear':3,
 'right_ear':4,
 'left_shoulder':5,
 'right_shoulder':6,
 'left_elbow':7,
 'right_elbow':8,
 'left_wrist':9,
 'right_wrist':10,
 'left_hip':11,
 'right_hip':12,
 'left_knee':13,
 'right_knee':14,
 'left_ankle':15,
 'right_ankle':16
}

# map bones to matplotlib color name
KEYPOINT_EDGE_INDS_TO_COLOR = {
 (0,1): 'm',
 (0,2): 'c',
 (1,3): 'm',
 (2,4): 'c',
 (0,5): 'm',
 (0,6): 'c',
 (5,7): 'm',
 (7,9): 'm',
 (6,8): 'c',
 (8,10): 'c',
 (5,6): 'y',
 (5,11): 'm',
 (6,12): 'c',
 (11,12): 'y',
 (11,13): 'm',
 (13,15): 'm',
 (12,14): 'c',
 (14,16): 'c'
}

def _keypoints_and_edges_for_display(keypoints_with_score,height,
                                  width,keypoint_threshold=0.11):
   """Returns high confidence keypoints and edges"""
   keypoints_all = []
   keypoint_edges_all = []
   edge_colors = []
   num_instances,_,_,_ = keypoints_with_score.shape
   for id in range(num_instances):
     kpts_x = keypoints_with_score[0,id,:,1]
     kpts_y = keypoints_with_score[0,id,:,0]
     kpts_scores = keypoints_with_score[0,id,:,2]
     kpts_abs_xy = np.stack(
         [width*np.array(kpts_x),height*np.array(kpts_y)],axis=-1)
     kpts_above_thrs_abs = kpts_abs_xy[kpts_scores > keypoint_threshold,: ]
     keypoints_all.append(kpts_above_thrs_abs)
     for edge_pair,color in KEYPOINT_EDGE_INDS_TO_COLOR.items():
       if (kpts_scores[edge_pair[0]] > keypoint_threshold and
           kpts_scores[edge_pair[1]] > keypoint_threshold):
         x_start = kpts_abs_xy[edge_pair[0],0]
         y_start = kpts_abs_xy[edge_pair[0],1]
         x_end = kpts_abs_xy[edge_pair[1],0]
         y_end = kpts_abs_xy[edge_pair[1],1]
         lien_seg = np.array([[x_start,y_start],[x_end,y_end]])
         keypoint_edges_all.append(lien_seg)
         edge_colors.append(color)
   if keypoints_all:
     keypoints_xy = np.concatenate(keypoints_all,axis=0)
   else:
     keypoints_xy = np.zeros((0,17,2))
   if keypoint_edges_all:
     edges_xy = np.stack(keypoint_edges_all,axis=0)
   else:
     edges_xy = np.zeros((0,2,2))
   return keypoints_xy,edges_xy,edge_colors

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

def draw_prediction_on_image(image, keypoints_with_scores, ground, crop_region=None, close_figure=False, output_image_height=None):
   """Draws the keypoint predictions on image"""
   height, width, channel = image.shape
   aspect_ratio = float(width) / height
   fig, ax = plt.subplots(figsize=(12 * aspect_ratio, 12))

   # To remove the huge white borders
   fig.tight_layout(pad=0)
   ax.margins(0)
   ax.set_yticklabels([])
   ax.set_xticklabels([])
   plt.axis('off')
   im = ax.imshow(image)

   line_segments = LineCollection([], linewidths=(3), linestyle='solid')
   ax.add_collection(line_segments)

   # Turn off tick labels
   scat = ax.scatter([], [], s=60, color='#FF1493', zorder=3)
   (keypoint_locs, keypoint_edges,
    edge_colors) = _keypoints_and_edges_for_display(
        keypoints_with_scores, height, width)
   #print(keypoint_edges)
   #ground_line = np.array([[[0, height * 0.9], [width, height * 0.9]]])
   #print(ground_line)
   #np.vstack([keypoint_edges, ground_line])
   #print(keypoint_edges)
   line_segments.set_segments(keypoint_edges)
   line_segments.set_color(edge_colors)

   #add angle for knee
   kneeR = {'x': keypoints_with_scores[0][0][13][0], "y": keypoints_with_scores[0][0][13][1],
            "score": keypoints_with_scores[0][0][13][2]}
   kneeL = {'x': keypoints_with_scores[0][0][14][0], "y": keypoints_with_scores[0][0][14][1],
            "score": keypoints_with_scores[0][0][14][2]}
   midPelvis = {'x': (keypoints_with_scores[0][0][12][0] + keypoints_with_scores[0][0][11][0])/2,
                "y": (keypoints_with_scores[0][0][12][1] + keypoints_with_scores[0][0][11][1])/2,
                "score": keypoints_with_scores[0][0][12][2] - keypoints_with_scores[0][0][11][2]}

   #angle = (math.atan2(kneeL['y'] - midPelvis['y'], kneeL['x'] - midPelvis['x']) -
   #         math.atan2(kneeR['y'] - midPelvis['y'], kneeR['x'] - midPelvis['x'])) * (180 / math.pi)


   x1, y1 = kneeR['x']*width, kneeR['y']*height
   x2, y2 = kneeL['x']*width, kneeL['y']*height
   x3, y3 = midPelvis['x'] * width, midPelvis['y'] * height

   point1 = [x1, y1]
   point2 = [x2, y2]
   point3 = [x3, y3]
   angle = compute_angle(point1, point2, point3)

   if (len(keypoint_locs)) > 16:
       # Convert points to numpy arrays for easier calculations
       points = np.array([keypoint_locs[13], keypoint_locs[14], (keypoint_locs[11] + keypoint_locs[12])/2])
       # Create a polygon patch representing the triangle
       triangle = patches.Polygon(points, closed=True, fill=True, edgecolor='r', linewidth=5)
       # Add the triangle patch to the plot
       ax.add_patch(triangle)
   # arc should only be for a joint, trangle should be used for mid pelvis to knees

   '''
   # Calculate the center of the arc
   center_x = (x1 + x2) / 2
   center_y = (y1 + y2) / 2
   center_x = x3
   center_y = y3
   # Calculate the radius of the arc (e.g., half the distance between keypoint1 and keypoint2)
   radius = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5 / 2
   #radius = 100
   # Calculate the start and end angles of the arc
   start_angle = 0  # Adjust as needed
   end_angle = angle  # Adjust as needed
   #end_angle = 90
   # Draw the arc
   arc = patches.Arc((center_x, center_y), 2 * radius, 2 * radius, angle=0, theta1=start_angle, theta2=end_angle,
             color='c', linewidth=5)  # Adjust color as needed

   ax.add_patch(arc)
'''
   # Add a line for the ground
   #ground_line = np.array([[[0, height * 0.9], [width, height * 0.9]]])
   if ground != None:
       #ground_line = np.array([[ground[0], ground[1]]])
       #ground_line = np.array([[ground[i], ground[i + 1]] for i in range(len(ground) - 1)])
       ground_line = np.array([[[0,  sum(ground) / len(ground)], [width,sum(ground) / len(ground)]]])
   #ground_line = np.array([[[450, 100], [503, 100]]])
   #ground_color = '#008000'  # Green color for the ground

   # Reshape ground_line to match the number of vertices per line segment
   #ground_line = ground_line.reshape(-1, 3)
       x = np.vstack([keypoint_edges, ground_line])
   #x = np.vstack([keypoint_edges])
   #print(x)
   else:
       x = np.vstack([keypoint_edges])

   line_segments.set_segments(x)
   ground_color = 'g'
   #edge_colors.append(ground_color)
   #y = np.concatenate([edge_colors, [ground_color]])
   #print(np.concatenate([edge_colors, [ground_color]]))
   line_segments.set_color(np.concatenate([edge_colors, [ground_color]]))
   #line_segments.set_color(edge_colors)


   if keypoint_edges.shape[0]:
     line_segments.set_segments(keypoint_edges)
     line_segments.set_color(edge_colors)
     line_segments.set_segments(x)
     line_segments.set_color(np.concatenate([edge_colors, [ground_color]]))

   if keypoint_locs.shape[0]:
     scat.set_offsets(keypoint_locs)
   if crop_region is not None:
     xmin = max(crop_region['x_min'] * width, 0.0)
     ymin = max(crop_region['y_min'] * height, 0.0)
     rec_width = min(crop_region['x_max'], 0.99) * width - xmin
     rec_height = min(crop_region['y_max'], 0.99) * height - ymin
     rect = patches.Rectangle(
         (xmin,ymin),rec_width,rec_height,
         linewidth=1,edgecolor='b',facecolor='none')
     ax.add_patch(rect)
   fig.canvas.draw()
   #image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
   image_from_plot = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
   # Determine the width and height of your canvas or image
   #width, height = 2128, 1200  # Adjust these values based on your actual dimensions

    #giffy2
   #width, height = 2132, 1200
   #fly.gif
   width, height = 855, 1200

   #width, height = 2133, 1200

   # Adjust reshape based on the size of the array
   #image_from_plot = image_from_plot.reshape((height, width, -1))
   # Determine the width and height of your canvas or image

   #print("Actual array size:", image_from_plot.size)
   #print(fig.canvas.get_width_height()[::-1] + (3,))

   #width, height = 1200, 1200
   # Determine the width and height of your canvas or image
   #width, height = 675, 1200
   # Adjust reshape based on the size of the array
   #image_from_plot = image_from_plot.reshape((height, width, -1))
   #image_from_plot = image_from_plot.reshape((width,height, 3))
   # Adjust reshape based on the size of the array

   image_from_plot = image_from_plot.reshape((height, width, -1))
   #image_from_plot = image_from_plot.reshape(
   #    fig.canvas.get_width_height()[::-1] + (3,))
   plt.close(fig)
   if output_image_height is not None:
     output_image_width = int(output_image_height / height * width)
     image_from_plot = cv2.resize(
         image_from_plot, dsize=(output_image_width, output_image_height),
          interpolation=cv2.INTER_CUBIC)
   return image_from_plot


def draw_nonfigure_prediction(image, keypoints_with_scores, crop_region=None, close_figure=False, output_image_height=None):
   """Draws the keypoint predictions on image"""
   height, width, channel = image.shape
   aspect_ratio = float(width) / height
   fig, ax = plt.subplots(figsize=(12 * aspect_ratio, 12))
   # To remove the huge white borders
   fig.tight_layout(pad=0)
   ax.margins(0)
   ax.set_yticklabels([])
   ax.set_xticklabels([])
   plt.axis('off')
   im = ax.imshow(image)
   line_segments = LineCollection([], linewidths=(4), linestyle='solid')
   ax.add_collection(line_segments)
   # Turn off tick labels
   scat = ax.scatter([], [], s=60, color='#FF1493', zorder=3)
   (keypoint_locs, keypoint_edges,
    edge_colors) = _keypoints_and_edges_for_display(
        keypoints_with_scores, height, width)

   #print(keypoint_edges)
   #print(ground)
   x = np.vstack([keypoint_edges])
   #x = np.vstack([keypoint_edges, ground])
   line_segments.set_segments(x)
   #ground_color = 'c'
   #line_segments.set_color(np.concatenate([edge_colors, [ground_color]]))
   #line_segments.set_segments(keypoint_edges)
   line_segments.set_color(edge_colors)

   """ 
   # Add a line for the ground
   ground_line = np.array([[[0, height * 0.9], [width, height * 0.9]]])
   ground_color = 'c'
   x = np.vstack([keypoint_edges, ground_line])
   line_segments.set_segments(x)
   line_segments.set_color(np.concatenate([edge_colors, [ground_color]]))
   """

   if keypoint_edges.shape[0]:
     line_segments.set_segments(keypoint_edges)
     line_segments.set_color(edge_colors)
     line_segments.set_segments(x)
     #line_segments.set_color(np.concatenate([edge_colors, [ground_color]]))

   if keypoint_locs.shape[0]:
     scat.set_offsets(keypoint_locs)
   if crop_region is not None:
     xmin = max(crop_region['x_min'] * width, 0.0)
     ymin = max(crop_region['y_min'] * height, 0.0)
     rec_width = min(crop_region['x_max'], 0.99) * width - xmin
     rec_height = min(crop_region['y_max'], 0.99) * height - ymin
     rect = patches.Rectangle(
         (xmin,ymin),rec_width,rec_height,
         linewidth=1,edgecolor='b',facecolor='none')
     ax.add_patch(rect)
   fig.canvas.draw()
   image_from_plot = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
   # Determine the width and height of your canvas or image
   width, height = 855, 1200

   # Adjust reshape based on the size of the array
   #print("Actual array size:", image_from_plot.size)
   #print(fig.canvas.get_width_height()[::-1] + (3,))

   image_from_plot = image_from_plot.reshape((height, width, -1))
   plt.close(fig)
   if output_image_height is not None:
     output_image_width = int(output_image_height / height * width)
     image_from_plot = cv2.resize(
         image_from_plot, dsize=(output_image_width, output_image_height),
          interpolation=cv2.INTER_CUBIC)
   return image_from_plot

def get_ground_points(locs):
    left_ankle_pos = []
    #right_ankle_pos = (keypoints_with_scores[0][0][16][0], keypoints_with_scores[0][0][16][1])
    #print("check below ")
    #print(right_ankle_pos)

    #ankle_positions currently containes position of the right ankle. i can perform regression on this or perform regression on both
    # left and right ankle but n eed to decide how to get the ground instead of averaging the ankle in all positions
    # once ground is detected, should be able to determine contact time by seeing how long ankle pos is at certain range of ground point

    # adjust for camera angle

def to_gif(images, fps, loop):
   """Converts image sequence (4D numpy array) to gif."""
   imageio.mimsave('./animation.gif', images, fps=fps, loop=loop, duration = 5)
   return embed.embed_file('./animation.gif')

def progress(value, max=100):
   return HTML("""
       <progress
           value='{value}'
           max='{max}',
           style='width: 100%'
       >
           {value}
       </progress>
   """.format(value=value, max=max))

#model_name = "movenet_thunder"
model_name = "movenet_lightning"
if "tflite" in model_name:
   if "movenet_lightning" in model_name:
       #!wget -q -O model.tflite https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/3?lite-format=tflite
     input_size = 192
   elif "movenet_thunder" in model_name:
     #!wget -q -O model.tflite https://tfhub.dev/google/lite-model/movenet/singlepose/thunder/3?lite-format=tflite
     input_size = 256
   else:
     raise ValueError("Unsupported model name: %s" % model_name)
   interpreter = tf.lite.Interpreter(model_path="model.tflite")
   interpreter.allocate_tensors()

   def movenet(input_image):
     """Runs detection on an input image"""
     input_image = tf.cast(input_image, dtype=tf.float32)
     input_details = interpreter.get_input_details()
     output_details = interpreter.get_output_details()
     interpreter.set_tensor(input_details[0]['index'], input_image.numpy())
     interpreter.invoke()
     keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])
     return keypoints_with_scores
else:
   if "movenet_lightning" in model_name:
     #module = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/3")
     module = hub.load("https://www.kaggle.com/models/google/movenet/TensorFlow2/singlepose-lightning/4")
     input_size = 192
   elif "movenet_thunder" in model_name:
     module = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/3")
     input_size = 256
   else:
     raise ValueError("Unsupported model name: %s" % model_name)

   def movenet(input_image):
     """Runs detection on an input image"""
     model = module.signatures['serving_default']
     input_image = tf.cast(input_image, dtype=tf.int32)
     outputs = model(input_image)
     keypoint_with_scores = outputs['output_0'].numpy()
     return keypoint_with_scores
'''
image_path = 'img3.JPG'
image = tf.io.read_file(image_path)
image = tf.image.decode_jpeg(image)
input_image = tf.expand_dims(image, axis=0)
input_image = tf.image.resize_with_pad(input_image, input_size, input_size)
keypoint_with_scores = movenet(input_image)
print(keypoint_with_scores[0][0][13])
print(keypoint_with_scores[0][0][14])
kneeR = {'x': keypoint_with_scores[0][0][13][0],"y": keypoint_with_scores[0][0][13][1],"score": keypoint_with_scores[0][0][13][2]}
kneeL = {'x': keypoint_with_scores[0][0][14][0],"y": keypoint_with_scores[0][0][14][1],"score": keypoint_with_scores[0][0][14][2]}
midPelvis = {'x': keypoint_with_scores[0][0][12][0] - keypoint_with_scores[0][0][11][0],"y":  keypoint_with_scores[0][0][12][1] - keypoint_with_scores[0][0][11][1],"score":  keypoint_with_scores[0][0][12][2] - keypoint_with_scores[0][0][11][2]}

angle = (math.atan2(kneeL['y'] - midPelvis['y'], kneeL['x'] - midPelvis['x']) -
         math.atan2(kneeR['y'] - midPelvis['y'], kneeR['x'] - midPelvis['x'])) * (180 / math.pi)

if angle < 0:
    angle = angle + 360
    #pass  # Uncomment the line above if you want to add 360 when angle is negative

#if leftWrist['score'] > 0.3 and leftElbow['score'] > 0.3 and leftShoulder['score'] > 0.3:
    # print(angle)
    elbowAngle = angle
    print(elbowAngle)
#else:
    # print('Cannot see elbow')
#    pass  # Uncomment the line above if you want to handle the case where the elbow is not visible

display_image = tf.expand_dims(image, axis=0)
display_image = tf.cast(tf.image.resize_with_pad(display_image, 1280, 1280), dtype=tf.int32)
output_overlay = draw_prediction_on_image(np.squeeze(display_image.numpy(), axis=0), keypoint_with_scores)
plt.figure(figsize=(5, 5))
plt.imshow(output_overlay)
plt.axis('off')
plt.show()'''

#######################################################################
MIN_CROP_KEYPOINT_SCORE = 0.2

def init_crop_region(image_height, image_width):
  """Defines the default crop region"""
  if image_width > image_height:
    box_height = image_width / image_height
    box_width = 1.0
    y_min = (image_height / 2 - image_width / 2) / image_height
    x_min = 0.0
  else:
    box_height = 1.0
    box_width = image_height / image_width
    y_min = 0.0
    x_min = (image_width / 2 - image_height / 2) / image_width

  return {
    'y_min': y_min,
    'x_min': x_min,
    'y_max': y_min + box_height,
    'x_max': x_min + box_width,
    'height': box_height,
    'width': box_width
  }

def torso_visible(keypoints):
  """Checks whether there are enough torso keypoints"""
  return ((keypoints[0, 0, KEYPOINT_DICT['left_hip'], 2] >
           MIN_CROP_KEYPOINT_SCORE or
          keypoints[0, 0, KEYPOINT_DICT['right_hip'], 2] >
           MIN_CROP_KEYPOINT_SCORE) and
          (keypoints[0, 0, KEYPOINT_DICT['left_shoulder'], 2] >
           MIN_CROP_KEYPOINT_SCORE or
          keypoints[0, 0, KEYPOINT_DICT['right_shoulder'], 2] >
           MIN_CROP_KEYPOINT_SCORE))

def determine_torso_and_body_range(
    keypoints, target_keypoints, center_y, center_x):
  """Calculates the maximum distance from each keypoints to the center location"""
  torso_joints = ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']
  max_torso_yrange = 0.0
  max_torso_xrange = 0.0
  for joint in torso_joints:
    dist_y = abs(center_y - target_keypoints[joint][0])
    dist_x = abs(center_x - target_keypoints[joint][1])
    if dist_y > max_torso_yrange:
      max_torso_yrange = dist_y
    if dist_x > max_torso_xrange:
      max_torso_xrange = dist_x

  max_body_yrange = 0.0
  max_body_xrange = 0.0
  for joint in KEYPOINT_DICT.keys():
    if keypoints[0, 0, KEYPOINT_DICT[joint], 2] < MIN_CROP_KEYPOINT_SCORE:
      continue
    dist_y = abs(center_y - target_keypoints[joint][0]);
    dist_x = abs(center_x - target_keypoints[joint][1]);
    if dist_y > max_body_yrange:
      max_body_yrange = dist_y

    if dist_x > max_body_xrange:
      max_body_xrange = dist_x

  return [max_torso_yrange, max_torso_xrange, max_body_yrange, max_body_xrange]

def determine_crop_region(
      keypoints, image_height,
      image_width):
  """Determines the region to crop the image for the model to run inference on"""
  target_keypoints = {}
  for joint in KEYPOINT_DICT.keys():
    target_keypoints[joint] = [
      keypoints[0, 0, KEYPOINT_DICT[joint], 0] * image_height,
      keypoints[0, 0, KEYPOINT_DICT[joint], 1] * image_width
    ]

  if torso_visible(keypoints):
    center_y = (target_keypoints['left_hip'][0] +
                target_keypoints['right_hip'][0]) / 2;
    center_x = (target_keypoints['left_hip'][1] +
                target_keypoints['right_hip'][1]) / 2;

    (max_torso_yrange, max_torso_xrange,
      max_body_yrange, max_body_xrange) = determine_torso_and_body_range(
          keypoints, target_keypoints, center_y, center_x)

    crop_length_half = np.amax(
        [max_torso_xrange * 1.9, max_torso_yrange * 1.9,
          max_body_yrange * 1.2, max_body_xrange * 1.2])

    tmp = np.array(
        [center_x, image_width - center_x, center_y, image_height - center_y])
    crop_length_half = np.amin(
        [crop_length_half, np.amax(tmp)]);

    crop_corner = [center_y - crop_length_half, center_x - crop_length_half];

    if crop_length_half > max(image_width, image_height) / 2:
      return init_crop_region(image_height, image_width)
    else:
      crop_length = crop_length_half * 2;
      return {
        'y_min': crop_corner[0] / image_height,
        'x_min': crop_corner[1] / image_width,
        'y_max': (crop_corner[0] + crop_length) / image_height,
        'x_max': (crop_corner[1] + crop_length) / image_width,
        'height': (crop_corner[0] + crop_length) / image_height -
            crop_corner[0] / image_height,
        'width': (crop_corner[1] + crop_length) / image_width -
            crop_corner[1] / image_width
      }
  else:
    return init_crop_region(image_height, image_width)

def crop_and_resize(image, crop_region, crop_size):
  """Crops and resize the image to prepare for the model input."""
  boxes=[[crop_region['y_min'], crop_region['x_min'],
          crop_region['y_max'], crop_region['x_max']]]
  output_image = tf.image.crop_and_resize(
      image, box_indices=[0], boxes=boxes, crop_size=crop_size)
  return output_image

def feedback():
    print("Based on the produced kinogram, here are the following suggestions:")
    print("Toe Off:\n - Stance Foot = Perpendicular to ground\n "
          "- Noticeable knee bend in stance leg (excessive pushing = stunted swing leg recovery / exacerbating rear side mechanics)")

def run_inference(movenet, image, crop_region, crop_size):
  """Runs model inferece on the cropped region"""
  image_height, image_width, _ = image.shape
  input_image = crop_and_resize(
    tf.expand_dims(image, axis=0), crop_region, crop_size=crop_size)
  # Run model inference.
  keypoints_with_scores = movenet(input_image)

  (keypoint_locs, keypoint_edges,
    edge_colors) = _keypoints_and_edges_for_display(
        keypoints_with_scores, image_height, image_width)

  ankleR = (keypoints_with_scores[0][0][16][0], keypoints_with_scores[0][0][16][1])
  #print(ankleR)
  ankle_positions.append(ankleR)
  '''
  kneeR = {'x': keypoints_with_scores[0][0][13][0], "y": keypoints_with_scores[0][0][13][1],
           "score": keypoints_with_scores[0][0][13][2]}
  kneeL = {'x': keypoints_with_scores[0][0][14][0], "y": keypoints_with_scores[0][0][14][1],
           "score": keypoints_with_scores[0][0][14][2]}
  midPelvis = {'x': keypoints_with_scores[0][0][12][0] - keypoints_with_scores[0][0][11][0],
               "y": keypoints_with_scores[0][0][12][1] - keypoints_with_scores[0][0][11][1],
               "score": keypoints_with_scores[0][0][12][2] - keypoints_with_scores[0][0][11][2]}

  angle = (math.atan2(kneeL['y'] - midPelvis['y'], kneeL['x'] - midPelvis['x']) -
           math.atan2(kneeR['y'] - midPelvis['y'], kneeR['x'] - midPelvis['x'])) * (180 / math.pi)
  if angle < 0:
      #angle = angle + 360
      pass  # Uncomment the line above if you want to add 360 when angle is negative

      # if leftWrist['score'] > 0.3 and leftElbow['score'] > 0.3 and leftShoulder['score'] > 0.3:
      # print(angle)

  elbowAngle = angle
  angles.append(angle)
  #print(elbowAngle)
'''

  if (len(keypoint_locs)) > 16:
      # Convert points to numpy arrays for easier calculations
      points = np.array([keypoint_locs[13], keypoint_locs[14], (keypoint_locs[11] + keypoint_locs[12]) / 2])
      angle = compute_angle(points[0], points[1], points[2])
      angles.append(angle)

  # Update the coordinates.
  for idx in range(17):
    keypoints_with_scores[0, 0, idx, 0] = (
        crop_region['y_min'] * image_height +
        crop_region['height'] * image_height *
        keypoints_with_scores[0, 0, idx, 0]) / image_height
    keypoints_with_scores[0, 0, idx, 1] = (
        crop_region['x_min'] * image_width +
        crop_region['width'] * image_width *
        keypoints_with_scores[0, 0, idx, 1]) / image_width
  return keypoints_with_scores

#from moviepy.editor import VideoFileClip
#videoClip = VideoFileClip("IMG_3852.mov")
#videoClip = VideoFileClip("fly.mov")
#videoClip.write_gif("fly.gif")

#image_path = 'giffy2.gif'
image_path = 'fly.gif'
#image_path = 'test_giffy.gif'
#image_path = 'perf.gif'
image = tf.io.read_file(image_path)
image = tf.image.decode_gif(image)

# Load the input image.
num_frames, image_height, image_width, _ = image.shape
crop_region = init_crop_region(image_height, image_width)

ankleR_pos = []
ankleL_pos = []
output_images = []
max_knee_seperation = 0
min_y = 100000
min_x = 100000
max_y = 0
total_distance = 0
total_time = 0
prev_pos = None
ground_points = []
ground = 740
height = 0
knee_hip_alignment_support = 100
knee_hip_alignment_strike = 100
kinogram = [0,0,0,0,0]

ground_contacts = 0
testframes = []
prev_ank_L = 0
prev_ank_R = 0
down = 1
search_start = 0
search_end = 0

ground_ten_frames = [750, 760]
deepest_ten = 0

bar = tqdm(total=num_frames-1)
#bar = display(progress(0, num_frames-1), display_id=True)

# ask user for ground approx
#plt.imshow(image[0, :, :, :], cmap='gray')  # Assuming you want to plot the first channel in grayscale
#plt.axis('off')
#plt.show()


#ground = input("Ground: ")
#ground = float(ground)

for frame_idx in range(num_frames):
  keypoints_with_scores = run_inference(
      movenet, image[frame_idx, :, :, :], crop_region,
      crop_size=[input_size, input_size])

  (keypoint_locs, keypoint_edges,
   edge_colors) = _keypoints_and_edges_for_display(
      keypoints_with_scores, image_height, image_width)

  if (len(keypoint_locs)>16):
      #print(keypoint_locs)
      ankR = {'x': keypoint_locs[16][0], "y": keypoint_locs[16][1],
               "score": keypoints_with_scores[0][0][13][2]}
      ankL = {'x': keypoint_locs[15][0], "y": keypoint_locs[15][1],
               "score": keypoints_with_scores[0][0][15][2]}
      kneeR = {'x': keypoint_locs[14][0], "y": keypoint_locs[14][1],
               "score": keypoints_with_scores[0][0][13][2]}
      kneeL = {'x': keypoint_locs[13][0], "y": keypoint_locs[13][1],
               "score": keypoints_with_scores[0][0][14][2]}
      midPelvis = {'x': (keypoint_locs[12][0] + keypoint_locs[11][0])/2,
                   "y": (keypoint_locs[12][1] + keypoint_locs[11][1])/2,
                   "score": keypoints_with_scores[0][0][12][2] - keypoints_with_scores[0][0][11][2]}

    # max thigh seperation
    # works for toe off, could also do max distance between ankle and opposite knee
    # to get different feet/sides, just check which foot is in front of pelvis
      if ((kneeR['x'] - kneeL['x']) ** 2 + (kneeR['y'] - kneeL['y']) ** 2) ** 0.5 > max_knee_seperation:
          max_knee_seperation = ((kneeR['x'] - kneeL['x']) ** 2 + (kneeR['y'] - kneeL['y']) ** 2) ** 0.5
          #key_frames = frame_idx
          kinogram[0] = frame_idx
          #ground_ten_frames.append(ankR['y'])

    # max proj
      points = np.array([keypoint_locs[13], keypoint_locs[14], (keypoint_locs[11] + keypoint_locs[12]) / 2, keypoint_locs[11], keypoint_locs[12], keypoint_locs[15], keypoint_locs[16]])
      angle = compute_angle(points[0], points[1], points[2])
      kneeAngR = compute_angle(points[4], points[1], points[6])
      kneeAngL = compute_angle(points[3], points[0], points[5])
      '''print(((midPelvis['y'] - ground) ** 2) ** 0.5)
      print(ankL['y'])
      print(ankR['y'])
      print(angle)
      print(abs(kneeL['x'] - kneeR['x']))
      print(abs(ankL['x'] - ankR['x']))'''
      #if ((midPelvis['y'] - ground) ** 2) ** 0.5 > height and ankL['y'] + 5 < ground and ankR['y'] + 5 < ground and angle > 50 and abs(kneeL['x'] - kneeR['x']) > 10:
      if ((midPelvis['y'] - ground) ** 2) ** 0.5 > height and ankL['y'] < ground and ankR['y'] < ground and angle > 50 and abs(kneeL['x'] - kneeR['x']) > 10 and abs(ankL['x'] - ankR['x']) > 10:
          #if frame_idx > 6 and frame_idx < 26:
          #print(kneeAngR)
          #print(kneeAngL)
          if kneeL['x'] - kneeR['x'] > 0 and 100 < kneeAngL < 130 : #left leg is in front
              height = ((midPelvis['y'] - ground) ** 2) ** 0.5
              #key_frames = frame_idx
              kinogram[1] = frame_idx
          if kneeL['x'] - kneeR['x'] < 0 and 100 < kneeAngR < 130:  # right leg is in front
              height = ((midPelvis['y'] - ground) ** 2) ** 0.5
              #key_frames = frame_idx
              kinogram[1] = frame_idx


    # full support left
      # should also check foot is on ground
      if abs(kneeL['x'] - keypoint_locs[11][0]) < knee_hip_alignment_support:
        knee_hip_alignment_support = abs(kneeL['x'] - keypoint_locs[11][0])
        kinogram[4] = frame_idx
      #  key_frames = frame_idx

    # strike R
      if abs(kneeL['x'] - keypoint_locs[11][0]) < knee_hip_alignment_strike and ankL['y'] + 10 < ground:
         knee_hip_alignment_strike = abs(kneeL['x'] - keypoint_locs[11][0])
         kinogram[2] = frame_idx
      #   key_frames = frame_idx

    # touch down
      if ((ankR['y'] - ankL['y']) ** 2) ** 0.5 > max_y:
            max_y = ((ankR['y'] - ankL['y']) ** 2) ** 0.5
            kinogram[3] = frame_idx
      #      key_frames = frame_idx

      if prev_pos is not None:
          # Calculate distance between current and previous position
          distance = ((midPelvis['x'] - prev_pos['x'])**2 + (midPelvis['y'] - prev_pos['y'])**2)**0.5

          # Calculate time between frames
          time_elapsed = 1 / 8  # Assuming uniform frame rate

          # Accumulate distance and time
          total_distance += distance
          total_time += time_elapsed

          # Update previous figure position for next iteration
      prev_pos = midPelvis

    ## consider touchdowns when knees are together or when arms are in line with body
      '''
      if frame_idx % 15 == 0 and frame_idx > 1:
          #if ankL['y'] > ankR['y']:
            #ground_ten_frames.append([ankL['x'], ankL['y']])
          #ground_ten_frames.append([deepest_ten[0],deepest_ten[1]])
          ground_ten_frames.append(deepest_ten)
          #else:
            #  ground_ten_frames.append([ankR['x'], ankR['y']])
          deepest_ten = 0
      else:
          if (ankL['y'] > deepest_ten).any():
              deepest_ten = ankL['y']
          else:
              deepest_ten =  ankR['y']
        '''

      if abs(kneeR['x'] - kneeL['x']) < 10:
          switch += 1
          switches.append((frame_idx,switch))

      if prev_ank_L == 0:
          prev_ank_L = ankL['y']

      if prev_ank_R == 0:
          prev_ank_R = ankR['y']

    #find initial range of contacts by seeing when knees are together, then go backwards to find the initial ground contact
    # based on some threshold plus knee closeness
      #if (ankL['y'] >= 745 or ankR['y'] >= 745) and down == 1:
      #if ankL['y'] - prev_ank_L < 0 and abs(ankL['x'] - midPelvis['x']) < 15:
      if abs(kneeL['x'] - kneeR['x']) < 5:
      #if abs(ankL['y'] - prev_ank_L) < 1:
          ground_contacts += 1
          #testframes.append(frame_idx)
          ground_points.append(max(ankL['y'],ankR['y']))

          for i in range(search_start, frame_idx):
              keypoints_with_scores = run_inference(
                  movenet, image[i, :, :, :], crop_region,
                  crop_size=[input_size, input_size])

              (keypoint_locs, keypoint_edges,
               edge_colors) = _keypoints_and_edges_for_display(
                  keypoints_with_scores, image_height, image_width)

              if (len(keypoint_locs) > 16):
                  # print(keypoint_locs)
                  ankR2 = {'x': keypoint_locs[16][0], "y": keypoint_locs[16][1],
                          "score": keypoints_with_scores[0][0][13][2]}
                  ankL2 = {'x': keypoint_locs[15][0], "y": keypoint_locs[15][1],
                          "score": keypoints_with_scores[0][0][15][2]}
                  kneeR2 = {'x': keypoint_locs[14][0], "y": keypoint_locs[14][1],
                           "score": keypoints_with_scores[0][0][13][2]}
                  kneeL2 = {'x': keypoint_locs[13][0], "y": keypoint_locs[13][1],
                           "score": keypoints_with_scores[0][0][14][2]}
                  #print(abs(ankL2['y'] - max(ankL['y'],ankR['y'])))
                  #print(abs(ankR2['y'] - max(ankL['y'], ankR['y'])))
                  if (abs(ankL2['y'] - max(ankL['y'],ankR['y'])) < 10 or abs(ankR2['y'] - max(ankL['y'],ankR['y'])) < 10) and abs(kneeL2['x'] - kneeR2['x']) < 15:
                      testframes.append(i)

        #down = 0

      prev_ank_R = ankR['y']
      prev_ank_L = ankL['y']
      search_start = frame_idx

      #if  (ankL['y'] <= 735 and ankR['y'] <= 735) and down == 0:
      #    down = 1
      #print(ground_ten_frames)
      #ground_points.append(ankL)
      '''
      #ground_points = get_ground_points(keypoint_locs)
      #g = keypoint_locs[15][:, :, np.newaxis]  # Add the new axis along axis 2
      #ground_points.concatenate(keypoint_locs[15])
      new_array = np.empty(ground_points.size + len(keypoint_locs[15]))
      # Copy the original array into the new array
      new_array[:-len(keypoint_locs[15])] = ground_points
      # Add the new item to the end of the new array
      new_array[-len( keypoint_locs[15]):] =  keypoint_locs[15]
      ground_points = new_array'''

  #print(keypoints_with_scores)
  if len(ground_ten_frames) >= 2:
    output_images.append(draw_prediction_on_image(
      image[frame_idx, :, :, :].numpy().astype(np.int32),
      #np.zeros((image_height, image_width, 3), dtype=np.int32),
      keypoints_with_scores, ground_ten_frames, crop_region=None,
      close_figure=True, output_image_height=300))
    #if len(ground_ten_frames) >= 3:
    #    ground_ten_frames = ground_ten_frames[1:]  # Remove the first element
  else:
      output_images.append(draw_prediction_on_image(
          image[frame_idx, :, :, :].numpy().astype(np.int32),
          # np.zeros((image_height, image_width, 3), dtype=np.int32),
          keypoints_with_scores, None, crop_region=None,
          close_figure=True, output_image_height=300))


  #ground_points = get_ground_points(keypoints_with_scores)

  crop_region = determine_crop_region(
      keypoints_with_scores, image_height, image_width)
  #bar.update(progress(frame_idx, num_frames-1))
  bar.update(1)

#reshaped_array2 = ground_points.reshape((1, 1, 2))

# Concatenate arrays
#concatenated_array = np.concatenate((array1, reshaped_array2), axis=0)

'''output_images.append(draw_nonfigure_prediction(
      #image[frame_idx, :, :, :].numpy().astype(np.int32),
      np.zeros((image_height, image_width, 3), dtype=np.int32),
      keypoints_with_scores, reshaped_array2, crop_region=None,
      close_figure=True, output_image_height=300))'''
#need to run another loop below to add the ground points to the images jsut like above

#print(ground_ten_frames)
imp_frames = []
for i in testframes:
    contact = True
    ind = i
    prev_dis = 0
    while contact:
        keypoints_with_scores = run_inference(
            movenet, image[ind, :, :, :], crop_region,
            crop_size=[input_size, input_size])

        (keypoint_locs, keypoint_edges,
         edge_colors) = _keypoints_and_edges_for_display(
            keypoints_with_scores, image_height, image_width)

        if (len(keypoint_locs) > 16):
            # print(keypoint_locs)
            ankR = {'x': keypoint_locs[16][0], "y": keypoint_locs[16][1],
                     "score": keypoints_with_scores[0][0][13][2]}
            ankL = {'x': keypoint_locs[15][0], "y": keypoint_locs[15][1],
                     "score": keypoints_with_scores[0][0][15][2]}
            kneeR = {'x': keypoint_locs[14][0], "y": keypoint_locs[14][1],
                      "score": keypoints_with_scores[0][0][13][2]}
            kneeL = {'x': keypoint_locs[13][0], "y": keypoint_locs[13][1],
                      "score": keypoints_with_scores[0][0][14][2]}
            elbL = {'x': keypoint_locs[7][0], "y": keypoint_locs[7][1],
                     "score": keypoints_with_scores[0][0][7][2]}
            elbR = {'x': keypoint_locs[8][0], "y": keypoint_locs[8][1],
                     "score": keypoints_with_scores[0][0][8][2]}
            midPelvis = {'x': (keypoint_locs[12][0] + keypoint_locs[11][0]) / 2,
                         "y": (keypoint_locs[12][1] + keypoint_locs[11][1]) / 2,
                         "score": keypoints_with_scores[0][0][12][2] - keypoints_with_scores[0][0][11][2]}
            #print(((kneeL['x'] - ankR['x']) ** 2 + (kneeL['y'] - ankR['y']) ** 2) ** 0.5)
            #print(((kneeR['x'] - kneeL['x']) ** 2 + (kneeR['y'] - kneeL['y']) ** 2) ** 0.5)
            #print(prev_dis)
            #print(ankR['y'] - ankL['y'])
            #print(ind)
            #print(ankR['score'])
            #print(ankL['score'])
            #print(abs(elbR['x'] - midPelvis['x']))
            #print("")
            # ensure hand position is sufficiently far (this will ensure that no stoppages occur when knees are close
            if (ankR['y'] - ankL['y'] <= 0):
                #if (((kneeR['x'] - kneeL['x']) ** 2 + (kneeR['y'] - kneeL['y']) ** 2) ** 0.5 < prev_dis) and ankR['score'] > 0.5 and abs(elbR['x'] - midPelvis['x'])>25:
                if (((kneeR['x'] - kneeL['x']) ** 2 + (kneeR['y'] - kneeL['y']) ** 2) ** 0.5 < prev_dis + 2) and ankR['score'] > 0.5 and abs(elbR['x'] - midPelvis['x']) > 25:
                    contact = False
                else:
                    prev_dis = ((kneeR['x'] - kneeL['x']) ** 2 + (kneeR['y'] - kneeL['y']) ** 2) ** 0.5
                    imp_frames.append(ind)
                    ind += 1
                    if ind>=num_frames:
                        contact = False
            else:
                #if (((kneeL['x'] - ankR['x']) ** 2 + (kneeL['y'] - ankR['y']) ** 2) ** 0.5 < prev_dis) and ankR['score'] > 0.5  and abs(elbR['x'] - midPelvis['x'])>25:
                if (((kneeR['x'] - kneeL['x']) ** 2 + (kneeR['y'] - kneeL['y']) ** 2) ** 0.5 < prev_dis + 2) and ankR['score'] > 0.5 and abs(elbR['x'] - midPelvis['x']) > 25:
                    contact = False
                else:
                    prev_dis = ((kneeR['x'] - kneeL['x']) ** 2 + (kneeR['y'] - kneeL['y']) ** 2) ** 0.5
                    imp_frames.append(ind)
                    ind += 1
                    if ind>=num_frames:
                        contact = False
        else:
            ind += 1
            if ind >= num_frames or ind > i + 12:
                contact = False

print("Below is imp")
print(imp_frames)

# Prepare gif visualization.
output = np.stack(output_images, axis=0)
g = to_gif(output, fps=8, loop=True)
#############################
##################################
#testframes = [47,48,49,50,51,52,53,54]
print(testframes)
imp_frames = [22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]

num_frames = len(imp_frames)
fig, axs = plt.subplots(1, num_frames, figsize=(num_frames * 5, 5))

for i in range(num_frames):
    axs[i].imshow(output[imp_frames[i]])
    axs[i].axis('off')
    axs[i].set_title(f"Contact {i + 1}")

plt.tight_layout()
plt.show()

print(kinogram)
num_frames = len(kinogram)
fig, axs = plt.subplots(1, num_frames, figsize=(num_frames * 5, 5))
# Sample text (replace with your text)
#texts = ["Text below image 1", "Text below image 2", "Text below image 3", "Text below image 4", "Text below image 5"]
# Formatted text for all elements
# Formatted text for all elements
texts = [
    "Toe Off (Last Frame Where Rear Foot is in Contact)\n"
    "▪ Stance Foot = Perpendicular to ground\n"
    "▪ Noticeable knee bend in stance leg \n(excessive pushing = stunted swing leg recovery/exacerbating rear side mechanics)\n"
    "▪ 90-degree angle between thighs\n"
    "▪ Toe of swing leg directly vertical to kneecap\n"
    "▪ Front shin parallel to rear thigh\n"
    "▪ 90 dorsiflexion in swing foot\n"
    "▪ Forearms perpendicular to each other \n(ear arm relatively open, front arm relatively closed)\n"
    "▪ No Arching Back",

    "Maximum Vertical Projection (Both feet parallel to ground)\n"
    "▪ 90 degrees between thighs\n"
    "▪ >110 degrees knee angle in front leg\n"
    "▪ Dorsiflexion of front foot\n"
    "▪ Maximal peace in athlete (fluid, relaxed)\n"
    "▪ Neutral head\n"
    "▪ Straight line between rear leg knee and shoulders\n"
    "▪ Slight cross body arm swing – neutral rotation of spine",

    "Strike (Rear leg knee directly vertical to pelvis)\n"
    "▪ Femur Perpendicular to Ground\n"
    "▪ 20-40-degree gap between thighs\n"
    "▪ Front foot beginning to supinate\n"
    "▪ Extension of knee comes from strong reflexive action\n of posterior chain extending the thigh back towards the COM (Never cue clawing or pawing)\n"
    "▪ Faster athletes = Faster rotational velocities = Greater\n strength on hamstrings = greater reflexive contraction at stretch = more violent extension of front thigh",

    "Touch Down (Front foot first touches the ground)\n"
    "▪ Knees together – Faster athletes may have \nswing leg slightly in front of stance knee\n"
    "▪ If knee is behind – indication of slower rotational\n velocities = over pushing = exacerbation of rear side mechanics.\n"
    "▪ Shank of stance shin perpendicular to ground, heel directly underneath knee\n"
    "▪ Hands parallel to each other – close to intersection\n of glutes-ham (similar elbow angles)\n"
    "▪ Initial contact with outside of ball of foot\n"
    "▪ Allow elbow joint to open as hands pass hips \n(close behind and in front)",

    "Full Support (Stance Leg directly under pelvis – \nfront toe lined with front pelvis; rear toe lined with rear pelvis)\n"
    "▪ Swing foot tucked under glutes (foot fully dorsiflexed)\n"
    "▪ Swing leg in front of hips (thigh 45 degrees from perpendicular)\n"
    "▪ Stance amortization occurs more at ankles than hips or knee\n"
    "▪ Excessive amortization can manifest from lack of specific\n strength, a more horizontal application\n of force, or initial touch down too far in front of COM\n"
    "▪ Better athletes = less amortization\n"
    "▪ Plantar flex prior to touchdown"
]



for i in range(num_frames):
    image = output[kinogram[i]]
    display_image = tf.expand_dims(image, axis=0)
    display_image = tf.cast(tf.image.resize_with_pad(display_image, 1200, 855), dtype=tf.int32)
    output_overlay = draw_nonfigure_prediction(np.squeeze(display_image.numpy(), axis=0), np.empty((0, 0, 0, 0)))
    axs[i].imshow(output_overlay)
    axs[i].axis('off')
    if i == 0:
        axs[i].set_title(f"Toe Off")
    elif i == 1:
        axs[i].set_title(f"Max Vert Proj")
    elif i == 2:
        axs[i].set_title(f"Strike")
    elif i == 3:
        axs[i].set_title(f"Touch Down")
    else:
        axs[i].set_title(f"Full Support")

# Step 3: Add Text
#for i, text in enumerate(texts):
#    axs[i].text(0.5, -0.1, text, horizontalalignment='center', verticalalignment='center', transform=axs[i].transAxes)

# Add text underneath each plot
for i, text in enumerate(texts):
    #axs[i].text(0.5, -0.5, text, ha='center', va='center', wrap=True, fontsize=3)
    axs[i].text(0.5, -0.1, text, horizontalalignment='center', verticalalignment='center', transform=axs[i].transAxes, fontsize=5)


plt.tight_layout()
plt.show()

'''
image = output[key_frames]
display_image = tf.expand_dims(image, axis=0)
display_image = tf.cast(tf.image.resize_with_pad(display_image, 1200, 855), dtype=tf.int32)
#output_overlay = draw_prediction_on_image(np.squeeze(display_image.numpy(), axis=0), keypoints_with_scores)
#image_from_plot = image_from_plot.reshape((height, width, -1))
output_overlay = draw_nonfigure_prediction(np.squeeze(display_image.numpy(), axis=0), np.empty((0, 0, 0, 0)))
plt.figure(figsize=(5, 5))
plt.imshow(output_overlay)
plt.axis('off')
plt.show()
'''

print('switches: '+str(switch))
print('distance: '+str(total_distance))
print('time: '+str(total_time))
print('contacts: '+str(ground_contacts))
feedback()

#clip = VideoFileClip("animation.gif")
#clip.write_videofile("gifvid.mp4")

plt.plot(angles, marker='o', linestyle='-')
plt.title('Hip Angle Over Frames')
plt.xlabel('Frame Index')
plt.ylabel('Hip Angle (degrees)')
plt.grid(True)
plt.show()

# Detect position changes
position_changes = []
for i in range(1, len(ankle_positions)):
    position_change = np.linalg.norm(np.array(ankle_positions[i]) - np.array(ankle_positions[i-1]))
    if position_change > threshold:
        position_changes.append(i)
# Visualize the ankle positions and position changes
ankle_x, ankle_y = zip(*ankle_positions)
plt.plot(range(len(ankle_y)), ankle_y, marker='o', linestyle='-', label='Ankle Position')

plt.plot(range(len(switches)), switches, marker='o', linestyle='-', label='Switches Count')

# Check if there are position changes before trying to unpack
if position_changes:
    change_x, change_y = zip(*[ankle_positions[i] for i in position_changes])
    #plt.scatter(change_x, change_y, color='red', label='Position Change')
    plt.scatter(range(len(change_y)), change_y, color='red', label='Position Change')

plt.title('Ankle Position Over Time')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.legend()
plt.show()
