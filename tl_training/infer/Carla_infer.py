from moviepy.editor import VideoFileClip

import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from PIL import Image

from object_detection.utils import label_map_util

from object_detection.utils import visualization_utils as vis_util


PATH_TO_CKPT = "../train/tl_inferenceCarla_faster_r-cnn/frozen_inference_graph.pb"
PATH_TO_LABELS = "../data/output.pbtxt"

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


category_index = {1: {'id': 1, 'name': u'traffic_light_green'},
                  2: {'id': 2, 'name': u'traffic_light_red'},
                  3: {'id': 3, 'name': u'traffic_light_green'},
                  4: {'id': 4, 'name': u'traffic_light_green'},
                  5: {'id': 5, 'name': u'traffic_light_red'},
                  6: {'id': 6, 'name': u'traffic_light_red'},
                  7: {'id': 7, 'name': u'traffic_light_yellow'},
                  8: {'id': 8, 'name': u'traffic_light_yellow'},
                  9: {'id': 9, 'name': u'traffic_light_red'},
                  10: {'id': 10, 'name': u'traffic_light_green'},
                  11: {'id': 11, 'name': u'traffic_light_green'},
                  12: {'id': 12, 'name': u'traffic_light_green'},
                  13: {'id': 13, 'name': u'traffic_light_red'},
                  14: {'id': 14, 'name': u'traffic_light_red'}}



sess = None

with detection_graph.as_default():
    sess=tf.Session(graph=detection_graph)

    # Definite input and output Tensors for detection_graph
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')



def predict(image):
      #image = Image.open(image_path)
      # the array based representation of the image will be used later in order to prepare the
      # result image with boxes and labels on it.
      #image_np = load_image_into_numpy_array(image)
      image_np = np.array(image)  
      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      image_np_expanded = np.expand_dims(image_np, axis=0)
      # Actual detection.
      (boxes, scores, classes, num) = sess.run(
          [detection_boxes, detection_scores, detection_classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})
      # Visualization of the results of a detection.
      #print((scores))
      vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          use_normalized_coordinates=True,
          min_score_thresh=0.6,
          line_thickness=4)
      return image_np





def processVideo(input_video,output):
  clip1 = VideoFileClip(input_video)
  print("about to predict on video",input_video)
  out_clip = clip1.fl_image(predict)
  out_clip.write_videofile(output,audio=False)


if __name__ == '__main__':
    processVideo('loop_traffic_lights.mp4','./loop_traffic_lights_out.mp4')
    processVideo('traffic_lights.mp4','./traffic_lights_out.mp4')

