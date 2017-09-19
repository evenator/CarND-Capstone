import numpy as np
import os
import io
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import json

from collections import defaultdict
from io import StringIO
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from PIL import Image

from object_detection.utils import label_map_util

from object_detection.utils import visualization_utils as vis_util


PATH_TO_CKPT = "../train/tl_inferencesimulator/frozen_inference_graph.pb"
PATH_TO_LABELS = "../data/output.pbtxt"
NUM_CLASSES = 3
IMAGE_SIZE = (12, 8)


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


label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


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
          min_score_thresh=0.2,
          line_thickness=4)
      return image_np





def processVideo(input_video,output):
  clip1 = VideoFileClip(input_video)
  print("about to predict on video",input_video)
  out_clip = clip1.fl_image(predict)
  out_clip.write_videofile(output,audio=False)


if __name__ == '__main__':
  path = '../data/simulator_imgs/green'
  outpath = '../data/simulator_imgs/greenout'
  if not os.path.exists(outpath):
    os.makedirs(outpath)
  rootDir = path
  for dirName, subdirList, fileList in os.walk(rootDir):
    print('Found directory: %s' % dirName)
    for fname in fileList:
      if fname.endswith('jpg') and not fname.startswith('.'):
         fullname = os.path.join(dirName, '{}'.format(fname))
         print(fullname)
         with open(fullname) as data_file:    
           with tf.gfile.GFile(os.path.join(path, '{}'.format(fname)), 'rb') as fid:
              encoded_jpg = fid.read()
              encoded_jpg_io = io.BytesIO(encoded_jpg)
              image = Image.open(encoded_jpg_io)
              image_np = predict(image)
              plt.figure(figsize=IMAGE_SIZE)
              plt.imshow(image_np)
              plt.savefig(os.path.join(outpath, '{}'.format(fname)))
              plt.close()

