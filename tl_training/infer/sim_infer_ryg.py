import numpy as np
import os
import io
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import json
import math


from collections import defaultdict
from io import StringIO
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from PIL import Image

from object_detection.utils import label_map_util

from object_detection.utils import visualization_utils as vis_util

PATH_TO_CKPT = "../train/tl_inferencesimulator_faster_r-cnn/frozen_inference_graph.pb"
PATH_TO_LABELS = "../data/bosch_output.pbtxt"
NUM_CLASSES = 14
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


#label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
#categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
#category_index = label_map_util.create_category_index(categories)

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

"""
category_index = {1: {'id': 1, 'name': u'traffic_light_red'},
                  2: {'id': 2, 'name': u'traffic_light_yellow'},
                  3: {'id': 3, 'name': u'traffic_light_green'}}
"""

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

def changelabel(label):
  if (label=='traffic_light_red'):
     return 'red'
  if (label=='traffic_light_green'):
     return 'green'
  if (label=='traffic_light_yellow'):
     return 'yellow'
  return label

def predict(image,fname,folder):
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
      min_score_thresh=0.5
      vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          use_normalized_coordinates=True,
          min_score_thresh=min_score_thresh,
          line_thickness=4)

      data = {}
      data['filename']=fname
      data['folder']=folder
      data['creator']='synthetic'
      (width,height) =  image.size
      data['image_w_h']= image.size
      data['objects']=[]
      scores2=np.squeeze(scores)
      boxes2=np.squeeze(boxes)
      classes2=np.squeeze(classes).astype(np.int32)
  
      #print(category_index)
      for i in range(boxes2.shape[0]):
         if scores2 is None or scores2[i] > min_score_thresh:
                  box = tuple(boxes2[i].tolist())
                  #print(classes2[i])
                  if classes2[i] in category_index.keys():
                     #class_name = changelabel(category_index[classes2[i]]['name'])
                     class_name = category_index[classes2[i]]['name']
                  label = {}
                  label['label']=class_name
                  # force correct label
                  #label['label']=folder
                  #print(box)
                  x = int(round(box[1]*width))
                  y = int(round(box[0]*height))
                  x2 = int(round(box[3]*width))
                  y2 = int(round(box[2]*height))
                  label['x_y_w_h']= (x,y,(x2-x),(y2-y))
                  data['objects'].append(label)
      jsondata=json.dumps(data)
      #print(jsondata)


      return image_np,jsondata





def processVideo(input_video,output):
  clip1 = VideoFileClip(input_video)
  print("about to predict on video",input_video)
  out_clip = clip1.fl_image(predict)
  out_clip.write_videofile(output,audio=False)


def processDir(color):
  path = '../data/simulator_imgs/'+color
  outpath = '../data/simulator_imgs/'+color+'outfaster_rcnn'
  jsonoutpath = '../data/simulator_imgs/'+color+'outfaster_rcnn/annotations'
  if not os.path.exists(outpath):
    os.makedirs(outpath)
  if not os.path.exists(jsonoutpath):
    os.makedirs(jsonoutpath)
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
              image_np,jsondata = predict(image,fname,color)
              im = Image.fromarray(image_np)
              im.save(os.path.join(outpath, '{}'.format(fname)))
              #plt.figure(figsize=IMAGE_SIZE)
              #plt.imshow(image_np)
              #plt.savefig(os.path.join(outpath, '{}'.format(fname)))
              #plt.close()
              jsonname = fname.replace('jpg','json')
              jsonfilename=(os.path.join(jsonoutpath, '{}'.format(jsonname)))
              with open(jsonfilename, 'w') as outfile:
                  outfile.write(jsondata)
                  outfile.close()

if __name__ == '__main__':
   processDir('green')
   processDir('yellow')
   processDir('red')
