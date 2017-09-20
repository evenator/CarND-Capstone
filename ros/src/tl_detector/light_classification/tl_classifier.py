"""
TODO:
1. Uncomment, "from styx_msgs.msg import TrafficLight"
2. Take care of CV image's BGR --to--> RGB (?) format conversion.
3. Return the "index" corresponding to the color of TL.
4. Switch between sim and carla models based on some config setting (?). 
"""

from styx_msgs.msg import TrafficLight

import numpy as np
import os
import io
import sys
import tensorflow as tf
import rospy
from collections import defaultdict
from io import StringIO
from PIL import Image
import cv2


PATH_TO_CKPT = "../../../../tl_training/train/tl_inferencesimulator/frozen_inference_graph.pb"
#PATH_TO_CKPT = "./frozen_inference_graph.pb"
NUM_CLASSES = 3


class TLClassifier(object):
    def __init__(self):
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
          self.od_graph_def = tf.GraphDef()
          with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            self.serialized_graph = fid.read()
            self.od_graph_def.ParseFromString(self.serialized_graph)
            tf.import_graph_def(self.od_graph_def, name='')
        pass

    def changelabel(self,label):
        if (label=='traffic_light_red'):
            return 'red'
        if (label=='traffic_light_green'):
            return 'green'
        if (label=='traffic_light_yellow'):
            return 'yellow'
        return label


    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        category_index = {1: {'id': 1, 'name': u'traffic_light_red'},
                  2: {'id': 2, 'name': u'traffic_light_yellow'},
                  3: {'id': 3, 'name': u'traffic_light_green'}}


        sess = None
        light = ['red','yellow','green']

        with self.detection_graph.as_default():
            sess=tf.Session(graph=self.detection_graph)
            # Definite input and output Tensors for detection_graph
            image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

        # we trained the network with RGB images
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        image_np = np.array(image)
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        # Actual detection.
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})

        # this threshold may want to be 0.2 ??
        min_score_thresh=0.4
        data = {}
        (width,height) =  image.size
        #data['image_w_h']= image.size
        data['objects']=[]

        scores2=np.squeeze(scores)
        boxes2=np.squeeze(boxes)
        classes2=np.squeeze(classes).astype(np.int32)

        green_count = 0
        red_count = 0
        yellow_count = 0
        class_name = None

        for i in range(boxes2.shape[0]):
            if scores2 is None or scores2[i] > min_score_thresh:
                box = tuple(boxes2[i].tolist())
                if classes2[i] in category_index.keys():
                    class_name = self.changelabel(category_index[classes2[i]]['name'])
                label = {}
                label['label']=class_name

                x = int(round(box[1]*width))
                y = int(round(box[0]*height))
                x2 = int(round(box[3]*width))
                y2 = int(round(box[2]*height))
                label['x_y_w_h']= (x,y,(x2-x),(y2-y))
                data['objects'].append(label)

                if label['label'] == 'green':
                    green_count += 1
                elif label['label'] == 'red':
                    red_count += 1
                elif label['label'] == 'yellow':
                    yellow_count += 1

        tl = [red_count, yellow_count, green_count]
        lightcode = [TrafficLight.RED,TrafficLight.YELLOW,TrafficLight.GREEN]
        return lightcode[tl.index(max(tl))]


tl = TLClassifier()

if __name__ == '__main__':

  # Testing. TODO: Nitin: Make sure that we pass the image in the right (RGB) format.
  filenames = ['red.jpg','yellow.jpg','green.jpg']
  for fname in filenames:
    with open(fname) as data_file:
      with tf.gfile.GFile(fname, 'rb') as fid:
        encoded_jpg = fid.read()
        encoded_jpg_io = io.BytesIO(encoded_jpg)
        image = Image.open(encoded_jpg_io)
        light = tl.get_classification(image)
        print("Light is: ",light)

