"""Includes a class for classifying traffic lights."""
from collections import defaultdict
import cv2
import io
from io import StringIO
import numpy as np
import os
from PIL import Image
import rospy
from styx_msgs.msg import TrafficLight
import sys
import tensorflow as tf

NUM_CLASSES = 3

# For translating from TLClassifier light codes to TrafficLight light codes
LIGHT_CODES = [TrafficLight.RED, TrafficLight.YELLOW, TrafficLight.GREEN]


class TLClassifier(object):
    """
    A class to classify traffic lights.

    Uses a TensorFlow classifier to determine the state of a traffic light.
    """

    def __init__(self, path_to_ckpt, min_score_thresh=0.4):
        """
        Constructor.

        path_to_ckpt - Full path to the protobuff definition of the graph.
        min_score_thresh - Minimum ratio of boxes of one detection in order to accept it.
        """
        self.min_score_thresh = min_score_thresh
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(path_to_ckpt, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        with self.detection_graph.as_default():
            self.sess = tf.Session(graph=self.detection_graph)
            # Definite input and output Tensors for detection_graph
            self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
            self.output_tensors = [detection_boxes,
                                   detection_scores,
                                   detection_classes,
                                   num_detections]

    def get_classification(self, image):
        """
        Determine the color of the traffic light in the image.

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        category_index = {1: {'id': 1, 'name': u'traffic_light_red'},
                          2: {'id': 2, 'name': u'traffic_light_yellow'},
                          3: {'id': 3, 'name': u'traffic_light_green'}}

        # we trained the network with RGB images
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_np = np.array(image)
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        # Actual detection.
        feed_dict = {self.image_tensor: image_np_expanded}
        boxes, scores, classes, num = self.sess.run(self.output_tensors, feed_dict=feed_dict)

        width, height, channels = image.shape

        scores2 = np.squeeze(scores)
        boxes2 = np.squeeze(boxes)
        classes2 = np.squeeze(classes).astype(np.int32)

        green_count = 0
        red_count = 0
        yellow_count = 0
        class_name = None

        for i in range(boxes2.shape[0]):
            if scores2 is None or scores2[i] > self.min_score_thresh:
                box = tuple(boxes2[i].tolist())
                if classes2[i] in category_index.keys():
                    class_name = category_index[classes2[i]]['name']

                x = int(round(box[1]*width))
                y = int(round(box[0]*height))
                x2 = int(round(box[3]*width))
                y2 = int(round(box[2]*height))
                rospy.loginfo("traffic light: " + str((class_name, x, y, x2, y2)))

                if class_name == 'traffic_light_green':
                    green_count += 1
                elif class_name == 'traffic_light_red':
                    red_count += 1
                elif class_name == 'traffic_light_yellow':
                    yellow_count += 1

        tl = [red_count, yellow_count, green_count]
        max_tl = np.argmax(tl)
        rospy.loginfo("get_classification answer: %d", max_tl)
        return LIGHT_CODES[max_tl]


if __name__ == '__main__':
    tl = TLClassifier(sys.argv[1])

    # TODO: Nitin: Make sure that we pass the image in the right (RGB) format.
    filenames = ['red.jpg', 'yellow.jpg', 'green.jpg']
    for fname in filenames:
        with open(fname) as data_file:
            with tf.gfile.GFile(fname, 'rb') as fid:
                encoded_jpg = fid.read()
                encoded_jpg_io = io.BytesIO(encoded_jpg)
                image = Image.open(encoded_jpg_io)
                light = tl.get_classification(image)
                print("Light is: ", light)
