from PIL import Image
import numpy as np
import skimage.io as io
import tensorflow as tf
import cv2
import numpy as np
import os
import io
#import Image
from array import array
from PIL import Image
import csv

prefix = 'val'
prefix = 'train'

count = 0
csvfile = open('tf_data_'+prefix+'.csv','w')
writer = csv.writer(csvfile, delimiter=';')

#filename;class;xmin;ymin;xmax;ymax
line = []
line.append("filename")
line.append("class")
line.append("xmin")
line.append("ymin")
line.append("xmax")
line.append("ymax")
writer.writerow(line)


for string_record in tf.python_io.tf_record_iterator("TF_RECORDS/tl_"+prefix+".record"):
    #result = tf.train.Example.FromString(string_record)
    example = tf.train.Example()
    example.ParseFromString(string_record)
    
    height = (example.features.feature['image/height']
                                 .int64_list
                                 .value[0])
    print(height)
    
    width = (example.features.feature['image/width']
                                .int64_list
                                .value[0])
    print(width)

    label = (example.features.feature['image/object/class/label']
                                .int64_list
                                .value[0])
    print("label: ",label) 


    filename = "tl_" + str(count) + ".jpeg"
    count += 1

    img_string = (example.features.feature['image/encoded']
                                  .bytes_list
                                  .value[0])

    image = Image.open(io.BytesIO(img_string))
    (width2, height2) = image.size
    print(width2,height2,width,height)

    image.save('./val_tl_data/' + filename)

    xmin = (example.features.feature['image/object/bbox/xmin']
                                .float_list
                                .value[0])

    xmax = (example.features.feature['image/object/bbox/xmax']
                                .float_list
                                .value[0])
    
    ymin = (example.features.feature['image/object/bbox/ymin']
                                .float_list
                                .value[0])

    ymax = (example.features.feature['image/object/bbox/ymax']
                                .float_list
                                .value[0])
    line = []
    #csvstr = filename + ";" + str(xmin)
    line.append(filename)
    line.append("traffic_light")
    line.append(int(xmin*width))
    line.append(int(ymin*height))
    line.append(int(xmax*width))
    line.append(int(ymax*height))
    writer.writerow(line)
    #print(xmin,xmax,ymin,ymax)


print (count)
