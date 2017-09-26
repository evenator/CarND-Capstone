import tensorflow as tf
import pandas as pd
import os
import io
import json

from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict

from PIL import Image

flags = tf.app.flags
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')

FLAGS = flags.FLAGS

# TO-DO replace this with label map
def class_text_to_int(row_label):
    if row_label == 'red':
        return 2
    if row_label == 'RedLeft':
        return 5
    elif row_label == 'yellow':
        return 7
    elif row_label == 'YellowLeft':
        return 7
    elif row_label == 'green':
        return 1
    elif row_label == 'GreenLeft':
        return 3
    else:
        None

total = 0

def create_tf_example(jsondata,path,rootDir):
  global total
  #print("create_tf_example")
  # TODO(user): Populate the following variables from your example.
  fname = os.path.join(rootDir, '{}'.format(jsondata['filename']))
  statinfo = os.stat(fname)
  #print(fname)
  #print(statinfo.st_size)
  with tf.gfile.GFile(os.path.join(rootDir, '{}'.format(jsondata['filename'])), 'rb') as fid:
      encoded_jpg = fid.read()
  #print(len(encoded_jpg))
  total=total+len(encoded_jpg)
  encoded_jpg_io = io.BytesIO(encoded_jpg)
  image = Image.open(encoded_jpg_io)
  (width, height) = image.size
  #print("image.size")
  #print(image.size)
  #print(width,height)

  #print("widht,height %d %d" , (width,height))
  filename = jsondata['filename'].encode('utf8') # Filename of the image. Empty if image is not from file
  encoded_image_data = encoded_jpg # Encoded image bytes
  image_format = b'jpeg' # b'jpeg' or b'png'

  xmins = [] # List of normalized left x coordinates in bounding box (1 per box)
  xmaxs = [] # List of normalized right x coordinates in bounding box
             # (1 per box)
  ymins = [] # List of normalized top y coordinates in bounding box (1 per box)
  ymaxs = [] # List of normalized bottom y coordinates in bounding box
             # (1 per box)
  classes_text = [] # List of string class name of bounding box (1 per box)
  classes = [] # List of integer class id of bounding box (1 per box)

  for row in jsondata['objects']:
        #print(row)
        xmin = row['x_y_w_h'][0]
        ymin = row['x_y_w_h'][1]
        xmax = xmin+row['x_y_w_h'][2]
        ymax = ymin+row['x_y_w_h'][3]

        xmins.append(float(xmin) / width)
        xmaxs.append(float(xmax) / width)
        ymins.append(float(ymin) / height)
        ymaxs.append(float(ymax) / height)
        classes_text.append(row['label'].encode('utf8'))
        classes.append(class_text_to_int(row['label']))

  #print(height)
  #print(width)
  #print(filename)
  #print(classes_text)
  #print(classes)
  #print(xmins)
  #print(xmaxs)
  tf_example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(filename),
      'image/source_id': dataset_util.bytes_feature(filename),
      'image/encoded': dataset_util.bytes_feature(encoded_image_data),
      'image/format': dataset_util.bytes_feature(image_format),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
  }))
  return tf_example



def batch(writer,writer_val,path):
  i = 0
  train = 0
  val = 0
  # Set the directory you want to start from
  rootDir = 'simulator_imgs/'+path+'/annotations'
  for dirName, subdirList, fileList in os.walk(rootDir):
    print('Found directory: %s' % dirName)
    for fname in fileList:
      if fname.endswith('json') and not fname.startswith('.'):
         fullname = os.path.join(dirName, '{}'.format(fname))
         with open(fullname) as data_file:    
           data = json.load(data_file)
           #print('\t%s' % fname) 
           #print('\t%s' % data['objects']) 
           #print('\t%s' % data['filename']) 
           #print('\t%s' % data) 
           #print(path,rootDir)
           tf_example = create_tf_example(data,path,'simulator_imgs/'+path)
           # write 20% of samples to validation set
           #if not i % 5:
           #writer_val.write(tf_example.SerializeToString())
           #val = val + 1
           #else:
           writer.write(tf_example.SerializeToString())
           train = train + 1
           #i = i +1 
  print(path)
  print("val")
  print(val)
  print("train")
  print(train)


def main(_):
  writer = tf.python_io.TFRecordWriter("simulator_train.record")
  writer_val = tf.python_io.TFRecordWriter("simulator_val.record")
  batch(writer,writer_val,'green')
  batch(writer,writer_val,'yellow')
  batch(writer,writer_val,'red')
  writer.close()
  writer_val.close()

if __name__ == '__main__':
  tf.app.run()
