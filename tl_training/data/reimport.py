import tensorflow as tf
import pandas as pd
import os
import io
import hashlib


from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict

from PIL import Image

flags = tf.app.flags
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')

FLAGS = flags.FLAGS

# TO-DO replace this with label map
def class_text_to_int(row_label):
    if row_label == 'traffic_light_red':
        return 1
    elif row_label == 'traffic_light_yellow':
        return 2
    elif row_label == 'traffic_light_green':
        return 3
    else:
        None


def create_tf_example(group,path,prefix):
  print("create_tf_example")
  print(group)
  print("------")
  if group['class']=='remove':
     print("remove")
     return
  # TODO(user): Populate the following variables from your example.
  path = path + prefix +"_tl_data/"
  with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
      encoded_jpg = fid.read()
  encoded_jpg_io = io.BytesIO(encoded_jpg)
  image = Image.open(encoded_jpg_io)
  (width, height) = image.size
  print("image.size")
  print(image.size)
  print(width,height)

  key = hashlib.sha256(encoded_jpg).hexdigest()

  #print("widht,height %d %d" , (width,height))
  filename = group['filename'].encode('utf8')# Filename of the image. Empty if image is not from file
  #encoded_image_data = encoded_jpg_io.getbuffer() # Encoded image bytes
  encoded_image_data = encoded_jpg # Encoded image bytes
  image_format = None # b'jpeg' or b'png'
  image_format = b'jpeg'

  xmins = [] # List of normalized left x coordinates in bounding box (1 per box)
  xmaxs = [] # List of normalized right x coordinates in bounding box
             # (1 per box)
  ymins = [] # List of normalized top y coordinates in bounding box (1 per box)
  ymaxs = [] # List of normalized bottom y coordinates in bounding box
             # (1 per box)
  classes_text = [] # List of string class name of bounding box (1 per box)
  classes = [] # List of integer class id of bounding box (1 per box)

  row=group
  print(row)
  print('xmin')
  xm = int(row['xmin'])
  print("xm:["+str(xm)+"]")
  print("width:["+str(width)+"]")
  w = int(width)
  print(str(xm/w))
  xmins.append(float(row['xmin']) / width)
  xmaxs.append(float(row['xmax']) / width)
  ymins.append(float(row['ymin']) / height)
  ymaxs.append(float(row['ymax']) / height)
  classes_text.append('traffic light'.encode('utf8')) 
  #classes_text.append(row['class'].encode('utf8'))
  classes.append(class_text_to_int(row['class']))
  print("====")
  print(row['xmin'])
  print(xmins)
  print(width)
  print(float(row['xmin'])/width)
  print("====")

  tf_example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
      'image/filename': dataset_util.bytes_feature(filename),
      'image/source_id': dataset_util.bytes_feature(filename),
      'image/encoded': dataset_util.bytes_feature(encoded_jpg), 
      'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
  }))
  return tf_example

def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]

def processFile(prefix):
  writer = tf.python_io.TFRecordWriter("output_"+prefix+".record")
  path = "./"
  csv= path+"tf_data_"+prefix+".csv"
  examples = pd.read_csv(csv,delimiter=';',header=0)
  print(examples)
  for index, row in examples.iterrows():
        #print(group)
        tf_example = create_tf_example(row, path,prefix)
        if (tf_example ):
            writer.write(tf_example.SerializeToString())
  writer.close()

def main(_):
  processFile('train')
  processFile('val')



if __name__ == '__main__':
  tf.app.run()
