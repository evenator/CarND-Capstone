import tensorflow as tf
import pandas as pd
import os
import io

from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict

from PIL import Image

flags = tf.app.flags
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')

FLAGS = flags.FLAGS

# TO-DO replace this with label map
def class_text_to_int(row_label):
    if row_label == 'Red':
        return 1
    if row_label == 'RedLeft':
        return 1
    elif row_label == 'Yellow':
        return 2
    elif row_label == 'YellowLeft':
        return 2
    elif row_label == 'Green':
        return 3
    elif row_label == 'GreenLeft':
        return 3
    else:
        None

total = 0

def create_tf_example(group,path):
  global total
  print("create_tf_example")
  # TODO(user): Populate the following variables from your example.
  fname = os.path.join(path, '{}'.format(group.filename))
  statinfo = os.stat(fname)
  print(fname)
  print(statinfo.st_size)
  with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
      encoded_jpg = fid.read()
  print(len(encoded_jpg))
  total=total+len(encoded_jpg)
  encoded_jpg_io = io.BytesIO(encoded_jpg)
  image = Image.open(encoded_jpg_io)
  (width, height) = image.size
  #print("image.size")
  #print(image.size)
  #print(width,height)

  #print("widht,height %d %d" , (width,height))
  filename = group.filename.encode('utf8') # Filename of the image. Empty if image is not from file
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

  for index, row in group.object.iterrows():
        #print(row)
        #print('xmin')
        xm = int(row['xmin'])
        #print("xm:["+str(xm)+"]")
        #print("width:["+str(width)+"]")
        w = int(width)
        #print(str(xm/w))
        xmins.append(float(row['xmin']) / width)
        xmaxs.append(float(row['xmax']) / width)
        ymins.append(float(row['ymin']) / height)
        ymaxs.append(float(row['ymax']) / height)
        classes_text.append(row['attr'].encode('utf8'))
        classes.append(class_text_to_int(row['attr']))

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

def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def main(_):
  #writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
  writer = tf.python_io.TFRecordWriter("udacity_output.record")
  print("inside main")
  path = "../object-dataset/"
  csv= path+"labels.csv"
  examples = pd.read_csv(csv,delimiter=' ')
  print(examples)
  examples=examples.query('label == "trafficLight" and attr==attr' )
  print(examples)

  grouped = split(examples, 'filename')
  #print(grouped)
  for group in grouped:
        #print(group)
        tf_example = create_tf_example(group, path)
        writer.write(tf_example.SerializeToString())
  # TODO(user): Write code to read in your dataset to examples variable

  #for example in examples:
  #  tf_example = create_tf_example(example)
  #  writer.write(tf_example.SerializeToString())

  writer.close()

  print("total")
  print(total)
if __name__ == '__main__':
  tf.app.run()
