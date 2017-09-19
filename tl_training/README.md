
This directory contains scripts and data to train the model to recognize traffic lights.

The scripts are based on Tensorflow Object Detection API:

[Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/object_detection)

```
git clone https://github.com/tensorflow/models.git
# From tensorflow/models/
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
export OBJECTDETECTION= <path to where you checked it out>
```

The data directory includes training data and scripts to manipulate it.

```
TF_RECORDS - came from the udacity slack channel - someone else recorded the frames and labeled the traffic lights (with one class - traffic light)
extract.py - opens the TF_RECORDS files and dumps out jpeg images, and a csv file
tf_data_train.csv and  tf_data_val.csv - these are the CSV files that were dumped, but modified to have three classes (yellow/green/red)
reimport.py - takes the jpeg images and the csv files and creates new tf record files (output_train.record, output_val.record)
output.pbtxt - a protobuf file that maps the three classes to numbers
udacity_converter.py - takes the [udacity](http://bit.ly/udacity-annotations-autti) file from [annotations](https://github.com/udacity/self-driving-car/tree/master/annotations) and creates a tfrecord with just the traffic lights 
```

The train directory contains some configuration files and scripts to build the frozen pb file needed for inference.

```
dl.sh - download the ssd_inception_v2_coco_11_06_2017.tar.gz file that we use for transfer learning
train_tl.sh - this runs train.py to kick out a trained model checkpoint
eval_tl.sh - this will create an evaulation directory that can be used with tensorboard ( tensorboard --logdir=./NITIN_Evaluation3/ )
export_tl.sh - this creates a frozen model to be used in inference
ssd_inception_v2_coco_udacity_traffic3_NITIN.config   - this trains a 3 class classifier using the output_train.record/output_val.record
ssd_inception_v2_coco_udacity_traffic_NITIN.config - this trains a 1 class classifier using the original TF_RECORD files from slack
ssd_inception_v2_coco_udacity_traffic3big.config - this trains using the udacity data (which still needs to have a validation set created) 
   note: the udacity data doesn't look enough like the parking lot. I don't think it works well.
```

The infer directory contains a sample script to read and write a movie with the labels added. There are also two movies in here from the rosbag.

```
tl_infer.py - opens the two movies, and writes two new ones using the frozen graph
loop_traffic_lights.mp4
traffic_lights.mp4
```

