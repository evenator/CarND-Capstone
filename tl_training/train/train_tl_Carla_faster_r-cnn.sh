rm  Checkpoint3_FasterRCNN/*
python $OBJECTDETECTION/models/object_detection/train.py --logtostderr --pipeline_config_path=./faster_rcnn_resnet101_coco_Udacity_Carla.config --train_dir=./Checkpoint3_FasterRCNN
