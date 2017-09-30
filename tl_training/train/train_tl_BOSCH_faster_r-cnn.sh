rm  CheckpointBOSCH_FasterRCNN/*
python $OBJECTDETECTION/models/object_detection/train.py --logtostderr --pipeline_config_path=./faster_rcnn_resnet101_coco_Udacity_BOSCH.config --train_dir=./CheckpointBOSCH_FasterRCNN
