#rm  NITIN_Checkpoint3_FasterRCNN/*
#python $OBJECTDETECTION/models/object_detection/train.py --logtostderr --pipeline_config_path=./faster_rcnn_resnet101_coco_Udacity_Carla_NITIN.config --train_dir=./NITIN_Checkpoint3_FasterRCNN
rm  NITIN_CheckpointSimulator_FasterRCNN/*
python $OBJECTDETECTION/models/object_detection/train.py --logtostderr --pipeline_config_path=./faster_rcnn_resnet101_coco_Udacity_Simulator_NITIN.config --train_dir=./NITIN_CheckpointSimulator_FasterRCNN
