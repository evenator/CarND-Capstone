#rm  NITIN_Checkpoint3/*
#python $OBJECTDETECTION/models/object_detection/train.py --logtostderr --pipeline_config_path=./ssd_inception_v2_coco_udacity_traffic3_NITIN.config --train_dir=./NITIN_Checkpoint3
rm  NITIN_CheckpointSimulator/*
python $OBJECTDETECTION/models/object_detection/train.py --logtostderr --pipeline_config_path=./ssd_inception_v2_coco_udacity_simulator3_NITIN.config --train_dir=./NITIN_CheckpointSimulator
