#rm  NITIN_Checkpoint/*
#python ./train.py --logtostderr --pipeline_config_path=./ssd_inception_v2_coco_udacity_traffic_NITIN.config --train_dir=./NITIN_Checkpoint
rm  NITIN_Checkpoint3/*
#python ./train.py --logtostderr --pipeline_config_path=./ssd_inception_v2_coco_udacity_traffic3big.config --train_dir=./NITIN_Checkpoint3
python $OBJECTDETECTION/models/object_detection/train.py --logtostderr --pipeline_config_path=./ssd_inception_v2_coco_udacity_traffic3_NITIN.config --train_dir=./NITIN_Checkpoint3
