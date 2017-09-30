echo "This doesn't work well"
#rm -rf ./NITIN_Evaluation/
#mkdir NITIN_Evaluation
#python $OBJECTDETECTION/models/object_detection/eval.py --logtostderr --pipeline_config_path=./ssd_inception_v2_coco_udacity_traffic_NITIN.config --checkpoint_dir=./NITIN_Checkpoint/ --eval_dir=./NITIN_Evaluation/
rm -rf ./NITIN_Evaluation3/
mkdir NITIN_Evaluation3
#python $OBJECTDETECTION/models/object_detection/eval.py --logtostderr --pipeline_config_path=./ssd_inception_v2_coco_udacity_traffic3big.config --checkpoint_dir=./NITIN_Checkpoint3/ --eval_dir=./NITIN_Evaluation3/
python $OBJECTDETECTION/models/object_detection/eval.py --logtostderr --pipeline_config_path=./ssd_inception_v2_coco_udacity_traffic3_NITIN.config --checkpoint_dir=./NITIN_Checkpoint3/ --eval_dir=./NITIN_Evaluation3/
