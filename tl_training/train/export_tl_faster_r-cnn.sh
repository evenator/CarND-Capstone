#rm -rf tl_inference
#mkdir tl_inference
#python export_inference_graph.py \
#    --input_type image_tensor \
#    --pipeline_config_path ./ssd_inception_v2_coco_udacity_traffic_NITIN.config  \
#    --trained_checkpoint_prefix ./NITIN_Checkpoint/model.ckpt-500 \
#    --output_directory tl_inference

#rm -rf tl_inference3_faster_r-cnn
#mkdir tl_inference3_faster_r-cnn
#python $OBJECTDETECTION/models/object_detection/export_inference_graph.py \
#    --input_type image_tensor \
#    --pipeline_config_path ./faster_rcnn_resnet101_coco_Udacity_Carla_NITIN.config \
#    --trained_checkpoint_prefix ./NITIN_Checkpoint3_FasterRCNN/model.ckpt-500 \
#    --output_directory tl_inference3_faster_r-cnn


#
# Make the inference file for the simulator
#
rm -rf tl_inferencesimulator_faster_r-cnn
mkdir tl_inferencesimulator_faster_r-cnn
python $OBJECTDETECTION/models/object_detection/export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path ./faster_rcnn_resnet101_coco_Udacity_Simulator_NITIN.config \
    --trained_checkpoint_prefix ./NITIN_CheckpointSimulator_FasterRCNN/model.ckpt-500 \
    --output_directory tl_inferencesimulator_faster_r-cnn
