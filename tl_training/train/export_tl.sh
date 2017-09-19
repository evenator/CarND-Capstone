#rm -rf tl_inference
#mkdir tl_inference
#python export_inference_graph.py \
#    --input_type image_tensor \
#    --pipeline_config_path ./ssd_inception_v2_coco_udacity_traffic_NITIN.config  \
#    --trained_checkpoint_prefix ./NITIN_Checkpoint/model.ckpt-500 \
#    --output_directory tl_inference

rm -rf tl_inference3
mkdir tl_inference3
    #--pipeline_config_path ./ssd_inception_v2_coco_udacity_traffic3big.config   \
python $OBJECTDETECTION/models/object_detection/export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path ./ssd_inception_v2_coco_udacity_traffic3_NITIN.config \
    --trained_checkpoint_prefix ./NITIN_Checkpoint3/model.ckpt-500 \
    --output_directory tl_inference3
