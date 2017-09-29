#
# Make the inference file for Carla
#
rm -rf tl_inferenceCarla_faster_r-cnn
mkdir tl_inferenceCarla_faster_r-cnn
python $OBJECTDETECTION/models/object_detection/export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path ./faster_rcnn_resnet101_coco_Udacity_Carla.config \
    --trained_checkpoint_prefix ./Carla_FasterRCNN/model.ckpt-1000 \
    --output_directory tl_inferenceCarla_faster_r-cnn
