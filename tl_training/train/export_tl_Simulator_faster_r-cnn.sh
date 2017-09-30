#
# Make the inference file for the simulator
#
rm -rf tl_inferencesimulator_faster_r-cnn
mkdir tl_inferencesimulator_faster_r-cnn
python $OBJECTDETECTION/models/object_detection/export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path ./faster_rcnn_resnet101_coco_Udacity_Simulator.config \
    --trained_checkpoint_prefix ./CheckpointSimulator_FasterRCNN/model.ckpt-3000 \
    --output_directory tl_inferencesimulator_faster_r-cnn
