# CarND-Capstone - Imagination Ltd

## Team:

```
Ed Venator: evenator@gmail.com
  slack: @evenator
Nitin Daga: nisn_daga@yahoo.com
  slack: @nisn
Lucas Nogueira: lukscasanova@gmail.com
  slack: @lukscasanova
Alan Steremberg:	alan.steremberg@gmail.com
  slack: @alansteremberg
```

The capstone project is designed to pull together a bunch of self drviing car concepts and implement them on top of ROS and deploy it to a real self driving car.  This project was super exciting to work on.

We split this project into parts:

* Installation of a working ROS Environment

* Waypoint Following

* Traffic Light Object Detection

* Stopping at the detected traffic lights safely

The software is tested on both the Udacity simulator, as well as a rosbag of images from Carla.  Two different models were trained for the traffic light detector, and the ROS launch files contain the configuration to use the correct one.

## Waypoint Following

### TODO: FILL THIS IN ED&LUCAS

## Traffic Light Object Detection

Before our first team meeting Nitin had started to read papers and try to implement SSD algorithms in tensorflow.  After our first team meeting Alan and Nitin decided to team up and tackle the problem together.  The decision was made to use the tensorflow object detection API to build and implement the models.  This cut down the implementation time to just a few minutes to create the configuration files.  Most of the work was spent trying different techniques, and writing scripts to import and export the data to the TFRecord format that tensorflow uses.  This code is in the tl_training directory.  

We had a few first failed attempts before we decided to follow the medium post by Anthony Sarkis.  Before following this, we attempted to train the network with the Udacity data set. Unfortunatley a lot of the traffic lights in this data set are really far away and small. When we trained on this, the reflection of the traffic lights on the hood of Carla would be detected  - but the main light would not.  

Ed dumped a bunch of training data from the simulator, and separated it by color into three folders.  Alan hand edited 90 or so images using Mac LabelRect.  After building a tool to convert the images, and json from LabelRect to TFRectord. We trained a model using these 90 images, and then ran the rest of the 600 or so back through the inference. We forced the labels to the correct names - and outputed JSON that matched LabelRect.  We then used LabelRect to hand correct another batch of images. We repeated this until the model got pretty good.

When we deployed the traffic light detection onto the full stack, it didn't do terribly well.  We noticed Anthony Sarkis' post, and decided to download the Bosch data, and try his approach.  It took around 3 tried to get it working. We had problems with tensorflow complaining about NaN's, and some of the predictions didn't work that well.  After we got through that, we trained a reasonably good Bosch based model (transfer learning based on faster_rcnn_resnet101_coco_11_06_2017). We then used our Bosch trained model as the starting point for two more models, one with the simulator images, and one with images from Carla.     

We had to be careful to make sure we had a balanced data set as well. Otherwise the color detection skewed towards the color that had the largest number of samples.


In addition to some offline inference scripts, we added a new ROS variable called image_color_annotated that publishes the image_color with the bounding boxes, label, and score on them. This makes it so we can use the built in rosrun image_view to watch in realtime what is happening on the simulator or with Carla's rosbag.


We are super happy with the output from the models:
![green](imgs/screenshots/tl1.png)
It doesn't perform as well when the lights aren't clear, and are small:
![red](imgs/screenshots/tl2.png)
![red](imgs/screenshots/tl3.png)
![green](imgs/screenshots/tl4.png)
![red](imgs/screenshots/tl5.png)


 


