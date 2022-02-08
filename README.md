# Lidar and camera traffic tracking

## System
1.  Ubuntu 18.04
2.  ROS Melodic 


## Install project and dependent package:Yolov3 
1. cd catkin_ws/src
2. git clone --recursive https://github.com/leggedrobotics/darknet_ros.git
3. git clone https://github.com/imavgg/Tracking.git


## Run proejcts
1. Replace "Tracking/launch/darknet_ros.launch" with "darknet_ros/launch/darknet_ros.launch"
2. cd ../
3. catkin_make -DCMAKE_BUILD_TYPE=Release
4. roslaunch lidar_cluster final_tracking.launch.

## Feature 
Lidar cluster: Initialize, lidar RANSAC, 3D Bounding box.
Yolo: label, 2D bounding box
Tracking: Data Association

## Video
https://youtu.be/QvvLt-RX1DI

