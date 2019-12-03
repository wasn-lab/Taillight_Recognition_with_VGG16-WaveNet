#!/bin/bash

shopt -s extglob

if [ "$1" = "MMSL" ]
then

echo "MMSL"

rm -rf ../../README.md

rm -rf ../display
rm -rf ../launch/backup
rm -rf ../launch/launch/b1.launch
rm -rf ../launch/launch/display.launch
rm -rf ../launch/launch/os_test.launch
rm -rf ../launch/launch/rosbag_extractor.launch
rm -rf ../lgsvl
rm -rf ../libs/Alignment
rm -rf ../libs/BoundingBox/PCA
rm -rf ../libs/BoundingBox/LShape
rm -rf ../libs/Clustering/DBSCAN/Common
rm -rf ../libs/Clustering/Euclidean
rm -rf ../libs/Matching
rm -rf ../libs/TrafficFlow
rm -rf ../libs/Transmission/json.hpp
rm -rf ../libs/Transmission/RosModuleA.hpp
rm -rf ../libs/Transmission/RosModuleB1.hpp
rm -rf ../libs/Preprocess/GroundFilter/RayGroundFilter.*
rm -rf ../libs/Utility/KeyboardMouseEvent.*
rm -rf ../libs/Utility/CompressFunction.*
rm -rf ../lidars_grabber/src/UI
rm -rf ../lidars_preprocessing/src/Transmission
rm -rf ../lidars_preprocessing/src/S1Cluster
rm -rf ../lidars_preprocessing/src/S2Track
rm -rf ../lidars_preprocessing/src/S3Classify
rm -rf ../lidars_preprocessing/src/UI
rm -rf ../lidars_preprocessing/src/debug_tool.*
rm -rf ../msgs
rm -rf ../pcd_publisher
rm -rf ../rosbag_extractor
rm -rf ../rosbag_extractor_better
rm -rf ../script
rm -rf ../squeezesegnet/lidar_squseg_detect
rm -rf ../squeezesegnet/pretrain_squseg
rm -rf ../validation
rm -rf ../voxelnet

rm -rf `find ../ -iname 'main_a*'`
rm -rf `find ../ -iname 'main_b*'`
rm -rf `find ../ -iname 'readme.txt'`

elif [ "$1" = "S3" ]
then
	
echo "S3"
	
rm -rf ../../README.md

rm -rf ../display
rm -rf ../launch/backup
rm -rf ../launch/launch/display.launch
rm -rf ../launch/launch/hino1.launch
rm -rf ../launch/launch/hino2.launch
rm -rf ../launch/launch/os_test.launch
rm -rf ../launch/launch/rosbag_extractor.launch
rm -rf ../launch/launch/validation.launch
rm -rf ../lgsvl
rm -rf ../libs/Alignment
rm -rf ../libs/BoundingBox/PCA
rm -rf ../libs/BoundingBox/LShape
rm -rf ../libs/Clustering/DBSCAN/Common
rm -rf ../libs/Clustering/Euclidean
rm -rf ../libs/TrafficFlow
rm -rf ../libs/Transmission/json.hpp
rm -rf ../libs/Transmission/RosModuleA.hpp
rm -rf ../lidars_preprocessing/src/Transmission
rm -rf ../lidars_preprocessing/src/S1Cluster
rm -rf ../lidars_preprocessing/src/S2Track
rm -rf ../lidars_preprocessing/src/S3Classify
rm -rf ../lidars_preprocessing/src/debug_tool.*
rm -rf ../msgs
rm -rf ../pcd_publisher
rm -rf ../rosbag_extractor
rm -rf ../rosbag_extractor_better
rm -rf ../script
rm -rf ../squeezesegnet/lidar_squseg_detect
rm -rf ../squeezesegnet/pretrain_squseg
rm -rf ../validation
rm -rf ../voxelnet

rm -rf `find ../ -iname 'main_a*'`
rm -rf `find ../ -iname 'readme.txt'`

fi
