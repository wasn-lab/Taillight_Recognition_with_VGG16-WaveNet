cd ~/itriadv
catkin_make -DCATKIN_WHITELIST_PACKAGES="car_model;msgs;autoware_perception_msgs;camera_utils;camera_grabber;image_compressor;object_costmap_generator;drivenet_lib;drivenet;libs;alignment;fail_safe;hungarian;itri_tracking_2d;msg_replay;msg_recorder" -DCAR_MODEL=B1_V3
