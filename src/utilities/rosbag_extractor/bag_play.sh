rosparam set use_sim_time true

rosbag play *.bag --clock --topics \
/LidarFrontLeft/Compressed \
/LidarFrontRight/Compressed \
/LidarFrontTop/Compressed \
/cam/back_top_120/jpg \
/cam/front_bottom_60/jpg \
/cam/front_top_close_120/jpg \
/cam/front_top_far_30/jpg \
/cam/left_back_60/jpg \
/cam/left_front_60/jpg \
/cam/right_back_60/jpg \
/cam/right_front_60/jpg \
/RadFront \
/tf \
/veh_info
