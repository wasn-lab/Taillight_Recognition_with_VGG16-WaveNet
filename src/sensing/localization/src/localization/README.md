# localizer Packege (localizer)
LiDAR sensor with GPS and IMU localize in point cloud map

# Node
localizer

# Input
Topic               Message
LidFrontTop         msgs/PointCloud.msg
veh_info            msgs/ego_x
veh_info            msgs/ego_y
veh_info            msgs/ego_heading
veh_info            msgs/ego_speed


# Output
Topic               Message
ndt_pose           msgs/PointCloud.msg
current_pose       geometry_msgs::PoseStamped
current_pose_2vm   geometry_msgs::PoseStamped
localizer_pose     geometry_msgs::PoseStamped
current_points     sensor_msgs::PointCloud2
sbg_raw_pose       geometry_msgs::PoseStamped
sbg_vm_pose        geometry_msgs::PoseStamped
sbg_local_pose     geometry_msgs::PoseStamped
predict_pose       geometry_msgs::PoseStamped

