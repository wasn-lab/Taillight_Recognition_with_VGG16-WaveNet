geofence_map_pp package
(1)geofence_map_pp

### Input
n.subscribe("localization_to_veh", 1, callbackLocalizationToVeh);
n.subscribe("veh_info", 1, callbackVehInfo);
n.subscribe("nav_path_astar_final", 1, callbackAStar);
n.subscribe("/PedCross/Alert", 1, callbackPP_PedCross);
(suspended)n.subscribe("current_pose", 1, callbackCurrentPose);
(suspended)n.subscribe("dynamic_path_para", 1, callbackPoly); 

### Output
n.advertise<visualization_msgs::Marker>("PP_geofence_line", 1);
n.advertise<sensor_msgs::PointCloud2>("pp_point_cloud", 1);

### Description
(1) 計算PP暴風圈是否有在路徑內
(2) 計算PedCross是否在路徑內
(3) 透過CAN傳送結果給dsapce
(4) 產生geofence marker給rviz
