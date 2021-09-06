## planning_initial package

* planning_initial

### rosparam

* use_virtual_objects : 是否使用虛擬障礙物,此為當初為了自己模擬物件在不同位置時測試planning用,default關閉
* location_name : 根據區域讀取不同檔案

### Input

* node.subscribe("current_pose", 1, CurrentPoseCallback);
* node.subscribe("input/objects", 1, objectsCallback);
* node.subscribe("input/lidar_no_ground", 1, LidnogroundpointCallback); // /LidarAll/NonGround2
* node.subscribe("veh_info",1,currentVelocityCallback);
* node.subscribe("imu_data_rad",1,imudataCallback);
* node.subscribe("/traffic", 1, trafficCallback);
* node.subscribe("/Flag_Info02", 1, trafficDspaceCallback); // 已無使用
* node.subscribe("/BusStop/Info", 1, busstopinfoCallback);
* node.subscribe("occupancy_grid", 1, occgridCallback); //此為freespace
* node.subscribe("occupancy_grid_wayarea", 1, occgridwayareaCallback); //此為變寬車道的grid map

### Output

* node.advertise<geometry_msgs::PoseStamped>("rear_current_pose", 1, true);
* node.advertise<autoware_perception_msgs::DynamicObjectArray>("output/objects", 1, true);
* node.advertise<sensor_msgs::PointCloud2>("output/lidar_no_ground", 1, true);
* node.advertise<geometry_msgs::TwistStamped>("/localization/twist", 1, true);
* node.advertise<autoware_perception_msgs::TrafficLightStateArray>("output/traffic_light", 1, true);
* node.advertise<msgs::BusStopArray>("BusStop/Reserve", 1, true);
* node.advertise<nav_msgs::OccupancyGrid>("occupancy_grid_maporigin", 1, true);
* node.advertise<nav_msgs::OccupancyGrid>("occupancy_grid_wayarea_maporigin", 1, true); 

### Description

* rear_current_pose : 計算出後軸pose
* output/objects : 將objects這個topic轉成planning所要吃的topic name
* output/lidar_no_ground : 將ring_edge_point_cloud這個topic轉成planning所要吃的topic name
* /localization/twist : 將veh_info中的ego_speed轉成planning所要吃的topic name
* output/traffic_light : 將traffic這個topic配合Dmap_traffic_light_info.txt轉成planning所要吃的topic name
* BusStop/Reserve : 將BusStop/Info這個topic配合HDmap_bus_stop_info.txt轉成planning所要吃的topic name
* occupancy_grid_maporigin : 將occupancy_grid的frame_id平移到map frame上
* occupancy_grid_wayarea_maporigin : 將occupancy_grid_wayarea的frame_id平移到map frame上

### Supplymentary

* ```ring_edge_point_cloud與occupancy_grid```的差異
 * 兩者皆為edge_detection發出的freespace topic,不同點為
  1. ring_edge_point_cloud是以ring方式表示環周碰撞最近點,一圈點數較少,適合用在快速判斷碰撞點的node,如geofence與obstacle_stop_planning
  2. occupancy_grid是以grid map表示,以格子表示有無lidar point,適合用在路徑規劃上,如obstacle_avoidance_planning

