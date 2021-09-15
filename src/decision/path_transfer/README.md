## path_transfer package

* path_transfer

### rosparam

* angle_diff_setting_in : 決定是否進入轉彎的角度差設定,此角度差為自車所在之線段與路線終點的線段角度差
* angle_diff_setting_out : 決定是否出彎的角度差設定,此角度差為自車所在之線段與路線終點的線段角度差
* angle_diff_setting_distoturn : 決定轉彎點的角度差,此角度差為兩段相鄰線段的角度差

* z_diff_setting_in : 決定是否進入爬坡的高度差,此高度差為自車所在之線段與路線終點的線段高度差
* z_diff_setting_out : 決定是否離開爬坡的高度差,此高度差為自車所在之線段與路線終點的線段高度差
* slope_setting_distouphill : 決定開始爬坡點的斜率,此斜率每個線段的斜率

* end_path_size_set : 決定當線段剩餘的點數數量,已無使用

### Input

* node.subscribe("rear_current_pose", 1, CurrentPoseCallback);
* node.subscribe("/planning/scenario_planning/trajectory", 1, transfer_callback);
* node.subscribe("/planning/scenario_planning/lane_driving/behavior_planning/path", 1, transfer_path_callback);

### Output

* node.advertise<nav_msgs::Path>("nav_path_astar_final",1);
* node.advertise<nav_msgs::Path>("nav_path_astar_base_30",1);
* node.advertise<msgs::CurrentTrajInfo>("current_trajectory_info",1);
* node.advertise<std_msgs::Empty>("nav_path_astar_final/heartbeat",1);
* node.advertise<std_msgs::Float64>("veh_overshoot_orig_dis",1);
* node.advertise<std_msgs::Float64>("/control/lateral_cumulative_offset",1);
* node.advertise<std_msgs::Bool>("/control/end_path_flag",1 

### Description

* nav_path_astar_final : 轉發planning module最終出來的path topic
* nav_path_astar_base_30 : 轉發planning module中主要車道的path topic
* current_trajectory_info : 最終path的屬性,包含轉彎以及爬坡資訊
* nav_path_astar_final/heartbeat : 發出最終路線是否還有在發送的heartbeat
* veh_overshoot_orig_dis : 計算車輛偏離主要車道多遠
* /control/lateral_cumulative_offset : 此為車輛在1秒內偏移量,主要為LKC情境的指標之一
* /control/end_path_flag : 計算是否快到當下路線的終點

### Supplymentary

* ```nav_path_astar_final與nav_path_astar_base_30```的主要差異
	* nav_path_astar_final : 經過最後obstacle avoidance的path,此為最終車輛控制要控制到的path
	* nav_path_astar_base_30 : 經過lane change後的path,此為主要車輛要行駛的車道path
