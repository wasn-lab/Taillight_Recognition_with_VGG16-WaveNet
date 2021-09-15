## decision_maker package

* decision_maker

### Input

* node.subscribe("Flag_Info01", 1, avoidstatesubCallback); //ACC判斷要繞越之flag
* node.subscribe("Geofence_original", 1, obsdisbaseCallback);
* node.subscribe("veh_overshoot_orig_dis", 1, overshootorigdisCallback);
* node.subscribe("lane_event", 1, laneeventCallback); //已無使用
* node.subscribe("/planning/scenario_planning/lane_driving/lane_change_ready", 1, lanechangereadyCallback); //暫為使用

### Output

* node.advertise<std_msgs::Bool>("/planning/scenario_planning/lane_driving/obstacle_lane_change_approval", 10, true);
* node.advertise<std_msgs::Bool>("/planning/scenario_planning/lane_driving/motion_planning/obstacle_avoidance_planner/enable_avoidance", 10, true);
* node.advertise<std_msgs::Int32>("avoidpath_reach_goal", 10, true); 

### Description

* /planning/scenario_planning/lane_driving/obstacle_lane_change_approval : 發送lane change訊號
* /planning/scenario_planning/lane_driving/motion_planning/obstacle_avoidance_planner/enable_avoidance : 發送obstacle_avoidance_planning訊號
* avoidpath_reach_goal : 發送結束繞越訊號

### Supplymentary

* ```obstacle_lane_change_approval與obstacle_avoidance_planner/enable_avoidance```
	* 目前兩者都由同一個判斷式判斷是否要繞越,因此只能擇一,未來需要修改此判斷邏輯
	* 目前繞越主要由ACC作為判斷,當ACC速度cmd為0維持5秒後進入繞越模式
* avoidpath_reach_goal
	* 當前方無障礙物且車輛回到原車道後發出到達繞越終點

