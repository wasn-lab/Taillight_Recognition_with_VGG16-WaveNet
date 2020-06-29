geofence package
(1)geofence

### Input
n.subscribe("ring_edge_point_cloud", 1, callback_LidarAll);
n.subscribe("nav_path_astar_final", 1, astar_callback);
n.subscribe("nav_path_astar_base_30", 1, astar_original_callback);
n.subscribe("veh_predictpath", 1, deviate_path_callback);
n.subscribe("localization_to_veh", 1, LocalizationToVehCallback);
n.subscribe("astar_reach_goal", 1, overtake_over_Callback);
n.subscribe("PathPredictionOutput/radar", 1, chatterCallbackPCloud_Radar);
n.subscribe("/CameraDetection/polygon", 1, chatterCallbackCPoint);
(suspended)n.subscribe("dynamic_path_para", 1, chatterCallbackPoly);





### Output
n.advertise<visualization_msgs::Marker>("RadarMarker", 1);
n.advertise<visualization_msgs::Marker>("Geofence_line", 1);
n.advertise<std_msgs::Float64>("Geofence_PC", 1);
n.advertise<std_msgs::Float64>("Geofence_original", 1); 

### Description
(1) 正常行駛時計算路徑上物件geofence(lidar ring_edge)
(2) 正常行駛時計算路徑上物件geofence(camera polygon)
(3) 正常行駛時計算路徑上物件geofence(radar data)
(4) 車輛偏離時,計算預測路徑(偏離軌跡)上物件geofence
(5) 車輛繞越時,計算原路徑上的geofence
(6) 透過CAN傳送結果給dsapce
(7) 產生geofence marker給rviz



### Supplymentary
geofence class

geofence object可以計算物件或點雲資訊是否有在行駛軌跡上
Geofence XXX_Geofence(1.2);  //1.2為車道距離寬(由中縣向左又)


intialization
(1)Geofence::setPath
	->設定路徑
(2)Geofence::setPointCloud
	->設定點雲資訊(或是bounding box/polygon)

execution
(1)Geofence::Calculator()
	->計算geofence

function
(1)Geofence::getDistance()
	->取得路徑上物件的距離
(2)Geofence::getDistance_w()
	->取得路徑上較寬的範圍內(設定值+0.5)物件的距離
(3)Geofence::getFarest()
	->取得geofence對應的物件最遠的點(適用BBox), 若為點雲則與Geofence::getDistance()相同
(4)Geofence::getTrigger()
	->是否有物件在路徑上
(5)Geofence::getObjSpeed()
	->gefence對應之物件速度
(6)Geofence::getNearest_X()
	->gefence對應之物件x座標
(7)Geofence::getNearest_Y()
	->gefence對應之物件y座標
(8)Geofence::findDirection()
	->gefence所在的路徑切線方向

