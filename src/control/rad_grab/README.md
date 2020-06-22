rad_grab package
(1) RadFrontPub
(2) RadFrontPub_txt
(3) RadFrontSub(suspended)
(4) RadFrontSub_BBox
(5) RadFrontSub_CAN(suspended)
(6) RadFrontSub_PCloud
(7) RadFrontSub_rviz


RadFrontPub
### Input
CAN(radar raw data)

### Output
n.advertise<msgs::Rad>("RadFront", 1);

### Description
由d組提供的radar grabber


RadFrontPub_txt
### Input
CAN(radar raw data)

### Output
.txt

### Description
由d組提供的radar raw data轉txt



RadFrontSub_BBox
### Input
n.subscribe("RadFront", 1, callbackRadFront)

### Output
n.advertise<msgs::DetectedObjectArray>("PathPredictionOutput/radar", 1)

### Description
將raw data轉成BBox形式供geofence使用



RadFrontSub_PCloud
### Input
n.subscribe("RadFront", 1, callbackRadFront)

### Output
n.advertise<sensor_msgs::PointCloud2>("radar_point_cloud", 1);

### Description
將raw data轉成point cloud形式



RadFrontSub_rviz
### Input
n.subscribe("RadFront", 1, callbackRadFront)

### Output
n.advertise<visualization_msgs::Marker>("RadarPlotter", 1);

### Description
將raw data轉成marker形式供rviz使用






