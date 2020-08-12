# rad_grab package
- RadFrontPub
- RadFrontPub_txt
- RadFrontSub(suspended)
- RadFrontSub_BBox
- RadFrontSub_CAN(suspended)
- RadFrontSub_PCloud
- RadFrontSub_rviz


# RadFrontPub
### Input
CAN(radar raw data)

### Output
n.advertise<msgs::Rad>("RadFront", 1);

### Description
由d組提供的radar grabber


# RadFrontPub_txt
### Input
CAN(radar raw data)

### Output
.txt

### Description
由d組提供的radar raw data轉txt



# RadFrontSub_BBox
### Input
n.subscribe("RadFront", 1, callbackRadFront)

### Output
n.advertise<msgs::DetectedObjectArray>("PathPredictionOutput/radar", 1)

### Description
將raw data轉成BBox形式供geofence使用



# RadFrontSub_PCloud
### Input
n.subscribe("RadFront", 1, callbackRadFront)

### Output
n.advertise<sensor_msgs::PointCloud2>("radar_point_cloud", 1);

### Description
將raw data轉成point cloud形式



# RadFrontSub_rviz
### Input
n.subscribe("RadFront", 1, callbackRadFront)

### Output
n.advertise<visualization_msgs::Marker>("RadarPlotter", 1);

### Description
將raw data轉成marker形式供rviz使用
