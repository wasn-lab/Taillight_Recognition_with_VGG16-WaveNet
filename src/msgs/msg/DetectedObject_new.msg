Header header

# classification result, see include/detected_object_class_id.h for definitions
uint16 classId

float32 distance

# absolute speed
float32 absSpeed
# speed of "'relative velocity on relative coordinate'  //
# projecting onto object-to-ego-vehicle vector"
float32 relSpeed

BoxPoint bPoint
ConvexPoint cPoint
PointXYZ centerPoint
Quaternion bOrient

TrackInfo track

# detection source, see include/fusion_source_id.h for definition.'
uint16 fusionSourceId

CamInfo camInfo
RadarInfo radarInfo
LidarInfo lidarInfo
