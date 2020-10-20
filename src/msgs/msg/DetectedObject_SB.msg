Header header

# classification result, see include/detected_object_class_id.h for definitions
uint16 classId

float32 distance
float32 absSpeed
float32 relSpeed

PointXYZ center_point
PointXYZ center_point_gps
Quaternion heading
Quaternion heading_enu
BoxSize dimension

BoxPoint bPoint
ConvexPoint cPoint


TrackInfo track
uint16 fusionSourceId

CamInfo camInfo
RadarInfo radarInfo
LidarInfo lidarInfo
