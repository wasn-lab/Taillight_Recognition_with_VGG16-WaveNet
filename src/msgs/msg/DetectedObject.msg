Header header

# classification result
# 0: Unknown
# 1: Person
# 2: Bicycle
# 3: Motobike
# 4: Car
# 5: Bus
# 6: Truck
# 7: Sign
# 8: Light
# 9: Park
uint16 classId

float32 distance

# absolute speed
float32 absSpeed
# speed of "'relative velocity on relative coordinate'  //
# projecting onto object-to-ego-vehicle vector"
float32 relSpeed

BoxPoint bPoint
ConvexPoint cPoint
Quaternion bOrient

TrackInfo track

# detection source
# 0: Camera
# 1: Radar
# 2: Lidar
uint16 fusionSourceId

CamInfo camInfo
RadarInfo radarInfo
LidarInfo lidarInfo

PedInfo pedInfo