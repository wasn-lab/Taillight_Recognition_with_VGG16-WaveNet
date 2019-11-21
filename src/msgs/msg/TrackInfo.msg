# track id
uint32 id


# track history
# ----------------------------------
# track time
uint32 tracktime

# history vector
TrackState[] states

# history vector size
uint16 length  
# history vector capacity
uint8 max_length 

bool is_over_max_length
uint8 head


# velocity estimation
# ----------------------------------
# absolute velocity on relative coordinate
PointXYZV absolute_velocity
# relative velocity on relative coordinate
PointXYZV relative_velocity


# path prediction
# ----------------------------------
bool is_ready_prediction
PathPrediction[] forecasts
