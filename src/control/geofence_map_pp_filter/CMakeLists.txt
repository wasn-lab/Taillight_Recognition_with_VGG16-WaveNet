cmake_minimum_required(VERSION 2.8.3)
project(geofence_map_pp_filter)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  autoware_perception_msgs
  message_generation
)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES lidarxyz2lla
  CATKIN_DEPENDS roscpp  autoware_perception_msgs message_runtime
#  DEPENDS system_lib
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

add_executable(geofence_map_pp_filter 
  src/geofence_map_pp_filter.cpp)

target_link_libraries(geofence_map_pp_filter 
  ${catkin_LIBRARIES})

add_dependencies(geofence_map_pp_filter
  ${catkin_EXPORTED_TARGETS} 
  ${${PROJECT_NAME}_EXPORTED_TARGETS})





