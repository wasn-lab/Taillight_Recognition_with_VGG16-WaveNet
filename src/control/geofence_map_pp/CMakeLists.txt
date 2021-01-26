cmake_minimum_required(VERSION 3.1)
project(geofence_map_pp)

set(CMAKE_MODULE_PATH ${Project_SOURCE_DIR}/cmake)
include(CompilerFlags)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  tf2_ros
  tf2_geometry_msgs
  message_generation
  msgs
  car_model
)

catkin_package(CATKIN_DEPENDS
  roscpp
  rospy
  std_msgs
  tf2_ros
  tf2_geometry_msgs
  message_runtime
  msgs
  car_model
)

find_package(PCL)

link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

include_directories(${catkin_INCLUDE_DIRS}
${PCL_INCLUDE_DIRS}
)

add_executable(geofence_map_pp src/main.cpp src/Geofence.cpp)
target_link_libraries(geofence_map_pp ${catkin_LIBRARIES} ${PCL_LIBRARIES})
add_dependencies(geofence_map_pp ${catkin_EXPORTED_TARGETS})
