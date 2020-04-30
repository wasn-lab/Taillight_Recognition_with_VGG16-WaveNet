cmake_minimum_required(VERSION 2.8.3)
project(map_pub)
find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  message_generation
  genmsg
  roslib
  tf
)

set(CMAKE_MODULE_PATH ${Project_SOURCE_DIR}/cmake)
include(CompilerFlags)

## Find Point Cloud Library 1.8
find_package(PCL 1.8.1 REQUIRED)

find_package(Eigen3)

#generate_messages(
#  DEPENDENCIES
#  std_msgs
#)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS
  roscpp
  std_msgs

)
set(POINTS_MAP_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/done_map)

include_directories(
 include
  ${catkin_INCLUDE_DIRS}
)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(${PROJECT_NAME}_node src/map_pub_node.cpp)

target_link_libraries(${PROJECT_NAME}_node ${catkin_LIBRARIES} ${PCL_LIBRARIES})

MESSAGE("Generate ${PROJECT_NAME}")
