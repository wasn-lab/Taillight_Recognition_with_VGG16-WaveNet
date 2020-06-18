cmake_minimum_required(VERSION 2.8.3)
project(waypoint_planner)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  pcl_ros
  pcl_conversions
  tf
  libwaypoint_follower
  autoware_msgs
  #vector_map
  astar_search
  msgs
  car_model
)

set(CMAKE_MODULE_PATH ${Project_SOURCE_DIR}/cmake ${car_model_DIR})
include(CarModel)
include(CompilerFlags)

catkin_package(
  CATKIN_DEPENDS
  roscpp
  std_msgs
  tf
  libwaypoint_follower
  autoware_msgs
  #vector_map
  msgs
  astar_search
  car_model
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

add_executable(astar_avoid
  src/astar_avoid/astar_avoid.cpp
  src/astar_avoid/astar_avoid_node.cpp
)

target_link_libraries(astar_avoid
  ${catkin_LIBRARIES}
)

add_dependencies(astar_avoid
  ${catkin_EXPORTED_TARGETS}
)

#add_executable(velocity_set
#  src/velocity_set/velocity_set.cpp
#  src/velocity_set/velocity_set_path.cpp
#  src/velocity_set/velocity_set_info.cpp
#  src/velocity_set/libvelocity_set.cpp
#)

#target_link_libraries(velocity_set
#  ${catkin_LIBRARIES}
#)

#add_dependencies(velocity_set
#  ${catkin_EXPORTED_TARGETS}
#)

install(TARGETS astar_avoid #velocity_set
        ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
