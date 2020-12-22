cmake_minimum_required(VERSION 2.8.3)
project(astar_search_ai)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  tf
  tf_conversions
  pcl_ros
  sensor_msgs
  geometry_msgs
  autoware_msgs
  car_model
)

set(CMAKE_MODULE_PATH ${Project_SOURCE_DIR}/cmake ${car_model_DIR})
include(CarModel)
include(CompilerFlags)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES astar_search_ai
  CATKIN_DEPENDS
  roscpp
  tf
  tf_conversions
  pcl_ros
  sensor_msgs
  geometry_msgs
  autoware_msgs
  car_model
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

add_library(astar_search_ai
  src/astar_search_ai.cpp
  src/astar_util_ai.cpp
)

target_link_libraries(astar_search_ai
  ${catkin_LIBRARIES}
)

add_dependencies(astar_search_ai
  ${catkin_EXPORTED_TARGETS}
)

install(DIRECTORY include/${PROJECT_NAME}/
	DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
	FILES_MATCHING PATTERN "*.h"
)

install(TARGETS astar_search_ai
	ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
	LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
	RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

#if(CATKIN_ENABLE_TESTING)
#  find_package(rostest REQUIRED)
#  add_rostest_gtest(astar_search-test test/test_astar_search.test test/src/test_main.cpp test/src/test_astar_util.cpp test/src/test_astar_search.cpp test/src/test_class.cpp)
#  target_link_libraries(astar_search-test ${catkin_LIBRARIES} astar_search)
#endif()
