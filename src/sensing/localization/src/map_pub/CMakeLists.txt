cmake_minimum_required(VERSION 2.8.3)
project(map_pub)
add_compile_options(-std=c++11)
set(CMAKE_BUILD_TYPE "Release")
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif()

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  message_generation
  genmsg
  roslib
  tf
)

## Find Point Cloud Library 1.8
find_package(PCL 1.8.1 REQUIRED)


generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS
  roscpp
  std_msgs

)
set(POINTS_MAP_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/done_map)


if (NOT EXISTS ${POINTS_MAP_DIRS}/total_map.pcd)
      message( STATUS " downloading points map" )

    file(
    DOWNLOAD
    	http://118.163.54.109:8888/Share/ADV/LiDAR_maps/done_map_total_itri_downsample/total_map.pcd ${POINTS_MAP_DIRS}/total_map.pcd    	
    STATUS
        status
    LOG 
        log
    SHOW_PROGRESS
    )
endif()

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
