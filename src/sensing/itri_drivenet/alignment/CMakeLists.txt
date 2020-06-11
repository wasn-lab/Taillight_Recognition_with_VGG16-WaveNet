cmake_minimum_required(VERSION 2.8.3)
project(alignment)
 
set(CMAKE_MODULE_PATH ${Project_SOURCE_DIR}/cmake ${car_model_DIR})
include(CompilerFlags)
include(CarModel)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  roslib
  cv_bridge
  msgs
  car_model
  camera_utils
)  

catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS roscpp rospy msgs
)

find_package(libs)                
if (NOT ${libs_FOUND})
  message("libs not found. Skip building ${PROJECT_NAME}")
  return ()
endif ()

find_package(drivenet)                
if (NOT ${drivenet_FOUND})
  message("drivenet not found. Skip building ${PROJECT_NAME}")
  return ()
endif ()

######################
##      OPENCV      ##
######################
find_package(OpenCV REQUIRED PATHS /usr/local NO_DEFAULT_PATH)
if (NOT ${OpenCV_FOUND})
  message("OpenCV not found. Skip building ${PROJECT_NAME}")
  return ()
endif ()

######################
##       CUDA       ##
######################
find_package(CUDA) 
if (NOT ${CUDA_FOUND})
  message("CUDA not found. Skip building ${PROJECT_NAME}")
  return ()
endif ()

######################
##       PCL        ##
######################
find_package            (PCL 1.8 REQUIRED) 
if (NOT ${PCL_FOUND})
  message("PCL not found. Skip building ${PROJECT_NAME}")
  return ()
endif ()
include_directories     (${PCL_INCLUDE_DIRS})
link_directories        (${PCL_LIBRARY_DIRS})
add_definitions         (${PCL_DEFINITIONS})


include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
  ${car_model_BINARY_DIR}/include
  ${msgs_SOURCE_DIR}/include
  ${camera_utils_SOURCE_DIR}/include
  ${drivenet_lib_SOURCE_DIR}/include
  ${drivenet_SOURCE_DIR}/include
  ${libs_SOURCE_DIR}
  ${libs_SOURCE_DIR}/BoundingBox/ApproxMVBB
  ${libs_SOURCE_DIR}/Preprocess/VoxelGrid
  ${libs_SOURCE_DIR}/Clustering/DBSCAN/CUDA
)

set(LIBS ${OpenCV_LIBRARIES}
         ${CUDA_LIBRARIES}
         ${PCL_LIBRARIES}
         camera_utils
         drivenet
         libs)

if (CAR_MODEL_IS_B1)
  cuda_add_executable(${PROJECT_NAME}_b1 src/main_b1.cpp src/alignment.cpp src/visualization_util.cpp src/point_preprocessing.cpp src/points_in_image_area.cpp src/sync_message.cpp src/object_generator.cpp src/cloud_cluster.cpp)
  target_link_libraries(${PROJECT_NAME}_b1 ${catkin_LIBRARIES} ${LIBS})
  add_dependencies(${PROJECT_NAME}_b1 ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

  cuda_add_executable(${PROJECT_NAME}_b1_ssn_sub src/lidar_ssn_sub.cpp src/ssn_util.cpp)
  target_link_libraries(${PROJECT_NAME}_b1_ssn_sub ${catkin_LIBRARIES} ${LIBS})
  add_dependencies(${PROJECT_NAME}_b1_ssn_sub ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
elseif(CAR_MODEL_IS_B1_V2)
  cuda_add_executable(${PROJECT_NAME}_b1_v2 src/main_b1_v2.cpp src/alignment.cpp src/visualization_util.cpp src/point_preprocessing.cpp src/points_in_image_area.cpp src/sync_message.cpp src/object_generator.cpp src/cloud_cluster.cpp)
  target_link_libraries(${PROJECT_NAME}_b1_v2 ${catkin_LIBRARIES} ${LIBS})
  add_dependencies(${PROJECT_NAME}_b1_v2 ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

  cuda_add_executable(${PROJECT_NAME}_b1_v2_alignment src/alignment_visualization_b1_v2.cpp src/alignment.cpp src/visualization_util.cpp src/point_preprocessing.cpp src/points_in_image_area.cpp src/cloud_cluster.cpp)
  target_link_libraries(${PROJECT_NAME}_b1_v2_alignment ${catkin_LIBRARIES} ${LIBS})
  add_dependencies(${PROJECT_NAME}_b1_v2_alignment ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

  cuda_add_executable(${PROJECT_NAME}_b1_v2_ssn_sub src/lidar_ssn_sub.cpp src/ssn_util.cpp)
  target_link_libraries(${PROJECT_NAME}_b1_v2_ssn_sub ${catkin_LIBRARIES} ${LIBS})
  add_dependencies(${PROJECT_NAME}_b1_v2_ssn_sub ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
endif ()

