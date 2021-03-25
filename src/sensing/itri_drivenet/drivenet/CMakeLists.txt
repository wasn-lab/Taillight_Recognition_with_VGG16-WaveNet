cmake_minimum_required(VERSION 2.8.3)
project(drivenet)

set(CMAKE_MODULE_PATH ${Project_SOURCE_DIR}/cmake ${car_model_DIR})
include(CompilerFlags)
include(CarModel)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  msgs
  image_transport
  cv_bridge
  car_model
  camera_utils
  drivenet_lib
  object_costmap_generator
)
set(CMAKE_MODULE_PATH ${Project_SOURCE_DIR}/cmake)

find_package(OpenCV REQUIRED PATHS /usr/local NO_DEFAULT_PATH)
if (NOT ${OpenCV_FOUND})
  message("OpenCV not found. Skip building ${PROJECT_NAME}")
  return ()
else()
  #message(STATUS "OpenCV IncludeDir: ${OpenCV_INCLUDE_DIRS}")
  #message(STATUS "OpenCV Libararies: ${OpenCV_LIBRARIES}")
  #message(STATUS "OpenCV Libararies Dir: ${OpenCV_LIBRARY_DIRS}")

  if(EXISTS /usr/local/lib/libopencv_core.so)
    set(OpenCV_USER_LIBRARIES /usr/local/lib/libopencv_core.so
                              ${OpenCV_LIBRARIES})
  else()
    set(OpenCV_USER_LIBRARIES ${OpenCV_LIBRARIES})
  endif()
endif ()

######################
##       PCL        ##
######################
find_package            (PCL 1.8 REQUIRED) 
include_directories     (${PCL_INCLUDE_DIRS})
link_directories        (${PCL_LIBRARY_DIRS})
add_definitions         (${PCL_DEFINITIONS})

find_package(CUDA)
if (NOT ${CUDA_FOUND})
  message("CUDA not found. Skip building ${PROJECT_NAME}")
  return ()
endif ()

find_package(TensorRT)
if (NOT TensorRT_FOUND)
  message("Cannot find TensorRT. Skip ${PROJECT_NAME}")
  return()
endif()
#message(STATUS "TensorRT_LIBRARY: ${TensorRT_LIBRARY}")

catkin_package(
  INCLUDE_DIRS include
)

set(LIBS drivenet_lib
         object_costmap_generator
         camera_utils
         ${OpenCV_USER_LIBRARIES}
         ${PCL_LIBRARIES}
         nvinfer
         nvinfer_plugin
         ${CUDA_LIBRARIES}
         ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libnppisu.so
         ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libnppig.so
         gflags
         stdc++fs)

add_definitions("-DNPP")

include_directories(
  include
  ${INCLUDE_DIRS}
  ${car_model_BINARY_DIR}/include
  ${msgs_SOURCE_DIR}/include
  ${camera_utils_SOURCE_DIR}/include
  ${drivenet_lib_SOURCE_DIR}/include
  ${object_costmap_generator_SOURCE_DIR}/include
  ${CUDA_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS}
)
#message(STATUS "DRIVENETLIB_INCLUDE: ${DRIVENETLIB_INCLUDE}")

link_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CUDA_LIBRARY_DIRS}
  ${OpenCV_LIBRARY_DIRS}
)

###
file(GLOB_RECURSE LIB_SOURCES *.cpp)
list(REMOVE_ITEM LIB_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/campub_all.cpp" )
list(REMOVE_ITEM LIB_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/alignment_offline_b1.cpp" )
list(REMOVE_ITEM LIB_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/distance_estimation_b1_v2.cpp" )
list(REMOVE_ITEM LIB_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/drivenet_group_a.cpp" )
list(REMOVE_ITEM LIB_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/drivenet_group_b.cpp" )
list(REMOVE_ITEM LIB_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/drivenet_group_c.cpp" )
list(REMOVE_ITEM LIB_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/drivenet_side_detection.cpp" )
list(REMOVE_ITEM LIB_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/drivenet_top_detection.cpp" )
list(REMOVE_ITEM LIB_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/drivenet_single_camera.cpp" )
list(REMOVE_ITEM LIB_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/detection_image_repub.cpp" )
#message(STATUS "Lib_sources: " ${LIB_SOURCES})
file(GLOB_RECURSE EXCEPTION_SRCS src/exception_handling/*.cpp)
file(GLOB_RECURSE COMMON_UTILS src/common_utils/*.cpp)

cuda_add_library(${PROJECT_NAME} SHARED ${LIB_SOURCES} ${EXCEPTION_SRCS} ${COMMON_UTILS})
add_dependencies(${PROJECT_NAME} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(${PROJECT_NAME} ${LIBS} ${catkin_LIBRARIES} )

add_executable(${PROJECT_NAME}_util_node_obj_pub src/campub_all.cpp  ${EXCEPTION_SRCS})
add_dependencies(${PROJECT_NAME}_util_node_obj_pub ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(${PROJECT_NAME}_util_node_obj_pub ${LIBS} ${catkin_LIBRARIES} )

add_executable(${PROJECT_NAME}_util_node_img_repub src/detection_image_repub.cpp ${LIB_SOURCES} ${EXCEPTION_SRCS})
add_dependencies(${PROJECT_NAME}_util_node_img_repub ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(${PROJECT_NAME}_util_node_img_repub ${LIBS} ${catkin_LIBRARIES} )  

if (CAR_MODEL_IS_B1_V2)
  add_executable(${PROJECT_NAME}_detect_node_a_b1_v2 src/drivenet_group_a.cpp src/distance_estimation_b1_v2.cpp ${LIB_SOURCES} ${EXCEPTION_SRCS} ${COMMON_UTILS})
  add_dependencies(${PROJECT_NAME}_detect_node_a_b1_v2 ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
  target_link_libraries(${PROJECT_NAME}_detect_node_a_b1_v2 ${LIBS} ${catkin_LIBRARIES} )

  add_executable(${PROJECT_NAME}_detect_node_side_b1_v2 src/drivenet_side_detection.cpp src/distance_estimation_b1_v2.cpp ${LIB_SOURCES} ${EXCEPTION_SRCS} ${COMMON_UTILS})
  add_dependencies(${PROJECT_NAME}_detect_node_side_b1_v2 ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
  target_link_libraries(${PROJECT_NAME}_detect_node_side_b1_v2 ${LIBS} ${catkin_LIBRARIES} )

  add_executable(${PROJECT_NAME}_detect_node_top_b1_v2 src/drivenet_top_detection.cpp src/distance_estimation_b1_v2.cpp ${LIB_SOURCES} ${EXCEPTION_SRCS} ${COMMON_UTILS})
  add_dependencies(${PROJECT_NAME}_detect_node_top_b1_v2 ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
  target_link_libraries(${PROJECT_NAME}_detect_node_top_b1_v2 ${LIBS} ${catkin_LIBRARIES} )

  add_executable(${PROJECT_NAME}_detect_node_single_cam_b1_v2 src/drivenet_single_camera.cpp src/distance_estimation_b1_v2.cpp ${LIB_SOURCES} ${EXCEPTION_SRCS} ${COMMON_UTILS})
  add_dependencies(${PROJECT_NAME}_detect_node_single_cam_b1_v2 ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
  target_link_libraries(${PROJECT_NAME}_detect_node_single_cam_b1_v2 ${LIBS} ${catkin_LIBRARIES} )

  add_executable(${PROJECT_NAME}_depth_node_alignment_b1_v2 src/alignment_offline_b1.cpp ${EXCEPTION_SRCS} ${COMMON_UTILS})
  add_dependencies(${PROJECT_NAME}_depth_node_alignment_b1_v2 ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
  target_link_libraries(${PROJECT_NAME}_depth_node_alignment_b1_v2 ${LIBS} ${catkin_LIBRARIES} )

elseif(CAR_MODEL_IS_B1_V3)
  add_executable(${PROJECT_NAME}_detect_node_a_b1_v3 src/drivenet_group_a.cpp src/distance_estimation_b1_v2.cpp ${LIB_SOURCES} ${EXCEPTION_SRCS} ${COMMON_UTILS})
  add_dependencies(${PROJECT_NAME}_detect_node_a_b1_v3 ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
  target_link_libraries(${PROJECT_NAME}_detect_node_a_b1_v3 ${LIBS} ${catkin_LIBRARIES} )

  add_executable(${PROJECT_NAME}_detect_node_side_b1_v3 src/drivenet_side_detection.cpp src/distance_estimation_b1_v2.cpp ${LIB_SOURCES} ${EXCEPTION_SRCS} ${COMMON_UTILS})
  add_dependencies(${PROJECT_NAME}_detect_node_side_b1_v3 ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
  target_link_libraries(${PROJECT_NAME}_detect_node_side_b1_v3 ${LIBS} ${catkin_LIBRARIES} )

  add_executable(${PROJECT_NAME}_detect_node_top_b1_v3 src/drivenet_top_detection.cpp src/distance_estimation_b1_v2.cpp ${LIB_SOURCES} ${EXCEPTION_SRCS} ${COMMON_UTILS})
  add_dependencies(${PROJECT_NAME}_detect_node_top_b1_v3 ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
  target_link_libraries(${PROJECT_NAME}_detect_node_top_b1_v3 ${LIBS} ${catkin_LIBRARIES} )

  add_executable(${PROJECT_NAME}_detect_node_single_cam_b1_v3 src/drivenet_single_camera.cpp src/distance_estimation_b1_v2.cpp ${LIB_SOURCES} ${EXCEPTION_SRCS} ${COMMON_UTILS})
  add_dependencies(${PROJECT_NAME}_detect_node_single_cam_b1_v3 ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
  target_link_libraries(${PROJECT_NAME}_detect_node_single_cam_b1_v3 ${LIBS} ${catkin_LIBRARIES} )

  add_executable(${PROJECT_NAME}_depth_node_alignment_b1_v2 src/alignment_offline_b1.cpp ${EXCEPTION_SRCS} ${COMMON_UTILS})
  add_dependencies(${PROJECT_NAME}_depth_node_alignment_b1_v2 ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
  target_link_libraries(${PROJECT_NAME}_depth_node_alignment_b1_v2 ${LIBS} ${catkin_LIBRARIES} )

elseif(CAR_MODEL_IS_C1)
  add_executable(${PROJECT_NAME}_detect_node_a_c1 src/drivenet_group_a.cpp src/distance_estimation_b1_v2.cpp ${LIB_SOURCES} ${EXCEPTION_SRCS} ${COMMON_UTILS})
  add_dependencies(${PROJECT_NAME}_detect_node_a_c1 ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
  target_link_libraries(${PROJECT_NAME}_detect_node_a_c1 ${LIBS} ${catkin_LIBRARIES} )

  add_executable(${PROJECT_NAME}_detect_node_side_c1 src/drivenet_side_detection.cpp src/distance_estimation_b1_v2.cpp ${LIB_SOURCES} ${EXCEPTION_SRCS} ${COMMON_UTILS})
  add_dependencies(${PROJECT_NAME}_detect_node_side_c1 ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
  target_link_libraries(${PROJECT_NAME}_detect_node_side_c1 ${LIBS} ${catkin_LIBRARIES} )

  add_executable(${PROJECT_NAME}_detect_node_top_c1 src/drivenet_top_detection.cpp src/distance_estimation_b1_v2.cpp ${LIB_SOURCES} ${EXCEPTION_SRCS} ${COMMON_UTILS})
  add_dependencies(${PROJECT_NAME}_detect_node_top_c1 ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
  target_link_libraries(${PROJECT_NAME}_detect_node_top_c1 ${LIBS} ${catkin_LIBRARIES} )

  add_executable(${PROJECT_NAME}_depth_node_alignment_b1_v2 src/alignment_offline_b1.cpp ${EXCEPTION_SRCS} ${COMMON_UTILS})
  add_dependencies(${PROJECT_NAME}_depth_node_alignment_b1_v2 ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
  target_link_libraries(${PROJECT_NAME}_depth_node_alignment_b1_v2 ${LIBS} ${catkin_LIBRARIES} )
endif ()
