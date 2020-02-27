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
message(STATUS "TensorRT_LIBRARY: ${TensorRT_LIBRARY}")
if (NOT TensorRT_FOUND)
  message("Cannot find TensorRT. Skip ${PROJECT_NAME}")
  return()
endif()

catkin_package(
  INCLUDE_DIRS include
)

set(LIBS drivenet_lib
         object_costmap_generator
         camera_utils
         ${OpenCV_LIBRARIES}
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
  ${DRIVENETLIB_INCLUDE}
  ${OBJECTCOSTMAPGENERATOR_INCLUDE}
  ${CUDA_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS}
)
message(STATUS "DRIVENETLIB_INCLUDE: ${DRIVENETLIB_INCLUDE}")

link_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CUDA_LIBRARY_DIRS}
  ${OpenCV_LIBRARY_DIRS}
)

###
file(GLOB_RECURSE LIB_SOURCES *.cpp)
list(REMOVE_ITEM LIB_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/drivenet_60_b1.cpp" )
list(REMOVE_ITEM LIB_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/drivenet_120_1_b1.cpp" )
list(REMOVE_ITEM LIB_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/drivenet_120_2_b1.cpp" )
list(REMOVE_ITEM LIB_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/campub_all.cpp" )
list(REMOVE_ITEM LIB_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/alignment_offline_b1.cpp" )
list(REMOVE_ITEM LIB_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/distance_estimation_b1.cpp" )
list(REMOVE_ITEM LIB_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/distance_estimation_b1_v2.cpp" )
list(REMOVE_ITEM LIB_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/drivenet_group_a_b1_v2.cpp" )
list(REMOVE_ITEM LIB_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/drivenet_group_b_b1_v2.cpp" )
list(REMOVE_ITEM LIB_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/drivenet_group_c_b1_v2.cpp" )
message(STATUS "Lib_sources: " ${LIB_SOURCES})
add_library(${PROJECT_NAME} SHARED ${LIB_SOURCES})
file(GLOB_RECURSE EXCEPTION_SRCS src/exception_handling/*.cpp)
file(GLOB_RECURSE COMMON_UTILS src/common_utils/*.cpp)

if (CAR_MODEL_IS_B1)
  add_executable(${PROJECT_NAME}_node src/drivenet_60_b1.cpp src/distance_estimation_b1.cpp ${LIB_SOURCES} ${EXCEPTION_SRCS} ${COMMON_UTILS})
  add_executable(${PROJECT_NAME}_node1 src/drivenet_120_1_b1.cpp src/distance_estimation_b1.cpp ${LIB_SOURCES} ${EXCEPTION_SRCS} ${COMMON_UTILS})
  add_executable(${PROJECT_NAME}_node2 src/drivenet_120_2_b1.cpp src/distance_estimation_b1.cpp ${LIB_SOURCES} ${EXCEPTION_SRCS} ${COMMON_UTILS})
  add_executable(${PROJECT_NAME}_node3 src/campub_all.cpp  ${EXCEPTION_SRCS})
  add_executable(${PROJECT_NAME}_node4 src/alignment_offline_b1.cpp ${EXCEPTION_SRCS} ${COMMON_UTILS})

  add_dependencies(${PROJECT_NAME} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
  add_dependencies(${PROJECT_NAME}_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
  add_dependencies(${PROJECT_NAME}_node1 ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
  add_dependencies(${PROJECT_NAME}_node2 ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
  add_dependencies(${PROJECT_NAME}_node3 ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
  add_dependencies(${PROJECT_NAME}_node4 ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

  target_link_libraries(${PROJECT_NAME} ${catkin_LIBRARIES} ${LIBS})
  target_link_libraries(${PROJECT_NAME}_node ${catkin_LIBRARIES} ${LIBS})
  target_link_libraries(${PROJECT_NAME}_node1 ${catkin_LIBRARIES} ${LIBS})
  target_link_libraries(${PROJECT_NAME}_node2 ${catkin_LIBRARIES} ${LIBS})
  target_link_libraries(${PROJECT_NAME}_node3 ${catkin_LIBRARIES} ${LIBS})
  target_link_libraries(${PROJECT_NAME}_node4 ${catkin_LIBRARIES} ${LIBS})

elseif(CAR_MODEL_IS_B1_V2)
  add_executable(${PROJECT_NAME}_node_a src/drivenet_group_a_b1_v2.cpp src/distance_estimation_b1_v2.cpp ${LIB_SOURCES} ${EXCEPTION_SRCS} ${COMMON_UTILS})
  add_executable(${PROJECT_NAME}_node_b src/drivenet_group_b_b1_v2.cpp src/distance_estimation_b1_v2.cpp ${LIB_SOURCES} ${EXCEPTION_SRCS} ${COMMON_UTILS})
  add_executable(${PROJECT_NAME}_node_c src/drivenet_group_c_b1_v2.cpp src/distance_estimation_b1_v2.cpp ${LIB_SOURCES} ${EXCEPTION_SRCS} ${COMMON_UTILS})

  add_dependencies(${PROJECT_NAME}_node_a ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
  add_dependencies(${PROJECT_NAME}_node_b ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
  add_dependencies(${PROJECT_NAME}_node_c ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

  target_link_libraries(${PROJECT_NAME}_node_a ${catkin_LIBRARIES} ${LIBS})
  target_link_libraries(${PROJECT_NAME}_node_b ${catkin_LIBRARIES} ${LIBS})
  target_link_libraries(${PROJECT_NAME}_node_c ${catkin_LIBRARIES} ${LIBS})
endif ()
