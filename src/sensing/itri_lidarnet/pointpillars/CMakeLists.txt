cmake_minimum_required(VERSION 2.8.3)
project(lidar_point_pillars)

# set flags for CUDA availability
option(CUDA_AVAIL "CUDA available" OFF)
find_package(CUDA)
#find_package(autoware_pointpillars_build_flags REQUIRED)

#AW_CHECK_CUDA()

set(USE_CUDA ON)

if (USE_CUDA)
  message("CUDA is available!")
  message("CUDA Libs: ${CUDA_LIBRARIES}")
  message("CUDA Headers: ${CUDA_INCLUDE_DIRS}")
  set(CUDA_AVAIL ON)
else()
  message("CUDA NOT FOUND OR INCOMPATIBLE CMAKE VERSION FOUND")
  set(CUDA_AVAIL OFF)
endif ()

# set flags for TensorRT availability
option(TRT_AVAIL "TensorRT available" OFF)
# try to find the tensorRT modules
find_library(NVINFER NAMES nvinfer)
find_library(NVPARSERS NAMES nvparsers)
find_library(NVONNXPARSERS NAMES nvonnxparser)

if(NVINFER AND NVPARSERS AND NVONNXPARSERS)
  message("TensorRT is available!")
  message("NVINFER: ${NVINFER}")
  message("NVPARSERS: ${NVPARSERS}")
  message("NVONNXPARSERS: ${NVONNXPARSERS}")
  set(TRT_AVAIL ON)
else()
  message("TensorRT is NOT Available")
  set(TRT_AVAIL OFF)
endif()

# set flags for CUDNN availability
option(CUDNN_AVAIL "CUDNN available" OFF)
# try to find the CUDNN module
find_library(CUDNN_LIBRARY
  NAMES libcudnn.so${__cudnn_ver_suffix} libcudnn${__cudnn_ver_suffix}.dylib ${__cudnn_lib_win_name}
  PATHS $ENV{LD_LIBRARY_PATH} ${__libpath_cudart} ${CUDNN_ROOT_DIR} ${PC_CUDNN_LIBRARY_DIRS} ${CMAKE_INSTALL_PREFIX}
  PATH_SUFFIXES lib lib64 bin
  DOC "CUDNN library."
)

if(CUDNN_LIBRARY)
  message("CUDNN is available!")
  message("CUDNN_LIBRARY: ${CUDNN_LIBRARY}")
  set(CUDNN_AVAIL ON)
else()
  message("CUDNN is NOT Available")
  set(CUDNN_AVAIL OFF)
endif()

if(TRT_AVAIL AND CUDA_AVAIL AND CUDNN_AVAIL)
  find_package(catkin REQUIRED COMPONENTS
    roscpp
    roslib
    pcl_ros
    msgs
    std_msgs
    #autoware_pointpillars_msgs
  )
  set(CMAKE_CXX_STANDARD 11)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

  catkin_package(
    CATKIN_DEPENDS
      roslib
      pcl_ros
      msgs
      std_msgs
      #autoware_pointpillars_msgs
  )

  include_directories(
    include
    ${catkin_INCLUDE_DIRS}
    ${msgs_SOURCE_DIR}/include
  )

  set(SOURCE_FILES
    # nodes/car/lidar_point_pillars_node_car.cpp
    # nodes/car/point_pillars_ros_car.cpp
    # nodes/ped_cyc/lidar_point_pillars_node_ped_cyc.cpp
    # nodes/ped_cyc/point_pillars_ros_ped_cyc.cpp
  )

  # car --------------------------------------------------------------------
  add_executable(lidar_point_pillars_car
    #${SOURCE_FILES}
    nodes/car/lidar_point_pillars_node_car.cpp
    nodes/car/point_pillars_ros_car.cpp
  )

  add_dependencies(lidar_point_pillars_car
    ${catkin_EXPORTED_TARGETS}
  )

  cuda_add_library(gpu_point_pillars_lib_car
    # common
    nodes/common/anchor_mask_cuda.cu
    nodes/common/scatter_cuda.cu
    nodes/common/nms_cuda.cu
    # car
    nodes/car/preprocess_points_cuda_car.cu
    nodes/car/postprocess_cuda_car.cu
  )

  target_link_libraries(gpu_point_pillars_lib_car
    ${CUDA_LIBRARIES}
  )

  add_library(point_pillars_lib_car
    nodes/car/point_pillars_car.cpp
    nodes/car/preprocess_points_car.cpp
  )

  target_link_libraries(point_pillars_lib_car
    ${NVINFER}
    ${NVONNXPARSERS}
    ${CUDA_LIBRARIES}
    # ${CUDA_CUBLAS_LIBRARIES}
    ${CUDA_curand_LIBRARY}
    ${CUDNN_LIBRARY}
    gpu_point_pillars_lib_car
  )

  target_link_libraries(lidar_point_pillars_car
    ${catkin_LIBRARIES}
    point_pillars_lib_car
  )

  install(
    TARGETS
      gpu_point_pillars_lib_car
      point_pillars_lib_car
      lidar_point_pillars_car
    ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
    LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
    RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
  )

  # ped_cyc -----------------------------------------------------
  add_executable(lidar_point_pillars_ped_cyc
    #${SOURCE_FILES}
    nodes/ped_cyc/point_pillars_ros_ped_cyc.cpp
    nodes/ped_cyc/lidar_point_pillars_node_ped_cyc.cpp
  )

  add_dependencies(lidar_point_pillars_ped_cyc
    ${catkin_EXPORTED_TARGETS}
  )

  cuda_add_library(gpu_point_pillars_lib_ped_cyc
    # common
    nodes/common/anchor_mask_cuda.cu
    nodes/common/scatter_cuda.cu
    nodes/common/nms_cuda.cu
    # ped_cyc
    nodes/ped_cyc/preprocess_points_cuda_ped_cyc.cu
    nodes/ped_cyc/postprocess_cuda_ped_cyc.cu
  )

  target_link_libraries(gpu_point_pillars_lib_ped_cyc
    ${CUDA_LIBRARIES}
  )

  add_library(point_pillars_lib_ped_cyc
    nodes/ped_cyc/point_pillars_ped_cyc.cpp
    nodes/ped_cyc/preprocess_points_ped_cyc.cpp
  )

  target_link_libraries(point_pillars_lib_ped_cyc
    ${NVINFER}
    ${NVONNXPARSERS}
    ${CUDA_LIBRARIES}
    # ${CUDA_CUBLAS_LIBRARIES}
    ${CUDA_curand_LIBRARY}
    ${CUDNN_LIBRARY}
    gpu_point_pillars_lib_ped_cyc
  )

  target_link_libraries(lidar_point_pillars_ped_cyc
    ${catkin_LIBRARIES}
    point_pillars_lib_ped_cyc
  )

  install(
    TARGETS
      gpu_point_pillars_lib_ped_cyc
      point_pillars_lib_ped_cyc
      lidar_point_pillars_ped_cyc
    ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
    LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
    RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
  )


  #-----------------------------------------------------
  install(DIRECTORY launch/
    DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
    PATTERN ".svn" EXCLUDE
  )

  install(DIRECTORY include/
    DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}/${PROJECT_NAME}/
    PATTERN ".svn" EXCLUDE
  )

  ### test
  if (CATKIN_ENABLE_TESTING AND EXISTS /proc/driver/nvidia/)
    find_package(rostest REQUIRED)
    catkin_add_gtest(test-point_pillars test/src/test_point_pillars.cpp)
    target_link_libraries(test-point_pillars ${catkin_LIBRARIES} point_pillars_lib)
  endif()



else()
  find_package(catkin REQUIRED)
  catkin_package()
  message("PointPillars won't be built, CUDA and/or TensorRT were not found.")
endif()
