set(PARKNET_LIB "parknet")
set(PARKNET_EXE "itri_parknet")

# Setup *.cpp to build libparknet
set(PARKNET_LIB_SRCS
#    npp_wrapper.cpp
    parknet_advertise_utils.cpp
    parknet_fs_utils.cpp
    parknet_args_parser.cpp
    parknet_camera.cpp
    parknet_cv_colors.cpp
    parknet_visualizer.cpp
    parknet_image_utils.cpp
    parknet_image_manager.cpp
    parknet_pcd_manager.cpp
    parknet_logging.cpp
    parknet_subscription.cpp
    parknet_node_impl.cpp
    parknet_node.cpp
    trt_yolo3_detector.cpp
    yolo3_detector.cpp)

# -Wunused-local-typedef is caused by glog of older version
set_source_files_properties(parknet_node_impl.cpp PROPERTIES COMPILE_FLAGS -Wno-unused-local-typedef)

set(PARKNET_INCLUDE_DIRS
    ${catkin_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/darknet/include
    ${PROJECT_SOURCE_DIR}/darknet/src
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src
    ${car_model_BINARY_DIR}/include
    ${camera_utils_SOURCE_DIR}/include
    ${GLOG_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
    ${PROJECT_BINARY_DIR}/include)

set(PARKNET_DETECTOR_SRCS main.cpp)

# set up dependencies for pslot_detector
set(PARKNET_LIB_DEPS
    ${OpenCV_LIBS}
    ${PCL_IO_LIBRARY}
    ${PCL_COMMON_LIBRARY}
    camera_utils
    m pthread stdc++
    ${GFLAGS_LIBRARIES}
    ${GLOG_LIBRARIES}
    ${catkin_LIBRARIES})

if (USE_DARKNET)
  list(APPEND PARKNET_LIB_DEPS darknet)
endif ()

if (USE_CUDA)
    add_definitions("-DGPU")
    list(APPEND PARKNET_LIB_DEPS
                ${CUDA_CUDART_LIBRARY}
                ${CUDA_cublas_LIBRARY}
                ${CUDA_curand_LIBRARY})
endif ()

if (USE_TENSORRT)
  list(APPEND PARKNET_LIB_DEPS
      yolo_tensorrt
      stdc++fs
      ${TensorRT_LIBRARIES}
      nvinfer_plugin)
  list(APPEND PARKNET_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/yolo_tensorrt)
endif ()

include_directories(${PARKNET_INCLUDE_DIRS})

if (USE_CUDA)
    cuda_add_library(${PARKNET_LIB} STATIC ${PARKNET_LIB_SRCS})
    cuda_add_executable(${PARKNET_EXE} ${PARKNET_DETECTOR_SRCS})
    target_link_libraries(${PARKNET_LIB} ${PARKNET_LIB_DEPS})
    target_link_libraries(${PARKNET_EXE} ${PARKNET_LIB})
else ()
    add_library(${PARKNET_LIB} STATIC ${PARKNET_LIB_SRCS})
    add_executable(${PARKNET_EXE} ${PARKNET_DETECTOR_SRCS})
    target_link_libraries(${PARKNET_LIB} ${PARKNET_LIB_DEPS})
    target_link_libraries(${PARKNET_EXE} ${PARKNET_LIB})
endif ()
add_dependencies(${PARKNET_LIB} ${catkin_EXPORTED_TARGETS})
add_dependencies(${PARKNET_EXE} ${PARKNET_LIB} ${catkin_EXPORTED_TARGETS})
add_subdirectory(alignment)
add_subdirectory(test)
