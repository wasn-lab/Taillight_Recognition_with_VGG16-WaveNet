set(DEEPLAB_LIB "deeplab")
set(DEEPLAB_EXE "deeplab_node")
set(TENSORFLOW_ROOT_DIR ${Project_SOURCE_DIR}/dl_data/assets)

set(DEEPLAB_LIB_SRCS
  tf_utils.cpp
  deeplab_args_parser.cpp
  deeplab_node.cpp
  deeplab_node_impl.cpp
  deeplab_segmenter.cpp
  deeplab_segmenter_impl.cpp)

set(DEEPLAB_INCLUDE_DIRS
    ${PROJECT_SOURCE_DIR}/src
    ${catkin_INCLUDE_DIRS}
    ${TENSORFLOW_ROOT_DIR}/include
    ${camera_utils_SOURCE_DIR}/include
    ${GLOG_INCLUDE_DIRS})

set(DEEPLAB_DETECTOR_SRCS main.cpp)

# set up dependencies for pslot_detector
set(DEEPLAB_LIB_DEPS
  /usr/local/lib/libopencv_core.so
  tensorflow
  camera_utils
  ${OpenCV_LIBS}
  ${GFLAGS_LIBRARIES}
  ${GLOG_LIBRARIES}
  ${catkin_LIBRARIES})

include_directories(${DEEPLAB_INCLUDE_DIRS})
link_directories(${TENSORFLOW_ROOT_DIR}/lib)

add_library(${DEEPLAB_LIB} EXCLUDE_FROM_ALL STATIC ${DEEPLAB_LIB_SRCS})
target_link_libraries(${DEEPLAB_LIB} ${DEEPLAB_LIB_DEPS})

add_executable(${DEEPLAB_EXE} EXCLUDE_FROM_ALL ${DEEPLAB_DETECTOR_SRCS})
target_link_libraries(${DEEPLAB_EXE} ${DEEPLAB_LIB})

set(DEEPLAB_CMD_SRCS deeplab_cmd.cpp)
add_executable(deeplab_cmd EXCLUDE_FROM_ALL ${DEEPLAB_CMD_SRCS})
target_link_libraries(deeplab_cmd ${DEEPLAB_LIB})
