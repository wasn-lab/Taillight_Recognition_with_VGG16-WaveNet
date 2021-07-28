set(IMAGE_FLIP_NODE_SRCS
    image_flip_args_parser.cpp
    image_flip_main.cpp
    image_flip_node.cpp)

set(IMAGE_FLIP_INCLUDE_DIRS
    ${PROJECT_SOURCE_DIR}/src
    ${catkin_INCLUDE_DIRS}
    ${GLOG_INCLUDE_DIRS})

# set up dependencies for pslot_detector
set(IMAGE_FLIP_NODE_DEPS
    m pthread stdc++
    ${GFLAGS_LIBRARIES}
    ${GLOG_LIBRARIES}
    ${OpenCV_LIBS}
    ${catkin_LIBRARIES})

include_directories(${IMAGE_FLIP_INCLUDE_DIRS})

set(IMAGE_FLIP_NODE_EXE image_flip_node)
add_executable(${IMAGE_FLIP_NODE_EXE} ${IMAGE_FLIP_NODE_SRCS})
target_link_libraries(${IMAGE_FLIP_NODE_EXE} ${IMAGE_FLIP_NODE_DEPS})
