set(FILE_SAVER_LIB "itri_file_saver")

set(FILE_SAVER_LIB_SRCS
    image_saver_node.cpp
    image_saver_node_impl.cpp
    pcd_saver_node.cpp
    pcd_saver_node_impl.cpp
    video_saver_node.cpp
    video_saver_node_impl.cpp)

set(IMAGE_SAVER_SRCS image_saver_args_parser.cpp image_saver_main.cpp)
set(VIDEO_SAVER_SRCS video_saver_args_parser.cpp video_saver_main.cpp)
set(PCD_SAVER_SRCS pcd_saver_args_parser.cpp pcd_saver_main.cpp)

set(FILE_SAVER_INCLUDE_DIRS
    ${catkin_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/include
    ${PCL_INCLUDE_DIRS}
    ${GLOG_INCLUDE_DIRS})

# set up dependencies for pslot_detector
set(FILE_SAVER_LIB_DEPS
    m pthread stdc++
    ${GFLAGS_LIBRARIES}
    ${GLOG_LIBRARIES}
    ${catkin_LIBRARIES}
    ${OpenCV_LIBS}
    ${PCL_IO_LIBRARY}
    ${PCL_COMMON_LIBRARY})

include_directories(${FILE_SAVER_INCLUDE_DIRS})

add_library(${FILE_SAVER_LIB} STATIC EXCLUDE_FROM_ALL ${FILE_SAVER_LIB_SRCS})
add_dependencies(${FILE_SAVER_LIB} ${catkin_EXPORTED_TARGETS})
target_link_libraries(${FILE_SAVER_LIB} ${FILE_SAVER_LIB_DEPS})

add_executable(image_saver EXCLUDE_FROM_ALL ${IMAGE_SAVER_SRCS})
target_link_libraries(image_saver ${FILE_SAVER_LIB})

add_executable(video_saver EXCLUDE_FROM_ALL ${VIDEO_SAVER_SRCS})
target_link_libraries(video_saver ${FILE_SAVER_LIB})

add_executable(pcd_saver EXCLUDE_FROM_ALL ${PCD_SAVER_SRCS})
target_link_libraries(pcd_saver ${FILE_SAVER_LIB})
