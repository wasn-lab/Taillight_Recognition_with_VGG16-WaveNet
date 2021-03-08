set(INCLUDE_DIRS
    ${catkin_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
    ${GLOG_INCLUDE_DIRS})

set(LIB_DEPS
    ${GFLAGS_LIBRARIES}
    ${GLOG_LIBRARIES}
    ${catkin_LIBRARIES}
    ${PCL_IO_LIBRARY}
    ${PCL_COMMON_LIBRARY})

include_directories(${INCLUDE_DIRS})
set(EXE pcd_to_pc2)
set(SOURCES pcd_to_pc2_main.cpp pcd_to_pc2_args_parser.cpp pcd_to_pc2_node.cpp)

add_executable(${EXE} EXCLUDE_FROM_ALL ${SOURCES})
target_link_libraries(${EXE} ${LIB_DEPS})
add_dependencies(${EXE} ${catkin_EXPORTED_TARGETS})
