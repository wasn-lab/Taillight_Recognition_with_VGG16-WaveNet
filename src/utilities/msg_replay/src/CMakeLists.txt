set(MSG_REPLAY_SRCS
    args_parser.cpp
    msg_replay_node.cpp
    main.cpp)

set(MSG_REPLAY_INCLUDE_DIRS
    ${PROJECT_SOURCE_DIR}/src
    ${catkin_INCLUDE_DIRS}
    ${GLOG_INCLUDE_DIRS})

# set up dependencies for pslot_detector
set(MSG_REPLAY_DEPS
    ${GFLAGS_LIBRARIES}
    ${GLOG_LIBRARIES}
    ${catkin_LIBRARIES})

include_directories(${MSG_REPLAY_INCLUDE_DIRS})

set(MSG_REPLAY_EXE msg_replay_node)
add_executable(${MSG_REPLAY_EXE} ${MSG_REPLAY_SRCS})
target_link_libraries(${MSG_REPLAY_EXE} ${MSG_REPLAY_DEPS})
add_dependencies(${MSG_REPLAY_EXE} ${catkin_EXPORTED_TARGETS})
