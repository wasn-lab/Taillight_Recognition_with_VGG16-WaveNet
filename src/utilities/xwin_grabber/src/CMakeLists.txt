include_directories(
  ${PROJECT_SOURCE_DIR}/src
  ${X11_INCLUDE_DIR}
  ${GLOG_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS})

add_executable(xwin_grabber_node
  xwin_grabber_main.cpp
  xwin_grabber_node.cpp
  xwin_grabber_utils.cpp
  xwin_grabber.cpp
  xwin_grabber_args_parser.cpp)

target_link_libraries(xwin_grabber_node
  ${GFLAGS_LIBRARIES}
  ${GLOG_LIBRARIES}
  ${X11_X11_LIB}
  ${X11_Xcomposite_LIB}
  ${X11_Xext_LIB}
  ${OpenCV_LIBS}
  ${catkin_LIBRARIES})
