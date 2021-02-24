if (NOT CATKIN_ENABLE_TESTING)
  return()
endif ()

if (NOT GLOG_FOUND)
  message("glog not found. The test items are skipped.")
  return()
endif ()

include_directories(
  ${libs_SOURCE_DIR}/pub
  ${libs_SOURCE_DIR}/Format
  ${GLOG_INCLUDE_DIR}
  )

catkin_add_gtest(lidar_test
  main.cpp
  gen_msg.cpp
  format_test.cpp)

target_link_libraries(lidar_test libs ${GLOG_LIBRARY})
add_dependencies(lidar_test ${catkin_EXPORTED_TARGETS} libs)

