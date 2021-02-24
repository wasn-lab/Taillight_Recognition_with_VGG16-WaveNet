if (NOT CATKIN_ENABLE_TESTING)
  return()
endif ()

catkin_add_gtest(lidar_test
  main.cpp
  format_test.cpp)

target_link_libraries(lidar_test libs)
add_dependencies(lidar_test ${catkin_EXPORTED_TARGETS} libs)

