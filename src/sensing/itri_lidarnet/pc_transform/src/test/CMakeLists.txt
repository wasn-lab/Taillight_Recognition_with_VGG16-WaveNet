catkin_add_gtest(pc_transform_test
  pc_transform_test.cpp
  pc_transform_test_main.cpp)

target_link_libraries(pc_transform_test ${PC_TRANSFORM_LIB})
add_dependencies(pc_transform_test ${PC_TRANSFORM_LIB} ${catkin_EXPORTED_TARGETS})
