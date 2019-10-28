set(CAMERA_UTILS_TEST_DATA_DIR ${CMAKE_CURRENT_LIST_DIR}/../test_data)

set(CAMERA_UTILS_TEST_SOURCES
    camera_utils_test.cpp
    camera_distance_mapper_test.cpp
    imu_distance_test.cpp
    npp_remap_test.cpp
    npp_rotate_test.cpp
    npp_resize_test.cpp
    npp_utils_test.cpp
    main.cpp)

configure_file(camera_utils_test.h.in ${PROJECT_BINARY_DIR}/include/camera_utils_test.h)
catkin_add_gtest(${PROJECT_NAME}_test ${CAMERA_UTILS_TEST_SOURCES})

target_link_libraries(${PROJECT_NAME}_test ${PROJECT_NAME})
add_dependencies(${PROJECT_NAME}_test ${PROJECT_NAME})
