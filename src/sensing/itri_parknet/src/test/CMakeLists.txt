set(PARKNET_TEST_JPG ${PARKNET_TEST_DATA_DIR}/parking_lot.jpg)

include_directories(${PARKNET_INCLUDE_DIRS})
set(PARKNET_TEST_SRCS
    main.cpp
    parknet_advertise_utils_test.cpp
    parknet_advertise_utils_test_fast.cpp
    parknet_fs_utils_test.cpp
    parknet_image_utils_test.cpp
    parknet_node_impl_test.cpp
    trt_yolo3_detector_test.cpp
    yolo3_detector_test.cpp)
catkin_add_gtest(parknet_test ${PARKNET_TEST_SRCS})

target_link_libraries(parknet_test ${PARKNET_LIB} ${PARKNET_DETECTOR_LIBS})
add_dependencies(parknet_test ${PARKNET_LIB} ${catkin_EXPORTED_TARGETS})
