cmake_minimum_required(VERSION 3.1)
project(car_model)
include(cmake/CarModel.cmake)

catkin_package(
  CATKIN_DEPENDS roscpp std_msgs
)

configure_file(
    include/car_model.h.in
    ${PROJECT_BINARY_DIR}/include/car_model.h)
include_directories(${PROJECT_BINARY_DIR}/include)
catkin_add_gtest(
    car_model_test
    test/car_model_test.cpp)
