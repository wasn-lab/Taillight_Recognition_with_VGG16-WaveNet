cmake_minimum_required(VERSION 3.1)
project(car_model)

set(CAR_MODEL "B1" CACHE STRING "Car model (A, B1, C, HINO)")
set_property(CACHE CAR_MODEL PROPERTY STRINGS A B1 C HINO)

include(cmake/CarModel.cmake)

catkin_package(
  CATKIN_DEPENDS roscpp std_msgs
  CFG_EXTRAS CarModel.cmake
)

configure_file(
    include/car_model.h.in
    ${PROJECT_BINARY_DIR}/include/car_model.h)
include_directories(${PROJECT_BINARY_DIR}/include)
catkin_add_gtest(
    car_model_test
    test/car_model_test.cpp)
