cmake_minimum_required(VERSION 3.1)
project(car_model)

set(CAR_MODEL "B1" CACHE STRING "Car model (B1, B1V2 C, HINO)")
set_property(CACHE CAR_MODEL PROPERTY STRINGS B1 C B1V2 HINO)

include(cmake/CarModel.cmake)

catkin_package(
  CATKIN_DEPENDS roscpp std_msgs
  CFG_EXTRAS CarModel.cmake
)

configure_file(
    include/car_model.h.in
    ${PROJECT_BINARY_DIR}/include/car_model.h)

add_subdirectory(test)
if (CAR_MODEL_IS_B1)
  add_subdirectory(test_car_b1)
endif ()
