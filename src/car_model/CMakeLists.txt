cmake_minimum_required(VERSION 3.1)
project(car_model)

set(CAR_MODEL "B1_V2" CACHE STRING "Car model (A, B1, B1_V2, C, HINO)")
set_property(CACHE CAR_MODEL PROPERTY STRINGS A B1 B1_V2 C HINO)

set(CAR_MODEL_IS_B1 0)
set(CAR_MODEL_IS_B1_V2 0)
set(CAR_MODEL_IS_C 0)
set(CAR_MODEL_IS_HINO 0)

if ("${CAR_MODEL}" STREQUAL "B1")
  set(CAR_MODEL_IS_B1 1)
elseif ("${CAR_MODEL}" STREQUAL "B1_V2")
  set(CAR_MODEL_IS_B1_V2 1)
elseif ("${CAR_MODEL}" STREQUAL "C")
  set(CAR_MODEL_IS_C 1)
elseif ("${CAR_MODEL}" STREQUAL "HINO")
  set(CAR_MODEL_IS_HINO 1)
else ()
  message(FATAL_ERROR "Invalid car model: ${CAR_MODEL}")
endif ()

# default options
option(ENABLE_LIGHTNET "Enable traffic light detection" ON)
option(ENABLE_PARKNET "Enable parking slot detection" ON)
option(ENABLE_OPENROADNET "Enable open road detection" ON)
option(ENABLE_PEDCROSS "Enable pedestrian crossing detection" ON)
option(ENABLE_GOOGLE_BENCHMARK "Enable google benchmark for perf profiling" OFF)
include(cmake/Options${CAR_MODEL}.cmake)

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
