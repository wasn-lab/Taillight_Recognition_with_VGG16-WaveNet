cmake_minimum_required(VERSION 2.8.3)
project(camera_calibration)
 
set(CMAKE_MODULE_PATH ${Project_SOURCE_DIR}/cmake ${car_model_DIR})
include(CompilerFlags)
include(CarModel)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  roslib
  image_transport
  cv_bridge
  car_model
  camera_utils
)  

catkin_package(
  CATKIN_DEPENDS roscpp rospy
)

find_package(OpenCV REQUIRED PATHS /usr/local NO_DEFAULT_PATH)
if (NOT ${OpenCV_FOUND})
  message("OpenCV not found. Skip building ${PROJECT_NAME}")
  return ()
else()
  #message(STATUS "OpenCV IncludeDir: ${OpenCV_INCLUDE_DIRS}")
  #message(STATUS "OpenCV Libararies: ${OpenCV_LIBRARIES}")
  #message(STATUS "OpenCV Libararies Dir: ${OpenCV_LIBRARY_DIRS}")

  if(EXISTS /usr/local/lib/libopencv_core.so)
    set(OpenCV_USER_LIBRARIES /usr/local/lib/libopencv_core.so
                              ${OpenCV_LIBRARIES})
  else()
    set(OpenCV_USER_LIBRARIES ${OpenCV_LIBRARIES})
  endif()
endif ()

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${car_model_BINARY_DIR}/include
  ${camera_utils_SOURCE_DIR}/include
)

set(LIBS camera_utils
         ${OpenCV_USER_LIBRARIES}
)

if (CAR_MODEL_IS_C1)
  add_executable(${PROJECT_NAME}_node src/camera_calibration.cpp)
  target_link_libraries(${PROJECT_NAME}_node ${LIBS} ${catkin_LIBRARIES})
  add_dependencies(${PROJECT_NAME}_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
endif ()
