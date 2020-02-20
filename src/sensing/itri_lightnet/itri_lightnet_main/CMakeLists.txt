cmake_minimum_required(VERSION 2.8.3)
project(lightnet)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11 -O3)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  image_transport
  roscpp
  std_msgs
  msgs
  std_msgs
)

## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)
find_package(CUDA REQUIRED)
if (NOT ${CUDA_FOUND})
  message("OpenCV not found. Skip building ${PROJECT_NAME}")
  return ()
endif ()

if (CUDA_FOUND)
  message(STATUS "CUDA Include Dir: ${CUDA_INCLUDE_DIRS}")
  message(STATUS "CUDA Libararies: ${CUDA_LIBRARIES}")
  set(CUDA_LIBRARY_DIRS /usr/local/cuda/lib64) 
  message(STATUS "CUDA Libararies Dir: ${CUDA_LIBRARY_DIRS}")
endif()

find_package(OpenCV REQUIRED PATHS /usr/local NO_DEFAULT_PATH)
if (NOT ${OpenCV_FOUND})
  message("OpenCV not found. Skip building ${PROJECT_NAME}")
  return ()
endif ()
if(OpenCV_FOUND)
  message(STATUS "OpenCV IncludeDir: ${OpenCV_INCLUDE_DIRS}")
  message(STATUS "OpenCV Libararies: ${OpenCV_LIBRARIES}")
  message(STATUS "OpenCV Libararies Dir: ${OpenCV_LIBRARY_DIRS}")
endif()

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if your package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES signalnet
   CATKIN_DEPENDS cv_bridge image_transport roscpp std_msgs msgs itri_msgs
#  DEPENDS system_lib
)

###########
## Build ##
###########

#add wrapper
set(TRT_WRAPPER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/tensorRTWrapper/code)
add_subdirectory(${TRT_WRAPPER_DIR})
include_directories(${TRT_WRAPPER_DIR}/include)

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
# include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}/src/include
  ${CMAKE_CURRENT_SOURCE_DIR}/src/Framework
  ${CMAKE_CURRENT_SOURCE_DIR}/src/RosImagePubSub
  ${CMAKE_CURRENT_SOURCE_DIR}/src/itri_campus
)

#add jenro
#set(JENRO_SOURCES
 #   src/source_code/StepLadder.cpp 
#    src/source_code/NetPro.cpp  
 #   src/source_code/SignsPro.cpp 
 #   src/source_code/MainPro.cpp 
 #   src/source_code/RectTrack.cpp 
 #   src/source_code/LightPro.cpp 
  #  src/source_code/MyHaar.cpp 
 #   src/source_code/HAR_Car.cpp
  #  src/source_code/DistSignalTool.cpp 
 #   src/source_code/CamPro.cpp
 #   src/ITRISignalNet.cpp
 #   )
    
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/source_code)    

set(CPP_SOURCES
    src/main.cpp
    src/LightNet.cpp
    src/itri_campus/functions.cpp
#    src/Framework/CameraPerception.cpp
#    src/Framework/PerceptionOutput.cpp
 #   src/RosImagePubSub/RosImagePubSub.cpp
 #   src/SignalNetMain.cpp
#    ${JENRO_SOURCES}
    )
    
add_executable(${PROJECT_NAME}_node ${CPP_SOURCES})
target_link_libraries(${PROJECT_NAME}_node ${catkin_LIBRARIES} TrtNet ${OpenCV_LIBS} ${LEPTONICA_LIBRARIES} )
add_dependencies(${PROJECT_NAME}_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
