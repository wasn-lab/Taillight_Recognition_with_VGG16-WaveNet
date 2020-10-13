cmake_minimum_required(VERSION 2.8.3)
project(camera_radar_alignment)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  message_generation
  genmsg
  image_transport
  sensor_msgs
  geometry_msgs
  cv_bridge
  msgs
)

set(CMAKE_MODULE_PATH ${Project_SOURCE_DIR}/cmake)
include(CompilerFlags)


set(rqt_image_view_INCLUDE_DIRECTORIES
  include
  "${CATKIN_DEVEL_PREFIX}/${CATKIN_GLOBAL_INCLUDE_DESTINATION}"
)

## Generate messages in the 'msg' folder
#add_message_files(
#  DIRECTORY msg
#)

## Generate added messages and services with any dependencies listed here
#generate_messages(
#  DEPENDENCIES
#  std_msgs
#)

#catkin_package(

#)

include_directories(
 include
  ${catkin_INCLUDE_DIRS}
)


find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

######################
##       VTK        ##
######################
find_package        (VTK 6.2 REQUIRED)

######################
##       PCL        ##
######################
find_package            (PCL 1.8 REQUIRED)
include_directories     (${PCL_INCLUDE_DIRS})
link_directories        (${PCL_LIBRARY_DIRS})
add_definitions         (${PCL_DEFINITIONS})

######################
##       QT         ##
######################
find_package            (Qt5 REQUIRED COMPONENTS Widgets Core)
set                     (QT_HEADERS src/UI/QtViewer.h)
set                     (QT_FORMS   src/UI/QtViewer.ui)
QT5_WRAP_CPP            (QT_HEADERS_MOC   ${QT_HEADERS})
QT5_WRAP_UI             (QT_FORMS_HEADERS ${QT_FORMS})
ADD_DEFINITIONS         (${QT_DEFINITIONS})

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

file(GLOB_RECURSE SOURCES_SRCS *.cpp)
add_executable(${PROJECT_NAME} src/subscriber.cpp ${QT_FORMS_HEADERS} ${QT_HEADERS_MOC} ${SOURCES_SRCS})
add_dependencies(${PROJECT_NAME} msgs_generate_messages_cpp ${catkin_EXPORTED_TARGETS})
target_link_libraries(${PROJECT_NAME} ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} ${Eigen3_LIBRARIES} )

qt5_use_modules(${PROJECT_NAME} Widgets)
