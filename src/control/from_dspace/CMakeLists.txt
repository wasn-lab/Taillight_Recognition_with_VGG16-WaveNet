cmake_minimum_required(VERSION 2.8.3)
project(from_dspace)

set(CMAKE_MODULE_PATH ${Project_SOURCE_DIR}/cmake ${car_model_DIR})
include(CarModel)
include(CompilerFlags)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  message_generation
  msgs
  car_model
)

################################################
## Declare ROS messages, services and actions ##
################################################
## Generate added messages and services with any dependencies listed here
#generate_messages(
#  DEPENDENCIES
#  std_msgs
#  msgs
#)

###################################
## catkin specific configuration ##
###################################
catkin_package(CATKIN_DEPENDS
  roscpp 
  rospy 
  std_msgs 
  message_runtime 
  msgs
)

###########
## Build ##
###########
include_directories(${catkin_INCLUDE_DIRS})


#add_executable(GUI_Publisher01 src/GUI_Publisher01.cpp )
#target_link_libraries(GUI_Publisher01 ${catkin_LIBRARIES})
#add_dependencies(GUI_Publisher01 GUI_Publisher01_generate_messages_cpp)
#
#add_executable(GUI_Publisher02 src/GUI_Publisher02.cpp )
#target_link_libraries(GUI_Publisher02 ${catkin_LIBRARIES})
#add_dependencies(GUI_Publisher02 GUI_Publisher02_generate_messages_cpp)

#add_executable(GUI_Publisher03 src/GUI_Publisher03.cpp )
#target_link_libraries(GUI_Publisher03 ${catkin_LIBRARIES})
#add_dependencies(GUI_Publisher03 GUI_Publisher03_generate_messages_cpp)


add_executable(from_dspace src/from_dspace.cpp )
target_link_libraries(from_dspace ${catkin_LIBRARIES})
add_dependencies(from_dspace ${catkin_EXPORTED_TARGETS} )
#add_dependencies(gui_publisher_all ${${PROJECT_NAME}_EXPORTED_TARGETS})




