cmake_minimum_required(VERSION 2.8.3)
project(opengl_test)
set(CMAKE_MODULE_PATH ${Project_SOURCE_DIR}/cmake)

set (CMAKE_CXX_STANDARD 14)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(msgs REQUIRED)
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  sensor_msgs
  message_generation
#
  msgs # ITRI ICLU3 msg package for customized message type
#
  pcl_conversions
  pcl_ros
#
  image_transport
  cv_bridge
)
find_package(OpenCV REQUIRED)
#
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
find_package(assimp)
if (NOT assimp_FOUND)
  message(FATAL_ERROR "${PACKAGE_NAME} requires libassimp. Try: sudo apt-get install libassimp-dev")
endif ()

find_package(Bullet)
if (NOT BULLET_FOUND)
  message(FATAL_ERROR "${PACKAGE_NAME} requires bullet package. Try: sudo apt-get install libbullet-dev")
endif ()

include_directories( ${OPENGL_INCLUDE_DIRS}  ${GLUT_INCLUDE_DIRS} )


# Compile external dependencies
add_subdirectory (external)



################################################
## Declare ROS messages, services and actions ##
################################################



file(GLOB SOURCES_MESSAGE *.msg)
add_message_files(
  ${SOURCES_MESSAGE}
)


generate_messages(
  DEPENDENCIES
  std_msgs
  # msgs
)




catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS
  DEPENDS Eigen OpenCV
)
catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES opengl_test
   CATKIN_DEPENDS roscpp rospy std_msgs message_runtime image_transport sensor_msgs cv_bridge message_runtime
#  DEPENDS system_lib
)








include_directories(
    # include
    ${catkin_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
    include/async_buffer/
    include/ROS_interface/
    include/ROS_interface/include/
    include/ROS_interface/include/buffer_wrappers/
    #
    include/FILTER_LIB/
    include/FILTER_LIB/IIR/
    .
)
include_directories(
    include/
    include/TinyOBJ/
    include/Common/
    include/ViewManager/
    include/Shader/
    include/GL2DShape/
    include/RenderingModels/
    include/Scenes/
    #
	external/AntTweakBar-1.16/include/
	# external/glfw-3.1.2/include/
	external/glm-0.9.7.1/
	external/glew-1.13.0/include/
	# external/assimp-3.0.1270/include/
	# external/bullet-2.81-rev2613/src/
	.
)







link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})



set(ALL_GL_LIBS
	${OPENGL_LIBRARY}
	# glfw
	GLEW_1130
    ${GLUT_LIBRARY}
)


# Render models
set(RENDER_MODEL_PATH
    include/RenderingModels/
)
set(ALL_RENDER_MODELS
    ${RENDER_MODEL_PATH}rmBaseModel.cpp
    ${RENDER_MODEL_PATH}rmModelLoaderObj.cpp
    ${RENDER_MODEL_PATH}rmPointCloud.cpp
    ${RENDER_MODEL_PATH}rmLidarBoundingBox.cpp
    ${RENDER_MODEL_PATH}rmImageBoard.cpp
    ${RENDER_MODEL_PATH}rmColorBoard.cpp
    ${RENDER_MODEL_PATH}rmText2D.cpp
    # ${RENDER_MODEL_PATH}rmText3D.cpp
    ${RENDER_MODEL_PATH}rmText3D_v2.cpp
    ${RENDER_MODEL_PATH}rmSweepingObject.cpp
    ${RENDER_MODEL_PATH}rmBoundingBox2D.cpp
    ${RENDER_MODEL_PATH}rmGrid.cpp
    ${RENDER_MODEL_PATH}rmPolyLines3D.cpp
    ${RENDER_MODEL_PATH}rmCircle.cpp
    ${RENDER_MODEL_PATH}rmlv2TagBoundingBox2D.cpp
    ${RENDER_MODEL_PATH}rmlv2TagBoundingBox3D.cpp
    ${RENDER_MODEL_PATH}rmlv2ObjectTracking.cpp
    ${RENDER_MODEL_PATH}rmlv2PathPlanFake.cpp
    ${RENDER_MODEL_PATH}rmlv2PathPlan_v1.cpp
    ${RENDER_MODEL_PATH}rmlv2SpeedMeter.cpp
    #
    ${RENDER_MODEL_PATH}rmlv2TrafficLight.cpp
    ${RENDER_MODEL_PATH}rmlv2TrafficLightSimple.cpp
    ${RENDER_MODEL_PATH}rmMagicPowder.cpp
    #
    ${RENDER_MODEL_PATH}rmImageArray.cpp
    ${RENDER_MODEL_PATH}rmlv2TrafficLightImage.cpp

)





# Libraries
# ROS interfaces
#---------------------------------------------------#
add_library (async_buffer_v2_1 STATIC include/async_buffer/async_buffer_v2.cpp)

# add_library (ROS_interface_1 STATIC include/ROS_interface/ROS_interface.cpp)
# target_link_libraries(ROS_interface_1 async_buffer_v2_1)

# add_library (ROS_interface_v2_1 STATIC include/ROS_interface/ROS_interface_v2.cpp)
# target_link_libraries(ROS_interface_v2_1 async_buffer_v2_1)

# add_library (ROS_interface_v3_1 STATIC include/ROS_interface/ROS_interface_v3.cpp)
# target_link_libraries(ROS_interface_v3_1 async_buffer_v2_1)

add_library (ROS_interface_v4_1 STATIC include/ROS_interface/ROS_interface_v4.cpp)
add_dependencies(ROS_interface_v4_1 msgs_generate_messages_cpp ${catkin_EXPORTED_TARGETS})
target_link_libraries(ROS_interface_v4_1 async_buffer_v2_1)
#---------------------------------------------------#

# Filters
add_library (FILTER_LIB STATIC include/FILTER_LIB/FILTER_LIB.cpp include/FILTER_LIB/CIRCULAR_BUFFER_VECTOR include/FILTER_LIB/IIR/IIR.cpp)


# OpenGL libraries
#---------------------------------------------------#
add_library (tiny_obj_loader STATIC include/TinyOBJ/tiny_obj_loader.cpp)
target_link_libraries(tiny_obj_loader ${ALL_GL_LIBS}  )

add_library (Common STATIC include/Common/Common.cpp)
target_link_libraries(Common ${ALL_GL_LIBS} tiny_obj_loader )

# add_library (ViewManager STATIC include/ViewManager/ViewManager.cpp)
# target_link_libraries(ViewManager ${ALL_GL_LIBS} Common )

add_library (ViewManager_v2 STATIC include/ViewManager/ViewManager_v2.cpp include/ViewManager/transformFilter.cpp)
target_link_libraries(ViewManager_v2 ${ALL_GL_LIBS} Common )

add_library (Shader STATIC include/Shader/Shader.cpp)
target_link_libraries(Shader ${ALL_GL_LIBS} Common )

add_library (GL2DShape STATIC include/GL2DShape/GL2DShape.cpp)
target_link_libraries(GL2DShape ${ALL_GL_LIBS} Common )
#---------------------------------------------------#
set(ALL_GL_UTIL_LIBS
    # ViewManager
    ViewManager_v2
    Shader
    GL2DShape
)

# test_1
# add_executable(test_1 src/7.5.Point_Sprite/test_1.cpp include/ROS_ICLU3_v0.cpp include/Scenes/Scene.cpp ${ALL_RENDER_MODELS})
# target_link_libraries(test_1 ROS_interface_v4_1 ${ALL_GL_LIBS} ${ALL_GL_UTIL_LIBS} ANTTWEAKBAR_116_OGLCORE_GLFW ${OpenCV_LIBS} ${catkin_LIBRARIES} FILTER_LIB )
# test_republish_image
add_executable(test_republish_image src/7.5.Point_Sprite/test_republish_image.cpp   )
target_link_libraries(test_republish_image  ${OpenCV_LIBS} ${catkin_LIBRARIES} )
# point_sprite
# add_executable(point_sprite src/7.5.Point_Sprite/Point_Sprite.cpp )
# target_link_libraries(point_sprite ROS_interface_v4_1  ${ALL_GL_LIBS} ${ALL_GL_UTIL_LIBS} ANTTWEAKBAR_116_OGLCORE_GLFW ${OpenCV_LIBS} ${catkin_LIBRARIES} FILTER_LIB )
# visualizer2
add_executable(visualizer2 src/visualizer2_main.cpp include/ROS_ICLU3_v0.cpp include/Scenes/Scene.cpp ${ALL_RENDER_MODELS})
target_link_libraries(visualizer2 ROS_interface_v4_1  ${ALL_GL_LIBS} ${ALL_GL_UTIL_LIBS} ANTTWEAKBAR_116_OGLCORE_GLFW ${OpenCV_LIBS} ${catkin_LIBRARIES} FILTER_LIB )
