cmake_minimum_required(VERSION 2.8.3 FATAL_ERROR)
project(libs)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  roslib
  #std_msgs
  #genmsg
  grid_map_ros
  car_model
)

set(CMAKE_MODULE_PATH ${Project_SOURCE_DIR}/cmake ${car_model_DIR})
include(CompilerFlags)
include(CarModel)

## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)

## Uncomment this if the package has a setup.py. This macro ensures
## modules and global scripts declared therein get installed
## See http://ros.org/doc/api/catkin/html/user_guide/setup_dot_py.html
# catkin_python_setup()

################################################
## Declare ROS messages, services and actions ##
################################################

## To declare and build messages, services or actions from within this
## package, follow these steps:
## * Let MSG_DEP_SET be the set of packages whose message types you use in
##   your messages/services/actions (e.g. std_msgs, actionlib_msgs, ...).
## * In the file package.xml:
##   * add a build_depend tag for "message_generation"
##   * add a build_depend and a run_depend tag for each package in MSG_DEP_SET
##   * If MSG_DEP_SET isn't empty the following dependency has been pulled in
##     but can be declared for certainty nonetheless:
##     * add a run_depend tag for "message_runtime"
## * In this file (CMakeLists.txt):
##   * add "message_generation" and every package in MSG_DEP_SET to
##     find_package(catkin REQUIRED COMPONENTS ...)
##   * add "message_runtime" and every package in MSG_DEP_SET to
##   * uncomment the add_*_files sections below as needed
##     and list every .msg/.srv/.action file to be processed
##   * uncomment the generate_messages entry below
##   * add every package in MSG_DEP_SET to generate_messages(DEPENDENCIES ...)

## Generate messages in the 'msg' folder
#file(GLOB SOURCES_MESSAGE *.msg)
#add_message_files(
  #${SOURCES_MESSAGE}
#)

## Generate services in the 'srv' folder
#add_service_files(
  #FILES
  #AddTwoInts.srv
#)

## Generate actions in the 'action' folder
#add_action_files(
  #FILES
  #Action1.action
  #Action2.action
#)

## Generate added messages and services with any dependencies listed here
#generate_messages(
  #DEPENDENCIES
  #std_msgs
#)

################################################
## Declare ROS dynamic reconfigure parameters ##
################################################

## To declare and build dynamic reconfigure parameters within this
## package, follow these steps:
## * In the file package.xml:
##   * add a build_depend and a run_depend tag for "dynamic_reconfigure"
## * In this file (CMakeLists.txt):
##   * add "dynamic_reconfigure" to
##     find_package(catkin REQUIRED COMPONENTS ...)
##   * uncomment the "generate_dynamic_reconfigure_options" section below
##     and list every .cfg file to be processed

## Generate dynamic reconfigure parameters in the 'cfg' folder
# generate_dynamic_reconfigure_options(
  #cfg/DynReconf1.cfg
  #cfg/DynReconf2.cfg
# )

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if you package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
MACRO(HEADER_DIRECTORIES return_list)
    FILE(GLOB_RECURSE new_list *.h *.hpp)
    SET(dir_list "")
    FOREACH(file_path ${new_list})
        GET_FILENAME_COMPONENT(dir_path ${file_path} PATH)
        SET(dir_list ${dir_list} ${dir_path})
    ENDFOREACH()
    LIST(REMOVE_DUPLICATES dir_list)
    SET(${return_list} ${dir_list})
ENDMACRO()

HEADER_DIRECTORIES(SOURCES_INCS)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations

find_package(Eigen3)
include_directories(
  include
  ${EIGEN3_INCLUDE_DIR}
  ${catkin_INCLUDE_DIRS}
)

find_package(Glog)

## Declare a C++ library
#add_library(${PROJECT_NAME}
  #src/${PROJECT_NAME}/beginner_tutorials.cpp
#)

## Add cmake target dependencies of the library
## as an example, code may need to be generated before libraries
## either from message generation or dynamic reconfigure
#add_dependencies(${PROJECT_NAME} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Declare a C++ executable
## With catkin_make all packages are built within a single CMake context
## The recommended prefix ensures that target names across packages don't collide
#add_executable(${PROJECT_NAME}_node src/beginner_tutorials_node.cpp)

## Rename C++ executable without prefix
## The above recommended prefix causes long target names, the following renames the
## target back to the shorter version for ease of user use
## e.g. "rosrun someones_pkg node" instead of "rosrun someones_pkg someones_pkg_node"
#set_target_properties(${PROJECT_NAME}_node PROPERTIES OUTPUT_NAME node PREFIX "")

## Add cmake target dependencies of the executable
## same as for the library above
#add_dependencies(${PROJECT_NAME}_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Specify libraries to link a library or executable target against
#target_link_libraries(${PROJECT_NAME}_node
  #${catkin_LIBRARIES}
#)

#############
## Install ##
#############

# all install targets should use catkin DESTINATION variables
# See http://ros.org/doc/api/catkin/html/adv_user_guide/variables.html

## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination
# install(PROGRAMS
#   scripts/my_python_script
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark executables and/or libraries for installation
# install(TARGETS ${PROJECT_NAME} ${PROJECT_NAME}_node
#   ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark cpp header files for installation
# install(DIRECTORY include/${PROJECT_NAME}/
#   DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
#   FILES_MATCHING PATTERN "*.h"
#   PATTERN ".svn" EXCLUDE
# )

## Mark other files for installation (e.g. launch and bag files, etc.)
# install(FILES
#   # myfile1
#   # myfile2
#   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
# )

#############
## Testing ##
#############

## Add gtest based cpp test target and link libraries
# catkin_add_gtest(${PROJECT_NAME}-test test/test_beginner_tutorials.cpp)
# if(TARGET ${PROJECT_NAME}-test)
#   target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
# endif()

## Add folders to be run by python nosetests
# catkin_add_nosetests(test)

######################
##    use define    ##
######################

set (CMAKE_CXX_FLAGS   "${CMAKE_CXX_FLAGS} -D_FORCE_INLINES")

######################
##       VTK        ##
######################
find_package        (VTK REQUIRED)

######################
##       PCL        ##
######################
find_package            (PCL 1.9.1 REQUIRED)
include_directories     (${PCL_INCLUDE_DIRS})
link_directories        (${PCL_LIBRARY_DIRS})
add_definitions         (${PCL_DEFINITIONS})

######################
##       CUDA       ##
######################
find_package	        (CUDA)
if (NOT ${CUDA_FOUND})
  message("CUDA not found. Skip building ${PROJECT_NAME}")
  return ()
endif ()
set_directory_properties( PROPERTIES COMPILE_DEFINITIONS "" )
include_directories     (${CUDA_INCLUDE_DIRS})
message("CUDA found: ${CUDA_FOUND} CUDA version: ${CUDA_VERSION_STRING}")

######################
##       QT         ##
######################
find_package            (Qt5 REQUIRED COMPONENTS Widgets Core)
set                     (QT_HEADERS src/UI/QtViewer.h)
set                     (QT_FORMS   src/UI/QtViewer.ui)
QT5_WRAP_CPP            (QT_HEADERS_MOC   ${QT_HEADERS})
QT5_WRAP_UI             (QT_FORMS_HEADERS ${QT_FORMS})
ADD_DEFINITIONS         (${QT_DEFINITIONS})

######################
##      OpenCV      ##
######################
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})

######################
##      Tensorflow  ##
######################
if (NOT EXISTS /usr/local/lib/libtensorflow.so)

        set(TENSORFLOW_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../dl_data/assets)
        set(TENSORFLOW_INCLUDE_DIRS ${TENSORFLOW_ROOT_DIR}/include/tensorflow/c
                                    ${TENSORFLOW_ROOT_DIR}/include)
        set(TENSORFLOW_LIBRARIES ${TENSORFLOW_ROOT_DIR}/lib/libtensorflow.so)

	include_directories(${TENSORFLOW_INCLUDE_DIRS})
	link_directories(${TENSORFLOW_ROOT_DIR}/lib)

else()

	message( STATUS " Tensorflow status: installed" )	
	
	set(TENSORFLOW_INCLUDE_DIRS /usr/local/include/tensorflow/c)
	set(TENSORFLOW_LIBRARIES tensorflow)

	include_directories(${TENSORFLOW_INCLUDE_DIRS})
	link_directories(/usr/local/lib)

endif()

######################
##      catkin      ##
######################
catkin_package(
  INCLUDE_DIRS
  ${SOURCES_INCS}
  ${TENSORFLOW_INCLUDE_DIRS}

  #LIBRARIES beginner_tutorials
  #LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS roscpp rospy
  #std_msgs
  #DEPENDS system_lib
  #message_runtime
  LIBRARIES ${PROJECT_NAME}
)

######################
##      libs        ##
######################
set(SOURCES_SRCS
  BoundingBox/ApproxMVBB/Core/AABB.cpp
  BoundingBox/ApproxMVBB/Core/Common/MyMatrixTypeDefs.cpp
  BoundingBox/ApproxMVBB/Core/ConvexHull2D.cpp
  BoundingBox/ApproxMVBB/Core/Diameter/EstimateDiameter.cpp
  BoundingBox/ApproxMVBB/Core/Diameter/alloc.cpp
  BoundingBox/ApproxMVBB/Core/Diameter/rand.cpp
  BoundingBox/ApproxMVBB/Core/Diameter/util.cpp
  BoundingBox/ApproxMVBB/Core/GeometryPredicates/Predicates.cpp
  BoundingBox/ApproxMVBB/Core/GeometryPredicates/PredicatesInit.cpp
  BoundingBox/ApproxMVBB/Core/MinAreaRectangle.cpp
  BoundingBox/ApproxMVBB/Core/OOBB.cpp
  BoundingBox/ApproxMVBB/Core/ProjectedPointSet.cpp
  BoundingBox/ApproxMVBB/Core/RandomGenerators.cpp
  BoundingBox/ApproxMVBB/UseApproxMVBB.cpp
  BoundingBox/LShape/LShape.cpp
  BoundingBox/PCA/PCA.cpp
  Clustering/DBSCAN/CUDA/DBSCAN_CUDA.cpp
  Clustering/DBSCAN/CUDA/g_dbscan.cpp
  Clustering/DBSCAN/Common/DBSCAN.cpp
  Clustering/DBSCAN/VPTree/DBSCAN_VPTree.cpp
  Matching/NDT/LiDARStitchingAuto.cpp
  Matching/NDT/ndt_omp/gicp_omp.cpp
  Matching/NDT/ndt_omp/ndt_omp.cpp
  Matching/NDT/ndt_omp/voxel_grid_covariance_omp.cpp
  Preprocess/GroundFilter/PlaneGroundFilter.cpp
  Preprocess/GroundFilter/RayGroundFilter.cpp
  Preprocess/NoiseFilter/NoiseFilter.cpp
  Preprocess/NoiseFilter/VoxelFilter_CUDA.cpp
  Preprocess/SpatialFilter/CuboidFilter.cpp
  Preprocess/Transform/Transform_CUDA.cpp
  Preprocess/VoxelGrid/VoxelGrid_CUDA.cpp
  SSN_prepro/preprolib_squseg.cpp
  TensorFlow/tf_utils.cpp
  ToControl/EdgeDetect/edge_detect.cpp
  ToControl/GridMapGen/points_to_costmap.cpp
  TrafficFlow/S4FlowDetect.cpp
  Transmission/UdpModuleA.cpp
  Utility/CompressFunction.cpp
  Utility/KeyboardMouseEvent.cpp
  Utility/extract_Indices.cpp
  eval_utility/eval_utility.cpp
  shape_estimation/shape_estimator.cpp
  shape_estimation/model/bounding_box.cpp
  shape_estimation/filter/car_filter.cpp
  shape_estimation/filter/bus_filter.cpp
  shape_estimation/filter/truck_filter.cpp
  shape_estimation/filter/motor_filter.cpp
  shape_estimation/filter/no_filter.cpp
  shape_estimation/filter/pedestrian_filter.cpp
  shape_estimation/corrector/car_corrector.cpp
  shape_estimation/corrector/bus_corrector.cpp
  shape_estimation/corrector/truck_corrector.cpp
  shape_estimation/corrector/motor_corrector.cpp
  shape_estimation/corrector/pedestrian_corrector.cpp
  shape_estimation/corrector/no_corrector.cpp
  Clustering/DBSCAN/CUDA/g_dbscan.cu
  Preprocess/Transform/Transform_CUDA.cu
  Preprocess/VoxelGrid/VoxelGrid_CUDA.cu
  Format/pointcloud_format_conversion.cpp
  )


cuda_add_library(${PROJECT_NAME} ${SOURCES_SRCS})
target_link_libraries (${PROJECT_NAME} ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} ${PCL_LIBRARIES} ${CUDA_LIBRARIES} ${TENSORFLOW_LIBRARIES})
add_subdirectory(test)
