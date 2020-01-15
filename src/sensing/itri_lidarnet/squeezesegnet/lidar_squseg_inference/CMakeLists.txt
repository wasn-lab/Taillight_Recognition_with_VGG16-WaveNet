cmake_minimum_required(VERSION 2.8.3)
project(lidar_squseg_inference)
add_compile_options(-std=c++11)

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

find_package(catkin REQUIRED COMPONENTS
  pcl_ros
  roscpp
  sensor_msgs
  std_msgs
  msgs
  libs
)
 
generate_messages(DEPENDENCIES std_msgs)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES squeezeseg_preprocessing
#  CATKIN_DEPENDS pcl_ros roscpp sensor_msgs
#  DEPENDS system_lib
   CATKIN_DEPENDS msgs
)

set (CMAKE_BUILD_TYPE Release) 
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -D_FORCE_INLINES")

######################
##       Model      ##
######################

set(PATH_MODEL_LOCAL ${CMAKE_CURRENT_SOURCE_DIR}/model)

function(DownloadModle Vehicletype ModelDegree)

    file(
    DOWNLOAD
    	#http://nas.itriadv.co:8070/Share/ADV/LiDAR_Models/SqueezeSegNet/hino1/N90deg.pb
		#ftp://user%40Intern1:aaaaaa@nas.itriadv.co/SqueezeSegNet/hino1/N90deg.pb
    	ftp://Intern1:aaaaaa@nas.itriadv.co:21/SqueezeSegNet/${Vehicletype}/${ModelDegree} ${PATH_MODEL_LOCAL}/SqueezeSegNet/${Vehicletype}/${ModelDegree}    	
    STATUS
        status
    LOG 
        log
    SHOW_PROGRESS
    )
     
    list(GET status 0 status_code)
    list(GET status 1 status_string)
    
    if(NOT status_code EQUAL 0)
    	message(FATAL_ERROR "Error downloading ${Vehicletype} ${ModelDegree} : ${status_string}" "${log}")	
	endif()
	
endfunction()

if (NOT EXISTS ${PATH_MODEL_LOCAL}/SqueezeSegNet/hino1/P0deg.pb)
  
    message( STATUS " you have no HINO model file, let Wayne help you, downloading (12MB)" )
    make_directory(${PATH_MODEL_LOCAL}/SqueezeSegNet/hino1)
     
	DownloadModle("hino1" "N90deg.pb")
	DownloadModle("hino1" "P0deg.pb")
	DownloadModle("hino1" "P90deg.pb")
	DownloadModle("hino1" "P180deg.pb")
	
endif()

if (NOT EXISTS ${PATH_MODEL_LOCAL}/SqueezeSegNet/b1/P0deg.pb)
  
    message( STATUS " you have no b1 model file, let Wayne help you, downloading (12MB)" )
    make_directory(${PATH_MODEL_LOCAL}/SqueezeSegNet/b1)
     
	DownloadModle("b1" "N90deg.pb")
	DownloadModle("b1" "P0deg.pb")
	DownloadModle("b1" "P90deg.pb")
	DownloadModle("b1" "P180deg.pb")

endif()

#    message(INFO "check the version of model files, please wait")
#    file(MD5 ${PATH_MODEL_LOCAL}/SqueezeSegNet/hino1/N90deg.pb CHECKSUMN90)
#    
#    file(
#    DOWNLOAD 
#        http://nas.itriadv.co:8888/Share/ADV/LiDAR_Models/SqueezeSegNet/hino1/N90deg.pb ${PATH_MODEL_LOCAL}/SqueezeSegNet/hino1/N90deg_1.pb
#    STATUS
#        status
#    LOG 
#        log
#    SHOW_PROGRESS
#    )
#    list(GET status 0 status_code)
#    list(GET status 1 status_string)
#    
#    file(MD5 ${PATH_MODEL_LOCAL}/SqueezeSegNet/hino1/N90deg_1.pb CHECKSUMN90ON)
#    if(NOT ${CHECKSUMN90} STREQUAL ${CHECKSUMN90ON})
#        message(INFO "update N90deg to newest version...")
#        file(REMOVE ${PATH_MODEL_LOCAL}/SqueezeSegNet/hino1/N90deg.pb)
#        file(RENAME ${PATH_MODEL_LOCAL}/SqueezeSegNet/hino1/N90deg_1.pb ${PATH_MODEL_LOCAL}/SqueezeSegNet/hino1/N90deg.pb)
#    endif()


######################
##       PCL        ##
######################
find_package            (PCL 1.8 REQUIRED) 
include_directories     (${PCL_INCLUDE_DIRS})
link_directories        (${PCL_LIBRARY_DIRS})
add_definitions         (${PCL_DEFINITIONS})

include_directories(
  ${catkin_INCLUDE_DIRS}
)


file(GLOB_RECURSE SOURCES_SRCS *.cpp *.cu)
 
add_executable(squeezeseg_inference ${SOURCES_SRCS})
add_dependencies(squeezeseg_inference ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(squeezeseg_inference ${catkin_LIBRARIES} ${PCL_LIBRARIES})
