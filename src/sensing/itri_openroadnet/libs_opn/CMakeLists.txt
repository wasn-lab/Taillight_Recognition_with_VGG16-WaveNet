cmake_minimum_required(VERSION 2.8.3 FATAL_ERROR)
project(libs_opn)
  
## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11) 

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  roslib
  #std_msgs
  #genmsg
)                
                                                                                      
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

include_directories(
 include
  ${catkin_INCLUDE_DIRS}
)



######################
##    use define    ##
######################

set (CMAKE_VERBOSE_MAKEFILE ON)
set (CMAKE_BUILD_TYPE Release)
set (CMAKE_CXX_FLAGS   "${CMAKE_CXX_FLAGS} -fopenmp -D_FORCE_INLINES -O2")
set (CMAKE_CXX_FLAGS   "${CMAKE_CXX_FLAGS} -DOTHER_STUFF=1 " )


######################
##       CUDA       ##
######################
find_package	        (CUDA REQUIRED) 
set_directory_properties( PROPERTIES COMPILE_DEFINITIONS "" )
include_directories     (${CUDA_INCLUDE_DIRS}) 
set(CUDA_LIBS ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libnppisu.so
                ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libnppig.so
                ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libnppc.so
                ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libnppial.so
                ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libnppicc.so
                ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libnppicom.so
                ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libnppidei.so
                ${CUDA_cublas_LIBRARY}
                ${CUDA_curand_LIBRARY})
message("CUDA found: ${CUDA_FOUND} CUDA version: ${CUDA_VERSION_STRING}")

  #find_cuda_helper_libs(nppc)
  #find_cuda_helper_libs(nppial)
  #find_cuda_helper_libs(nppicc)
  #find_cuda_helper_libs(nppicom)
  #find_cuda_helper_libs(nppidei)
  #find_cuda_helper_libs(nppif)
  #find_cuda_helper_libs(nppig)
  #find_cuda_helper_libs(nppim)
  #find_cuda_helper_libs(nppist)
  #find_cuda_helper_libs(nppisu)
  #find_cuda_helper_libs(nppitc)
  #find_cuda_helper_libs(npps)

    
 
######################
##      OpenCV      ##
######################
find_package(OpenCV 3 REQUIRED PATHS /usr/local NO_DEFAULT_PATH) 
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})
 
######################
##        ##
###################### 
######################
##        ##
###################### 
set(PATH_SO_LOCAL ${CMAKE_CURRENT_SOURCE_DIR}/TensorFlow/Installed)
EXECUTE_PROCESS( COMMAND uname -m COMMAND tr -d '\n' OUTPUT_VARIABLE ARCHITECTURE )
message( STATUS " Your Platform Architecture: ${ARCHITECTURE}" )

if (NOT EXISTS /usr/local/lib/libtensorflow.so)

        if (NOT EXISTS ${PATH_SO_LOCAL}/lib/libtensorflow.so )

        message( STATUS " Tensorflow status: not installed, let me help you, wait wait wait downloading (100MB)" )
        make_directory(${PATH_SO_LOCAL})

        if( ${ARCHITECTURE} STREQUAL "x86_64" )

                file(
                DOWNLOAD http://nas.itriadv.co:8888/Share/ADV/Hino_git_data/ForLiDAR/libtensorflow-gpu-linux-x86_64-1.13.1.tar.gz
                        ${PATH_SO_LOCAL}/tensorflow.tar.gz
                STATUS
                    status
SHOW_PROGRESS
                LOG 
                    log
                )

                elseif(${ARCHITECTURE} STREQUAL "aarch64")

                file(
                DOWNLOAD http://nas.itriadv.co:8888/Share/ADV/Hino_git_data/ForLiDAR/libtensorflow-gpu-linux-aarch64-1.13.1.tar.gz
                         ${PATH_SO_LOCAL}/tensorflow.tar.gz
#                    http://118.163.54.109:8080/share.cgi?ssid=07319qW&fid=07319qW&filename=libtensorflow-gpu-linux-aarch64-1.13.1.tar.gz&openfolder=forcedownload&ep= ${PATH_SO_LOCAL}/tensorflow.tar.gz
                STATUS
                    status
SHOW_PROGRESS
                LOG 
                    log
                )

                endif()

                list(GET status 0 status_code)
        list(GET status 1 status_string)

        if(NOT status_code EQUAL 0)
                message(FATAL_ERROR "Error downloading tensorflow lib: ${status_string}" "${log}")
        else()
                execute_process(COMMAND ${CMAKE_COMMAND} -E tar xvzf ${PATH_SO_LOCAL}/tensorflow.tar.gz WORKING_DIRECTORY ${PATH_SO_LOCAL})
                file(REMOVE ${PATH_SO_LOCAL}/tensorflow.tar.gz)
        endif()

    endif()

    message( STATUS " Tensorflow status: installed" )

        include_directories(${PATH_SO_LOCAL}/include/tensorflow/c)
        link_directories(${PATH_SO_LOCAL}/lib)
set(TENSORFLOW_INCLUDES ${PATH_SO_LOCAL}/include/tensorflow/c)
    set(TENSORFLOW_LIBRARIES ${PATH_SO_LOCAL}/lib/libtensorflow.so)

else()

        message( STATUS " Tensorflow status: installed" )

        include_directories(/usr/local/include/tensorflow/c)
        link_directories(/usr/local/lib)

        set(TENSORFLOW_INCLUDES /usr/local/include/tensorflow/c)
        set(TENSORFLOW_LIBRARIES tensorflow)

endif()


catkin_package(
  INCLUDE_DIRS ${SOURCES_INCS}
  ${TENSORFLOW_INCLUDES}
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
file(GLOB_RECURSE SOURCES_SRCS *.cpp *.cu)            
cuda_add_library(${PROJECT_NAME} ${SOURCES_SRCS})
target_link_libraries (${PROJECT_NAME} ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} ${CUDA_LIBRARIES} cuda ${TENSORFLOW_LIBRARIES} ${CUDA_LIBS})

