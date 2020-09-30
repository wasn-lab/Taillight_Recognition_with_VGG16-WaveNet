cmake_minimum_required(VERSION 3.1)
project(dl_data)

find_package(catkin REQUIRED COMPONENTS car_model)

set(CMAKE_MODULE_PATH ${Project_SOURCE_DIR}/cmake ${car_model_DIR})
include(CarModel)


set(DRIVENET_WEIGHTS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../sensing/itri_drivenet/drivenet/data/yolo)
set(PARKNET_WEIGHTS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../sensing/itri_parknet/weights)
set(OPENROADNET_WEIGHTS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../sensing/itri_openroadnet/openroadnet)
set(LIDARNET_WEIGHTS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../sensing/itri_lidarnet/squeezesegnet/lidar_squseg_inference/model)
set(LIDARNET_V2_WEIGHTS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../sensing/itri_lidarnet/squeezesegnet/lidar_squseg_v2_inference/model)
set(CAMERA_UTILS_DATA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../sensing/camera_utils/data)
set(CAMERA_ALIGN_DATA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../sensing/itri_drivenet/drivenet/data/alignment)
set(CAMERA_GRABBER_DATA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../sensing/camera_grabber/src/CameraGrabber)
set(ASSETS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/assets)
set(PEDCROSS_MODELS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../sensing/itri_pedcross/models)
set(LIGHTNET_WEIGHTS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../sensing/itri_lightnet/itri_lightnet_main/resources)
set(LIGHTNET_TAINAN_WEIGHTS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../sensing/itri_lightnet_tainan/itri_lightnet_tainan_main/resources)
set(LOCALIZATION_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../sensing/localization)

function(download_file)
    cmake_parse_arguments(OPT "" "URL;DESTINATION;MD5" "" ${ARGN})

    if(NOT EXISTS "${OPT_DESTINATION}")
        set(need_download TRUE)
        message(STATUS "File ${OPT_DESTINATION} doesn't exist")
    else()
        file(MD5 "${OPT_DESTINATION}" file_md5)
        if(NOT "${file_md5}" STREQUAL "${OPT_MD5}")
            set(need_download TRUE)
            message(STATUS "File ${OPT_DESTINATION} is out of date")
        endif()
    endif()

    if(${need_download})
        message(STATUS "Downloading ${OPT_DESTINATION} ...")
        file(DOWNLOAD "${OPT_URL}" "${OPT_DESTINATION}")
        file(MD5 "${OPT_DESTINATION}" file_md5)
        if(NOT "${file_md5}" STREQUAL "${OPT_MD5}")
            message(WARNING "Expected hash of ${OPT_DESTINATION} is ${OPT_MD5}, "
                "but the actual hash is ${file_md5}")
        endif()
    else()
        message(STATUS "File ${OPT_DESTINATION} is up to date")
    endif()
endfunction()

#include(cmake/drivenet_weights.cmake)
#include(cmake/camera_distance.cmake)
include(cmake/camera_grabber.cmake)
#include(cmake/lidarnet_weights.cmake)
#include(cmake/libtensorflow_gpu.cmake)

#if (ENABLE_LIGHTNET)
#  include(cmake/lightnet_weights.cmake)
#  include(cmake/lightnet_tainan_weights.cmake)
#endif ()

#if (ENABLE_PEDCROSS)
#  include(cmake/pedcross_models.cmake)
#endif ()

#if (ENABLE_OPENROADNET)
#  include(cmake/openroadnet_weights.cmake)
#endif ()

#if (ENABLE_PARKNET)
#  include(cmake/parknet_weights.cmake)
#endif ()

#if (CAR_MODEL_IS_HINO)
#  include(cmake/camera_fov120_distance.cmake)
#endif ()

#include (cmake/localization.cmake)
