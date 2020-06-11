# /**
# MIT License

# Copyright (c) 2018 NVIDIA CORPORATION. All rights reserved.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# *
# */

if (COMPILER_IS_GNUCXX)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-function -Wno-unused-variable -Wno-unused-but-set-variable")
endif ()
file(GLOB CU_SRCS *.cu)

set(CXX_SRCS
    calibrator.cpp
    ds_image.cpp
    plugin_factory.cpp
    yoloplugin_lib.cpp
    yolov2.cpp
    yolo.cpp
    trt_utils.cpp
#    yolo_config_parser.cpp
    yolov3.cpp)

find_package(PkgConfig)
pkg_check_modules(GLIB REQUIRED glib-2.0)

include_directories(${CUDA_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${TensorRT_INCLUDE_DIRS} ${GLIB_INCLUDE_DIRS} ${GLFAGS_INCLUDE_DIRS})

add_definitions("-DTENSORRT_VERSION_MAJOR=${TensorRT_VERSION_MAJOR}")
if (COMPILER_IS_GNUCXX)
  set_source_files_properties(yoloplugin_lib.cpp PROPERTIES COMPILE_FLAGS -Wno-sign-compare)
  set_source_files_properties(yolo.cpp PROPERTIES COMPILE_FLAGS -Wno-sign-compare)
else ()
  # clang
  set_source_files_properties(yoloplugin_lib.cpp PROPERTIES COMPILE_FLAGS -Wno-sign-compare)
  set_source_files_properties(plugin_factory.cpp PROPERTIES COMPILE_FLAGS -Wno-unused-private-field)
  set_source_files_properties(calibrator.cpp PROPERTIES COMPILE_FLAGS -Wno-unused-private-field)
  set_source_files_properties(yolo.cpp PROPERTIES COMPILE_FLAGS -Wno-sign-compare)
endif ()

cuda_add_library(yolo_tensorrt STATIC ${CXX_SRCS} ${CU_SRCS})
target_link_libraries(yolo_tensorrt ${OpenCV_LIBS})
