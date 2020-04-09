cmake_minimum_required(VERSION 3.1)
project(hungarian)

set(CMAKE_MODULE_PATH ${Project_SOURCE_DIR}/cmake)
include(CompilerFlags)

find_package(catkin REQUIRED COMPONENTS)

catkin_package(CATKIN_DEPENDS)

set(HUNGARIAN_SOURCES src/hungarian.cpp)
set(HUNGARIAN_INCLUDE_DIRS ${catkin_INCLUDE_DIRS} src)

include_directories(${HUNGARIAN_INCLUDE_DIRS})
add_library(${PROJECT_NAME} STATIC ${HUNGARIAN_SOURCES})
target_link_libraries(${PROJECT_NAME})
