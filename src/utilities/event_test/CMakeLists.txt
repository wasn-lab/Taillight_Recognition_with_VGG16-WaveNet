cmake_minimum_required(VERSION 3.1)
project(event_test)

find_package(catkin REQUIRED COMPONENTS roscpp)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
)