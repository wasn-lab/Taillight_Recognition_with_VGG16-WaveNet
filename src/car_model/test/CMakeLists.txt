find_package(OpenCV)
find_package(PCL)
# PCL Variables: see PCLConfig.cmake (usu in /usr/local/share/pcl-1.9)
# Workaround for PCL bug: see https://github.com/PointCloudLibrary/pcl/issues/776
set_directory_properties(PROPERTIES COMPILE_DEFINITIONS "")

include_directories(
  ${PROJECT_BINARY_DIR}/include
  ${PCL_INCLUDE_DIRS}
  ${GLOG_INCLUDE_DIRS})
catkin_add_gtest(
    car_model_test
    car_model_test.cpp
    env_checker.cpp)
