#ifndef ALL_HEADER_H_
#define ALL_HEADER_H_

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <stdexcept>
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#include <omp.h>
#include <thread>
#include <unistd.h> //sleep
#include <functional>
#include <cerrno>
#include <cstdlib>
#include <condition_variable>

#include <ros/ros.h>
#include <std_msgs/Header.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <pcl/console/parse.h>
#include <pcl/console/print.h>
#include <pcl/console/time.h>

#include <pcl/io/pcd_io.h>
#include <pcl/io/boost.h>

#include <pcl/common/centroid.h>
#include <pcl/common/time.h>

#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/point_cloud_color_handlers.h>

#include <pcl/point_types.h>
#include <pcl/conversions.h>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/point_cloud.h>

#include <boost/asio.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/property_tree/ini_parser.hpp>

#include <QtCore>
#include <QApplication>
#include <QMainWindow>
#include <vtkRenderWindow.h>

#include <cuda.h>
#include <cuda_runtime.h>

using namespace std;
using namespace pcl;
using namespace pcl::console;
using namespace pcl::visualization;
using namespace boost::property_tree;

// for Debug 
#define ENABLE_DEBUG_MODE false

#endif /* ALL_HEADER_H_ */
