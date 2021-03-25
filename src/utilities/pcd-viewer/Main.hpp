#pragma once
#include <cstdio>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <utility>

// PCL
#include <pcl/common/transforms.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

// Also include GLFW to allow for graphical display
#define GLFW_INCLUDE_GLU
#include <GLFW/glfw3.h>

#include <xercesc/util/PlatformUtils.hpp>
#include <xercesc/parsers/XercesDOMParser.hpp>
#include <xercesc/dom/DOM.hpp>
#include <xercesc/sax/HandlerBase.hpp>
using namespace xercesc;

typedef struct
{
    std::string classId;
    pcl::PointCloud<pcl::PointXYZ>::Ptr corner;
} BoxInfo;

void loadPCDFile(std::string filename);
void loadXMLFile(std::string filename);
void loadTXTFile(std::string filename);
void displayPointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud);
void displayBoundingBox(BoxInfo &bbox);
