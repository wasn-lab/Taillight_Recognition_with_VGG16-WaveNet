
#ifndef DEBUG_TOOL_H_
#define DEBUG_TOOL_H_

#include "all_header.h"
#include "KeyboardMouseEvent.h"
#include "GlobalVariable.h"
#include "CompressFunction.h"

void show_cloud(PointCloud<PointXYZI>::ConstPtr cloud);

void show_cloud(PointCloud<PointXYZ>::ConstPtr cloud);
void show_cloud(PointCloud<PointNormal>::ConstPtr cloud);
void show_cloud(PointCloud<PointXYZRGB>::ConstPtr cloud);

void log_3D(CLUSTER_INFO* cluster_info, int cluster_size);

void log_2D(CLUSTER_INFO* cluster_info, int cluster_size);

#endif
