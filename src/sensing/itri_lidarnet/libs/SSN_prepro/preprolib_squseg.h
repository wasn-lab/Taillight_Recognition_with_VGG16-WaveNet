#ifndef __PREPROLIB_SQUSEG__
#define __PREPROLIB_SQUSEG__

#include <pcl/io/pcd_io.h>
#include <pcl/console/print.h>
#include <pcl/console/parse.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <sys/types.h>
#include <dirent.h>
#include <errno.h>
#include <string>
#include "../UserDefine.h"

// #define imageWidth 512
#define imageHeight 64

using namespace std;


int getdir(string dir, vector<string> &filenames);
int ILcomb(string inputname_i, string inputname_l, VPointCloudXYZIL::Ptr cloud_il);
VPointCloudXYZIDL sph_proj(VPointCloudXYZIL::Ptr cloud_il, const float phi_center, const float phi_range, const float imageWidth);
VPointCloudXYZID sph_proj(VPointCloud::Ptr cloud_i, const float phi_center, const float phi_range, const float imageWidth);
void SSNspan_config(float *OUT_ptr, const char ViewType, const float phi_center);

#endif
