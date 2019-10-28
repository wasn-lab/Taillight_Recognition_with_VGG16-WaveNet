#ifndef __EVAL_UTILITY__
#define __EVAL_UTILITY__

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


using namespace std;

VPointCloudXYZIL pcdExtract_byClass(VPointCloudXYZIL::Ptr cloud_il, int class_index);

class pointcloudEval
{
  private:
    VPointCloudXYZIL::Ptr cloud_GT, cloud_PD;

  public:
    float iou;
    bool GT_cloudExist, PD_cloudExist;

    pointcloudEval ();
    pointcloudEval (VPointCloudXYZIL::Ptr input_cloud_GT, VPointCloudXYZIL::Ptr input_cloud_PD);

    ~pointcloudEval ();

    void IOUcal_pointwise(int class_index);


};


#endif
