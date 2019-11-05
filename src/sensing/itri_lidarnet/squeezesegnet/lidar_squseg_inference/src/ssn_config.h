#ifndef SSN_CONFIG_H_
#define SSN_CONFIG_H_


#include <string>
#include "preprolib_squseg.h"
#include "tf_utils.hpp"
#include <boost/filesystem.hpp> 
#include <pcl/common/time.h>
#include <ros/ros.h>
#include <ros/package.h>


namespace BFS = boost::filesystem;


void norm_mean(float* mean_ptr,string data_set,char ViewType,float phi_center);

void norm_std(float* std_ptr, string data_set,char ViewType,float phi_center);

float proj_center(string data_set, int index);

vector<float> phi_center_grid(char ViewType);


class TF_inference
{
  public:

    TF_inference();
    TF_inference(string input_data_set, char input_ViewType, float input_phi_center, int input_pub_type);
    ~TF_inference();
    int TF_init();
    void TF_run(VPointCloud::Ptr release_Cloud, VPointCloudXYZIL::Ptr result_cloud);
    void TF_quit();

  private:

    TF_Status *status = NULL;
    TF_Session *sess  = NULL;
    TF_Output out_op;

    std::vector<TF_Output> input_ops;

    string data_set;
    char ViewType = '0';
    float phi_center;
    string phi_center_name;
    int pub_type;

    float INPUT_MEAN[5], INPUT_STD[5];
    float SPAN_PARA[2];                 // {span, imagewidth}
    float x_projCenter = 0;
    float z_projCenter = 0;

    pcl::StopWatch stopWatch;
};

#endif /* SSN_CONFIG_H_ */
