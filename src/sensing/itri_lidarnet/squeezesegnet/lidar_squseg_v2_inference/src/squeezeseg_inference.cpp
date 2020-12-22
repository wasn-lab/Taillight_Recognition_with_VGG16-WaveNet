#include "squeezeseg_inference_nodelet.h"

// namespace BFS = boost::filesystem;

// position of projection center
// #define x_projCenter -2
// #define z_projCenter -1.4

using ssn_v2_nodelet::data_set;
using ssn_v2_nodelet::debug_output;
using ssn_v2_nodelet::hybrid_detect;
using ssn_v2_nodelet::LidarAllSub;
using ssn_v2_nodelet::nn_pub;
using ssn_v2_nodelet::pub_type;
using ssn_v2_nodelet::SSN_all;
using ssn_v2_nodelet::ViewType;

int main(int argc, char** argv)
{
  if (argc >= 2)
  {
    data_set = string(argv[1]);
    ViewType = *argv[2];
    // stringstream ss1, ss2, ss3;
    // ss1 << string (argv[3]);
    // ss1 >> phi_center;

    // ss2 << string (argv[4]);
    // ss2 >> pub_type;

    // ss3 << string (argv[5]);
    // ss3 >> hybrid_detect;
    stringstream ss1, ss2;
    ss1 << string(argv[3]);
    ss1 >> pub_type;

    ss2 << string(argv[4]);
    ss2 >> hybrid_detect;
  }

  ros::init(argc, argv, "cpp_preprocessing");
  ros::NodeHandle nh;

  ros::param::get("/debug_output", ssn_v2_nodelet::debug_output);

  LidarAllSub = nh.subscribe("/LidarAll/NonGround", 1, ssn_v2_nodelet::LidarsNodelet::callback_LidarAll);
  nn_pub = nh.advertise<pcl::PointCloud<pcl::PointXYZIL>>("/squ_seg/result_cloud", 1);

  // all_pub = nh.advertise<sensor_msgs::PointCloud2> ("/release_cloud", 1);

  // =============== Tensorflow ==================
  vector<float> phi_center_all = phi_center_grid(ViewType);

  for (size_t i = 0; i < phi_center_all.size(); i++)
  {
    SSN_all.push_back(TF_inference(data_set, ViewType, phi_center_all.at(i), pub_type));
  }

  // SSN_P0deg = TF_inference(data_set,ViewType,phi_center,pub_type);

  vector<int> TF_ERcode(phi_center_all.size());
  for (size_t i = 0; i < phi_center_all.size(); i++)
  {
    TF_ERcode.at(i) = SSN_all.at(i).TF_init();
  }

  // int TF_ERcode = SSN_P0deg.TF_init();

  ros::AsyncSpinner spinner(1);
  spinner.start();

  while (ros::ok())
  {
    ros::Rate(1).sleep();
  }

  for (size_t i = 0; i < phi_center_all.size(); i++)
  {
    SSN_all.at(i).TF_quit();
  }
  LidarAllSub.shutdown();

  // SSN_P0deg.TF_quit();

  return 0;
}
