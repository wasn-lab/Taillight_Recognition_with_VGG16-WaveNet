#include <iostream>
#include <cmath>
#include <mutex>
#include <omp.h>
#include <thread>

#include <ros/ros.h>
#include <ros/package.h>
#include <sensor_msgs/PointCloud2.h>

#include <pcl/point_types.h>
#include <pcl/common/time.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>

#include <eigen3/Eigen/Dense>

// #include "preprolib_squseg.h"

#include "ssn_config.h"

// #include "tf_utils.hpp"

// #include <boost/filesystem.hpp> 

#include "CuboidFilter.h"


// namespace BFS = boost::filesystem;

ros::Publisher nn_pub;        // spherical projection
// ros::Publisher all_pub;        // publish release_cloud subscribed from lidar_preprocessing (/Lidarall/Nonground)

string data_set;
bool hybrid_detect = 0;
// int pub_type = 0;
// float phi_center = 0;
// char ViewType = 'X';
// float INPUT_MEAN[5], INPUT_STD[5];
// float SPAN_PARA[2];         // {span, imagewidth}

// position of projection center
// #define x_projCenter -2
// #define z_projCenter -1.4
// const float x_projCenter = proj_center(data_set,0);
// const float z_projCenter = proj_center(data_set,1);

//  ======= global variabes of tensorflow =======
// std::vector<TF_Output> input_ops;
// TF_Output out_op;
// TF_Status *status;
// TF_Session *sess;

vector<TF_inference> SSN_all;

void
callback_LidarAll (const pcl::PointCloud<pcl::PointXYZI>::ConstPtr& msg)
{
  // cout << "TensorFlow Version: " << TF_Version() << endl;

  pcl::StopWatch stopWatch;

  VPointCloud::Ptr release_Cloud (new VPointCloud);

  *release_Cloud = *msg;

  if (release_Cloud->size () > 100)
  {
    //   ===============  temporally enable part of rule-based code for ensuring front-view detection ==========================
    VPointCloud::Ptr select_Cloud(new VPointCloud);
    if (hybrid_detect == 1)
    {
      *select_Cloud = CuboidFilter().pass_through_soild<PointXYZI>(release_Cloud, -30, 50, -3.0, 3.0, -3.12, 0);
    }
    //   ========================================================================================================================
    
    // the origin of coordinate is shifted -2 m in x-axis and -1.4 m in z-axis in SSN to maximize projection ratio
    // Note that following shift will be recovered in TF_inference::TF_run() !!!!!!!!!!!!!1
    const float x_projCenter = proj_center(data_set, 0);
    const float z_projCenter = proj_center(data_set, 1);

    for (size_t i = 0; i < release_Cloud->points.size (); i++)
    {
      release_Cloud->points[i].x = release_Cloud->points[i].x - x_projCenter;
      // float release_Cloud->points[i].y = release_Cloud->points[i].y;
      release_Cloud->points[i].z = release_Cloud->points[i].z - z_projCenter;
    }

    vector<VPointCloudXYZIL::Ptr> result_cloud;
    for(size_t i=0; i < SSN_all.size(); i++)
      result_cloud.push_back(VPointCloudXYZIL::Ptr(new VPointCloudXYZIL));

    // VPointCloudXYZIL::Ptr result_cloud(new VPointCloudXYZIL);

#if(false)
    vector<thread> mthreads;
    for(size_t i=0; i < SSN_all.size(); i++)
    {
      mthreads.push_back(thread(&TF_inference::TF_run, &SSN_all.at(i), release_Cloud, result_cloud.at(i)));
      // ros::Duration(0.02).sleep();
    }

    for(size_t i=0; i < SSN_all.size(); i++)
      mthreads.at(i).join();
#else
    SSN_all.at(0).TF_run(release_Cloud, result_cloud.at(0));
    SSN_all.at(1).TF_run(release_Cloud, result_cloud.at(1));
    SSN_all.at(2).TF_run(release_Cloud, result_cloud.at(2));
    SSN_all.at(3).TF_run(release_Cloud, result_cloud.at(3));
#endif

    VPointCloudXYZIL::Ptr result_cloud_all(new VPointCloudXYZIL);

    for(size_t i=0; i < SSN_all.size(); i++)
      *result_cloud_all += *result_cloud.at(i);

    result_cloud_all->header.frame_id = msg->header.frame_id;
    result_cloud_all->header.stamp = msg->header.stamp;
    // pcl_conversions::toPCL(ros::Time::now(), result_cloud.header.stamp);
    // result_cloud.header.seq = msg->header.seq;


    if (hybrid_detect)
    {
      for (size_t i = 0; i < (size_t) select_Cloud->points.size(); i++)
      {
        VPointXYZIL pointinfo;
        pointinfo.x = select_Cloud->points.at(i).x;
        pointinfo.y = select_Cloud->points.at(i).y;
        pointinfo.z = select_Cloud->points.at(i).z;
        pointinfo.intensity = select_Cloud->points.at(i).intensity;
        pointinfo.label = 4;

        result_cloud_all->push_back(pointinfo);
      }
    }

    nn_pub.publish (*result_cloud_all);
    result_cloud_all->clear();

    // ======== following comment used for debugging of subscription ========
    // sensor_msgs::PointCloud2 all_msg;
    // pcl::toROSMsg (*release_Cloud, all_msg);
    // all_msg.header.frame_id = "lidar";
    // all_msg.header.stamp = ros::Time::now ();
    // all_msg.header.seq = msg->header.seq;
    // all_pub.publish (all_msg);  // publish to /release_cloud
  }

  if (stopWatch.getTimeSeconds () > 0.05)
  {
    cout << "[SSN]:slow " << stopWatch.getTimeSeconds () << "s" << endl << endl;
  }
}

int
main (int argc,
      char **argv)
{
  char ViewType;
  // float phi_center;
  int pub_type;

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
    ss1 << string (argv[3]);
    ss1 >> pub_type;

    ss2 << string (argv[4]);
    ss2 >> hybrid_detect;
  }

  ros::init (argc, argv, "cpp_preprocessing");
  ros::NodeHandle nh;

  ros::Subscriber LidarAllSub = nh.subscribe ("/LidarAll/NonGround", 1, callback_LidarAll);
  nn_pub = nh.advertise<pcl::PointCloud<pcl::PointXYZIL>> ("/squ_seg/result_cloud", 1);

  // all_pub = nh.advertise<sensor_msgs::PointCloud2> ("/release_cloud", 1);

  // =============== Tensorflow ==================
  vector<float> phi_center_all = phi_center_grid(ViewType);
  
  for(size_t i=0; i < phi_center_all.size(); i++)
    SSN_all.push_back(TF_inference(data_set,ViewType,phi_center_all.at(i),pub_type));

  // SSN_P0deg = TF_inference(data_set,ViewType,phi_center,pub_type);

  vector<int> TF_ERcode(phi_center_all.size());
  for(size_t i=0; i < phi_center_all.size(); i++)
    TF_ERcode.at(i) = SSN_all.at(i).TF_init();

  // int TF_ERcode = SSN_P0deg.TF_init();

  ros::AsyncSpinner spinner (1);
  spinner.start ();

  while (ros::ok ())
  {
    ros::Rate(1).sleep ();
  }
  
  for(size_t i=0; i < phi_center_all.size(); i++)
    SSN_all.at(i).TF_quit();

  // SSN_P0deg.TF_quit();

  return 0;
}
