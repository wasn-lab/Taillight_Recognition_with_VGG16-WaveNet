#include "drivenet/alignment_offline_b1.h"
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

void AlignmentOff::init(int car_id)
{
  carId = car_id;
  pj.init(camera::id::front_60);
}

vector<int> AlignmentOff::run()
{
  return pj.project(1,2,3);
}

void callback_LidarAll(const sensor_msgs::PointCloud2::ConstPtr& msg)
{
  pcl::PointCloud<pcl::PointXYZI>::Ptr LidAll_cloudPtr(new pcl::PointCloud<pcl::PointXYZI>);  
  pcl::fromROSMsg(*msg, *LidAll_cloudPtr);
  std::cout << "success" << std::endl;
}

int main(int argc, char** argv)
{
  // new
  ros::init(argc, argv, "alignmentOff");
  ros::NodeHandle nh;

  ros::Subscriber LidarSc;
  LidarSc = nh.subscribe("LidarAll", 1, callback_LidarAll);

  AlignmentOff al;
  al.init(1);
  al.out = al.run();

  ros::spin();

  return 0;
}