#include "drivenet/alignment_offline_b1.h"
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

void AlignmentOff::init(int car_id)
{
  carId = car_id;
  pj.init(camera::id::front_60);
  imgW = 1920;
  imgH = 1208;
  groundUpBound = -2.44;
  groundLowBound = -2.84;
}

vector<int> AlignmentOff::run(float x, float y, float z)
{
  return pj.project(x,y,z);
}

// Main
AlignmentOff al;

void callback_LidarAll(const sensor_msgs::PointCloud2::ConstPtr& msg)
{
  pcl::PointCloud<pcl::PointXYZI>::Ptr LidAll_cloudPtr(new pcl::PointCloud<pcl::PointXYZI>);  
  pcl::fromROSMsg(*msg, *LidAll_cloudPtr);

  for(size_t i = 0; i < LidAll_cloudPtr->size(); i++)
  {
    if(LidAll_cloudPtr->points[i].z > al.groundLowBound && LidAll_cloudPtr->points[i].z < al.groundUpBound
    && LidAll_cloudPtr->points[i].x > 0)
    {
      al.out = al.run(LidAll_cloudPtr->points[i].x, LidAll_cloudPtr->points[i].y, LidAll_cloudPtr->points[i].z);
      if(al.out[0] > 0 && al.out[0] < al.imgW && al.out[1] > 0 && al.out[1] < al.imgH)
      {
        std::cout << "(" << al.out[0] << "," << al.out[1] << "):";
        std::cout << LidAll_cloudPtr->points[i].x << "," << LidAll_cloudPtr->points[i].y << "," << LidAll_cloudPtr->points[i].z << std::endl;
      }
    }

  }
}

int main(int argc, char** argv)
{
  // new
  ros::init(argc, argv, "alignmentOff");
  ros::NodeHandle nh;

  ros::Subscriber LidarSc;
  LidarSc = nh.subscribe("LidarAll", 1, callback_LidarAll);

  al.init(1);
  // al.out = al.run();

  ros::spin();

  return 0;
}