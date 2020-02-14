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

  spatial_points_ = new cv::Point3d*[imgW];
  assert(spatial_points_);
  // num_pcd_received_ = 0;
  for (int i = 0; i < imgW; i++)
  {
    spatial_points_[i] = new cv::Point3d[imgH];
  }
  assert(spatial_points_[imgW - 1]);

  for (int row = 0; row < imgW; row++)
  {
    for (int col = 0; col < imgH; col++)
    {
      spatial_points_[row][col].x = INIT_COORDINATE_VALUE;
      spatial_points_[row][col].y = INIT_COORDINATE_VALUE;
      spatial_points_[row][col].z = INIT_COORDINATE_VALUE;
    }
  }

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
        al.spatial_points_[al.out[0]][al.out[1]].x = LidAll_cloudPtr->points[i].x;
        al.spatial_points_[al.out[0]][al.out[1]].y = LidAll_cloudPtr->points[i].y;
        al.spatial_points_[al.out[0]][al.out[1]].z = LidAll_cloudPtr->points[i].z;

        // std::cout << al.spatial_points_[al.out[0]][al.out[1]].x;
        // std::cout << al.spatial_points_[al.out[0]][al.out[1]].y;
        // std::cout << al.spatial_points_[al.out[0]][al.out[1]].z;
      }
    }
  }
}

int main(int argc, char** argv)
{
  // new
  ros::init(argc, argv, "alignmentOff");
  ros::NodeHandle nh;
  ros::Rate r(10);


  ros::Subscriber LidarSc;
  LidarSc = nh.subscribe("LidarAll", 1, callback_LidarAll);

  al.init(1);
  // al.out = al.run();
  while (ros::ok())
  {
    ros::spinOnce();
    r.sleep();
  }

  // approx_nearest_points_if_necessary();
  // dump_dist_mapping();
  // dump_distance_in_json();

  // ros::spin();
  // return 0;
}