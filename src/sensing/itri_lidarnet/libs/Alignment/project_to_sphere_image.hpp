#ifndef PROJECT_TO_SPHERE_IMAGE_HPP_
#define PROJECT_TO_SPHERE_IMAGE_HPP_

#include <ros/ros.h>
#include <ros/package.h>
#include <rosbag/bag.h>
#include <std_msgs/Header.h>
#include <sensor_msgs/PointCloud2.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "../UserDefine.h"

using namespace pcl;

// spherical projection + front-facing 90 degrees -> ROI
void
project_to_sphere_image (PointCloud<PointXYZ>::Ptr input, int imageWidth,int imageHeight,float lidarHOV,float lidarVOV)
{

  PointCloud<PointXYZID> image_cloud;

  image_cloud.resize(imageHeight*imageWidth);

  float dphi   = (float)imageWidth  / (lidarHOV * D2R);
  float dtheta = (float)imageHeight / (lidarVOV * D2R);

#pragma omp parallel for
  for (size_t i = 0; i < input->points.size (); i++)
  {
    float x = input->points[i].x;
    float y = input->points[i].y;
    float z = input->points[i].z+4;
    float d = sqrt (x * x + y * y + z * z);

    float phi = atan2 (y, x);   // horizontal angle
    //float theta = acos (z / d); // vertical angle   >>> 0 < theta < pi | 69.9517 < theta < 96.3936

    int phi_p = (int) ((phi + (lidarHOV/2)*D2R) * dphi);
    int theta_p = (int) (-asin(z / d) * dtheta ); //angle of view

    //cout << "horizontal " << phi_p << ", vertical " << theta_p << endl;

    // front-facing 90 degrees
    if (phi > (-lidarHOV/2)*D2R && phi < (lidarHOV/2)*D2R )
    {
      if (phi_p < 0)    phi_p = 0;
      if (phi_p >= imageWidth) phi_p = imageWidth-1;

      if (theta_p < 0)   theta_p = 0;
      if (theta_p >= imageHeight) theta_p = imageHeight -1;

      image_cloud.points[theta_p*imageWidth  + phi_p].x = 0;
      image_cloud.points[theta_p*imageWidth  + phi_p].y = input->points[i].y;
      image_cloud.points[theta_p*imageWidth  + phi_p].z = input->points[i].z;
      image_cloud.points[theta_p*imageWidth  + phi_p].d = d;
    }

  }


  static ros::NodeHandle nh;
  static ros::Publisher spR_pub = nh.advertise<sensor_msgs::PointCloud2>("/image_cloud", 1);

  sensor_msgs::PointCloud2 spR_msg;
  pcl::toROSMsg(image_cloud, spR_msg);
  spR_msg.header.frame_id = "lidar";
  spR_msg.header.stamp = ros::Time::now();
  spR_pub.publish(spR_msg);

}

#endif
