#include "UI/QtViewer.h"
#include "all_header.h"
#include "projector3.h"
#include "ros/ros.h"
#include "sensor_msgs/PointCloud2.h"
#include "std_msgs/Header.h"
#include <boost/chrono.hpp>
#include <chrono>
#include <cstdlib>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <iostream>
#include <mutex>
#include <nodelet/nodelet.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <pcl/common/common_headers.h>
#include <pcl/common/transforms.h>  //transform​
#include <pcl/io/boost.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/image_encodings.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <thread>
#include <vector>

cv::Mat M_MID(cvSize(1920, 1208), CV_8UC3, cvScalar(0));
boost::mutex mutex_LidarAll;
Eigen::Matrix4f front_top_to_HDL;
cv::Mat trans_mid;

pcl::PointCloud<pcl::PointXYZI> LidarAll_cloud;
pcl::PointCloud<pcl::PointXYZI>::Ptr LidarAll_cloudPtr(new pcl::PointCloud<pcl::PointXYZI>);

void trans_function(cv::Mat& transform, float* p)
{
  // std::cout << p[0] << " " << p[1] << " "  << p[2] << " "  << p[3] << " "  <<
  // p[4] << " "  << p[5] << " "  << p[6] << std::endl;
  float cos_x = cosf(p[1] * (3.1416 / 180));
  float sin_x = sinf(p[1] * (3.1416 / 180));
  float cos_y = cosf(p[2] * (3.1416 / 180));
  float sin_y = sinf(p[2] * (3.1416 / 180));
  float cos_z = cosf(p[3] * (3.1416 / 180));
  float sin_z = sinf(p[3] * (3.1416 / 180));
  // std::cout << p[0] << " " << cos_x << " "  << sin_x << " "  << cos_y << " "
  // << sin_y << " "  << cos_z << " "  << sin_z << std::endl;

  float f[9] = { p[0], 0, 640, 0, -p[0], 360, 0, 0, 1 };
  cv::Mat foc = cv::Mat(3, 3, CV_32FC1, f);

  float Rx[9] = { 1, 0, 0, 0, cos_x, -sin_x, 0, sin_x, cos_x };
  cv::Mat rx = cv::Mat(3, 3, CV_32FC1, Rx);

  float Ry[9] = { cos_y, 0, sin_y, 0, 1, 0, -sin_y, 0, cos_y };
  cv::Mat ry = cv::Mat(3, 3, CV_32FC1, Ry);

  float Rz[9] = { cos_z, -sin_z, 0, sin_z, cos_z, 0, 0, 0, 1 };
  cv::Mat rz = cv::Mat(3, 3, CV_32FC1, Rz);
  float Txyz[3] = { p[4], p[6], p[5] };
  cv::Mat txyz = cv::Mat(3, 1, CV_32FC1, Txyz);
  cv::Mat rot_tmp = (rx * ry) * rz;
  rot_tmp = foc * rot_tmp;
  cv::hconcat(rot_tmp, txyz, transform);
  std::cout << transform << std::endl;
}

// 1272.796   -48.003   591.428   -60.000
//   15.913 -1276.116   251.744 -1340.000
//    0.037    -0.085     0.996    -0.500

void callbackCamera(const sensor_msgs::Image::ConstPtr& msg)
{
  cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
  M_MID = cv_ptr->image;
}

void callbackLidarAll(const pcl::PointCloud<pcl::PointXYZI>::ConstPtr& msg)
{
  mutex_LidarAll.lock();
  LidarAll_cloud.width = msg->points.size();
  LidarAll_cloud.height = 1;
  LidarAll_cloud.is_dense = false;
  LidarAll_cloud.points.resize(LidarAll_cloud.width);

#pragma omp parallel for
  for (size_t i = 0; i < msg->points.size(); i++)
  {
    LidarAll_cloud.points[i].x = (float)msg->points[i].x;
    LidarAll_cloud.points[i].y = (float)msg->points[i].y;
    LidarAll_cloud.points[i].z = (float)msg->points[i].z;
    LidarAll_cloud.points[i].intensity = (int)msg->points[i].intensity;
    LidarAll_cloud.points[i].intensity = 50;
  }
  *LidarAll_cloudPtr = LidarAll_cloud;

  mutex_LidarAll.unlock();
}

void detection(int argc, char** argv)
{
  ros::init(argc, argv, "LidFrontTop_idsXC2_fusion");
  ros::NodeHandle n;

  Projector3 projector;
  projector.init(0);

  image_transport::ImageTransport it(n);
  image_transport::Subscriber sub_image2 = it.subscribe("/cam/left_front_60", 1, callbackCamera);

  ros::Subscriber LidFrontTopSub = n.subscribe("/LidarAll", 1, callbackLidarAll);

  while (ros::ok())
  {
    projector.setcameraMat(0,0,0,0);
    projector.setprojectionMat(GlobalVariable::UI_PARA[0], GlobalVariable::UI_PARA[1] * 10, GlobalVariable::UI_PARA[2] * 10,GlobalVariable::UI_PARA[3],GlobalVariable::UI_PARA[4],GlobalVariable::UI_PARA[5]);

    pcl::PointCloud<pcl::PointXYZI>::Ptr release_cloud(new pcl::PointCloud<pcl::PointXYZI>);

    mutex_LidarAll.lock();
    cv::Mat M_MID_temp;
    M_MID.copyTo(M_MID_temp);
    cv::resize(M_MID_temp, M_MID_temp, cv::Size(608, 384), 0, 0, cv::INTER_LINEAR);
    *release_cloud = *LidarAll_cloudPtr;
    double scaleFactor = M_MID_temp.rows / 384;
    for (size_t i = 0; i < release_cloud->size(); i++)
    {
      if (!projector.outOfFov(release_cloud->points[i].x, release_cloud->points[i].y, release_cloud->points[i].z))
      {
        std::vector<int> result = projector.project(
            (float)release_cloud->points[i].x, (float)release_cloud->points[i].y, (float)release_cloud->points[i].z);
        result[0] = result[0] * scaleFactor;
        result[1] = result[1] * scaleFactor;
        if (result[0] >= 0 && result[1] >= 0 && result[0] < M_MID_temp.cols && result[1] < M_MID_temp.rows)
        {
          double hight = (double)release_cloud->points[i].z;
          if(hight > -2)
          {
            cv::circle(M_MID_temp, cv::Point(result[0], result[1]), 3, CV_RGB(0, 0, 0), -1, 8, 0);
          }
          int red_int = 0, gre_int = 0, blu_int = 0;
          double depths_float = (double)release_cloud->points[i].x;
            if (abs(depths_float) <= 7)
            {
              red_int = 255;
              gre_int = abs(depths_float) * 5;
              blu_int = 0;
            }
            else if (abs(depths_float) <= 10)
            {
              red_int = 510 - (abs(depths_float) * 5);
              gre_int = 255;
              blu_int = 0;
            }
            else if (abs(depths_float) <= 14)
            {
              red_int = 0;
              gre_int = 255;
              blu_int = (abs(depths_float) - 102) * 5;
            }
            else if (abs(depths_float) <= 21)
            {
              red_int = 0;
              gre_int = 1020 - (abs(depths_float) * 5);
              blu_int = 255;
            }
            else
            {
              red_int = (abs(depths_float) - 204) * 5;
              gre_int = 0;
              blu_int = 255;
            }
          cv::circle(M_MID_temp, cv::Point(result[0], result[1]), 1, CV_RGB(red_int, gre_int, blu_int), -1, 8, 0);
        }
      }
    }
    // cv::namedWindow("OPENCV_LIDAR_WINDOW_MID", CV_WINDOW_NORMAL);
    // cv::resizeWindow("OPENCV_LIDAR_WINDOW_MID", 1280, 720);
    // cv::imshow("OPENCV_LIDAR_WINDOW_MID", M_MID);
    // cv::waitKey(1);
    cv::namedWindow("Alignment_view", cv::WINDOW_NORMAL);
    cv::imshow("Alignment_view", M_MID_temp);
    cv::waitKey(1);

    mutex_LidarAll.unlock();
    ros::spinOnce();
  }
}

int main(int argc, char** argv)
{
  thread TheadDetection(detection, argc, argv);
  QApplication a(argc, argv);
  QtViewer w;
  w.show();
  a.exec();
  while (true)
  {
    boost::this_thread::sleep(boost::posix_time::microseconds(1000000));
  }
  return 0;
}
