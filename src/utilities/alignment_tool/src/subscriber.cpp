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
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

cv::Mat g_m_mid(cvSize(1920, 1208), CV_8UC3, cvScalar(0));
boost::mutex g_mutex_lidar_all;

pcl::PointCloud<pcl::PointXYZI> g_lidar_all_cloud;
pcl::PointCloud<pcl::PointXYZI>::Ptr g_lidar_all_cloud_ptr(new pcl::PointCloud<pcl::PointXYZI>);

void callbackCamera(const sensor_msgs::Image::ConstPtr& msg)
{
  cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
  g_m_mid = cv_ptr->image;
}

void callbackLidarAll(const pcl::PointCloud<pcl::PointXYZI>::ConstPtr& msg)
{
  g_mutex_lidar_all.lock();
  g_lidar_all_cloud.width = msg->points.size();
  g_lidar_all_cloud.height = 1;
  g_lidar_all_cloud.is_dense = false;
  g_lidar_all_cloud.points.resize(g_lidar_all_cloud.width);

#pragma omp parallel for
  for (size_t i = 0; i < msg->points.size(); i++)
  {
    g_lidar_all_cloud.points[i].x = (float)msg->points[i].x;
    g_lidar_all_cloud.points[i].y = (float)msg->points[i].y;
    g_lidar_all_cloud.points[i].z = (float)msg->points[i].z;
    g_lidar_all_cloud.points[i].intensity = (int)msg->points[i].intensity;
    g_lidar_all_cloud.points[i].intensity = 50;
  }
  *g_lidar_all_cloud_ptr = g_lidar_all_cloud;

  g_mutex_lidar_all.unlock();
}

void detection(int argc, char** argv)
{
  ros::init(argc, argv, "LidFrontTop_idsXC2_fusion");
  ros::NodeHandle n;

  Projector3 projector;
  projector.init(/*0*/);

  image_transport::ImageTransport it(n);
  image_transport::Subscriber sub_image2 = it.subscribe("/cam/front_bottom_60/raw", 1, callbackCamera);

  ros::Subscriber lid_front_top_sub = n.subscribe("/LidarAll", 1, callbackLidarAll);

  while (ros::ok())
  {
    //0:1, 1:0.1, 2:0.1, 3:0.1, 4:1, 5:1
    projector.setcameraMat(0,0,0,0);
    projector.setprojectionMat(GlobalVariable::UI_PARA[0], GlobalVariable::UI_PARA[1]*10,GlobalVariable::UI_PARA[2]*10, GlobalVariable::UI_PARA[3]*10,GlobalVariable::UI_PARA[4],GlobalVariable::UI_PARA[5]);

    pcl::PointCloud<pcl::PointXYZI>::Ptr release_cloud(new pcl::PointCloud<pcl::PointXYZI>);

    g_mutex_lidar_all.lock();
    cv::Mat m_mid_temp;
    g_m_mid.copyTo(m_mid_temp);
    cv::resize(m_mid_temp, m_mid_temp, cv::Size(608, 342), 0, 0, cv::INTER_LINEAR);
    *release_cloud = *g_lidar_all_cloud_ptr;
    double scale_factor = m_mid_temp.rows / 342.0;
    for (size_t i = 0; i < release_cloud->size(); i++)
    {
      if (!projector.outOfFov(release_cloud->points[i].x, release_cloud->points[i].y, release_cloud->points[i].z))
      {
        std::vector<int> result = projector.project(
            (float)release_cloud->points[i].x, (float)release_cloud->points[i].y, (float)release_cloud->points[i].z);
        result[0] = result[0] * scale_factor;
        result[1] = result[1] * scale_factor;
        if (result[0] >= 0 && result[1] >= 0 && result[0] < m_mid_temp.cols && result[1] < m_mid_temp.rows)
        {
          auto hight = (double)release_cloud->points[i].z;
          int red_int = 0, gre_int = 0, blu_int = 0;
          auto depths_float = (double)release_cloud->points[i].x;
          if(hight > -2.4 && hight < -0.3)
          {
            cv::circle(m_mid_temp, cv::Point(result[0], result[1]), 2.5, CV_RGB(0, 0, 0), -1, 8, 0);
          }
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
          cv::circle(m_mid_temp, cv::Point(result[0], result[1]), 0.75, CV_RGB(red_int, gre_int, blu_int), -1, 8, 0);
        }
      }
    }
    // cv::namedWindow("OPENCV_LIDAR_WINDOW_MID", CV_WINDOW_NORMAL);
    // cv::resizeWindow("OPENCV_LIDAR_WINDOW_MID", 1280, 720);
    // cv::imshow("OPENCV_LIDAR_WINDOW_MID", M_MID);
    // cv::waitKey(1);
    cv::namedWindow("Alignment_view", cv::WINDOW_NORMAL);
    cv::imshow("Alignment_view", m_mid_temp);
    cv::waitKey(1);

    g_mutex_lidar_all.unlock();
    ros::spinOnce();
  }
}

int main(int argc, char** argv)
{
  thread thead_detection(detection, argc, argv);
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
