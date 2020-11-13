#include "UI/QtViewer.h"
#include "all_header.h"
#include "projector.h"
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
#include <msgs/Rad.h>
#include <msgs/PointXYZV.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <thread>
#include <vector>

cv::Mat M_MID(cvSize(1920, 1208), CV_8UC3, cvScalar(0));
std::vector<msgs::PointXYZV> radar_points;
Eigen::Matrix4f front_top_to_HDL;
cv::Mat trans_mid;

void callbackCamera(const sensor_msgs::Image::ConstPtr& msg)
{
  cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
  M_MID = cv_ptr->image;
}

void callbackRadar(const msgs::Rad& msg)
{
  radar_points.resize(msg.radPoint.size());
  for(int i = 0; i < msg.radPoint.size(); i++)
  {
    radar_points[i] = msg.radPoint[i];
  }
}

void detection(int argc, char** argv)
{
  ros::init(argc, argv, "LidFrontTop_idsXC2_fusion");
  ros::NodeHandle n;

  Projector projector;
  projector.init();

  image_transport::ImageTransport it(n);
  image_transport::Subscriber sub_image2 = it.subscribe("/cam/front_bottom_60", 1, callbackCamera);

  ros::Subscriber RadFrontSub = n.subscribe("/RadFront", 1, callbackRadar);

  std::vector<double> camera_angle, radar_angle;
  while (ros::ok())
  {
    //0:1, 1:0.1, 2:0.1, 3:0.1, 4:1, 5:1
    //projector.setcameraMat(GlobalVariable::UI_PARA[4],GlobalVariable::UI_PARA[5],0,0);
    //projector.setprojectionMat(GlobalVariable::UI_PARA[0], GlobalVariable::UI_PARA[1] * 10,GlobalVariable::UI_PARA[2] * 10, GlobalVariable::UI_PARA[3] * 10,0,0);

    cv::Mat M_MID_temp;
    M_MID.copyTo(M_MID_temp);
    cv::resize(M_MID_temp, M_MID_temp, cv::Size(608, 384), 0, 0, cv::INTER_LINEAR);
    double scaleFactor = M_MID_temp.rows / 384;
    cv::circle(M_MID_temp, cv::Point(GlobalVariable::UI_PARA[0], GlobalVariable::UI_PARA[1] * 10), 3, CV_RGB(255, 0, 0), -1, 8, 0);
    if (GlobalVariable::UI_PARA[5] < 0 && GlobalVariable::UI_TESTING_BUTTOM == true)
    {
      GlobalVariable::UI_TESTING_BUTTOM = false;
      //double h_camera, double x_p, double y_p, double x_cw, double y_cw, double z_cw
      camera_angle = projector.calculateCameraAngle(1.94, GlobalVariable::UI_PARA[0], GlobalVariable::UI_PARA[1] * 10, GlobalVariable::UI_PARA[2]/10, GlobalVariable::UI_PARA[3]/10,  GlobalVariable::UI_PARA[4]/100, false);
    }else if (GlobalVariable::UI_PARA[5] > 0 && GlobalVariable::UI_TESTING_BUTTOM == true)
    {
      GlobalVariable::UI_TESTING_BUTTOM = false;
      //double camera_alpha, double camera_beta, double h_camera, double h_r, double x_p, double y_p, double x_r, double y_r, double L_x, double L_y
      radar_angle = projector.calculateRadarAngle(camera_angle[0], camera_angle[1], 1.96, 0.86, GlobalVariable::UI_PARA[4]/100, GlobalVariable::UI_PARA[0], GlobalVariable::UI_PARA[1] * 10, GlobalVariable::UI_PARA[2]/10, GlobalVariable::UI_PARA[3]/10, 0, -7, false);
    }
/*
    if (!radar_angle.empty()) 
    {
      for (size_t i = 0; i < radar_points.size(); i++)
      {
        std::vector<int> result = projector.project(camera_angle[0], camera_angle[1], 1.96, radar_angle[0], radar_angle[1], 0.86, GlobalVariable::UI_PARA[4]/100, -radar_points[i].x, radar_points[i].y, 0, -7, false);
        cv::circle(M_MID_temp, cv::Point(result[0], result[1]), 3, CV_RGB(0, 255, 0), -1, 8, 0);
        std::vector<int> test = projector.project(camera_angle[0], camera_angle[1], 1.96, radar_angle[0], radar_angle[1], 0.86, GlobalVariable::UI_PARA[4]/100, GlobalVariable::UI_PARA[2]/10, GlobalVariable::UI_PARA[3]/10, 0, -7, false);
        cv::circle(M_MID_temp, cv::Point(test[0], test[1]), 3, CV_RGB(0, 0, 255), -1, 8, 0);
      }
    }
*/
    for (size_t i = 0; i < radar_points.size(); i++)
    {
      //camera world y front x right
      //radar input y front x right
      //delphi x front y right
      std::vector<int> result = projector.project(radar_points[i].y, radar_points[i].x, 0, 0);
      //mingtai y front x left
      //std::vector<int> result = projector.project(-radar_points[i].x, radar_points[i].y, 0, -7);
      if (result[0] >= 0 && result[1] >= 0 && result[0] < M_MID_temp.cols && result[1] < M_MID_temp.rows)
      {
        int red_int = 0, gre_int = 0, blu_int = 0;
        double depths_float = (double)radar_points[i].y;
          if (abs(depths_float) <= 10)
          {
            red_int = 255;
            gre_int = 0;
            blu_int = 0;
          }
          else if (abs(depths_float) <= 20)
          {
            red_int = 255;
            gre_int = 125;
            blu_int = 0;
          }
          else if (abs(depths_float) <= 30)
          {
            red_int = 255;
            gre_int = 255;
            blu_int = 0;
          }
          else if (abs(depths_float) <= 40)
          {
            red_int = 0;
            gre_int = 255;
            blu_int = 0;
          }
          else if (abs(depths_float) <= 50)
          {
            red_int = 0;
            gre_int = 0;
            blu_int = 255;
          }
          else if (abs(depths_float) <= 60)
          {
            red_int = 0;
            gre_int = 255;
            blu_int = 255;
          }
          else 
          {
            red_int = 255;
            gre_int = 0;
            blu_int = 255;
          }
        std::stringstream ss;
        ss << std::setprecision(3) << abs(depths_float);
        string distance = ss.str();
        cv::circle(M_MID_temp, cv::Point(result[0], result[1]), 4, CV_RGB(red_int, gre_int, blu_int), -1, 8, 0);
        cv::putText(M_MID_temp, distance, cv::Point(result[0] + 3, result[1]), cv::FONT_HERSHEY_DUPLEX, 0.5, CV_RGB(red_int, gre_int, blu_int), 2);
      }
    }
/*
    for (size_t i = 0; i < release_cloud->size(); i++)
    {
      if (!projector.outOfFov(release_cloud->points[i].x, release_cloud->points[i].y, release_cloud->points[i].z))
      {
        std::vector<int> result = projector.project((float)release_cloud->points[i].x, (float)release_cloud->points[i].y, (float)release_cloud->points[i].z);
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
*/
    cv::namedWindow("Alignment_view", cv::WINDOW_NORMAL);
    cv::imshow("Alignment_view", M_MID_temp);
    cv::waitKey(1);

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
