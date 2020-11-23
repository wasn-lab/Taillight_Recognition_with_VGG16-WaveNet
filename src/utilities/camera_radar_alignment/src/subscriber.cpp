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
#include <pcl/common/transforms.h>  
#include <pcl/io/boost.h>
#include <sensor_msgs/image_encodings.h>
#include <msgs/Rad.h>
#include <msgs/PointXYZV.h>
#include <msgs/CamInfo.h>
#include <msgs/DetectedObject.h>
#include <msgs/DetectedObjectArray.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <thread>
#include <vector>
#include <cmath>

Projector projector;
cv::Mat M_MID(cvSize(1920, 1208), CV_8UC3, cvScalar(0));
std::vector<msgs::PointXYZV> radar_points;
msgs::DetectedObjectArray camera_objects;
Eigen::Matrix4f front_top_to_HDL;
cv::Mat trans_mid;
ros::Publisher fusion_result_pub;
ros::Publisher fusion_result_pointcloud_pub;

void callbackCameraObjects(const msgs::DetectedObjectArray& msg)
{
  if (!msg.objects.empty())
  {
    camera_objects = msg;
  }
}

void callbackCamera(const sensor_msgs::Image::ConstPtr& msg)
{
  cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
  M_MID = cv_ptr->image;
}

void callbackRadar(const msgs::Rad& msg)
{
  if (!msg.radPoint.empty())
  {
    radar_points.resize(msg.radPoint.size());
    for (size_t i = 0; i < msg.radPoint.size(); i++)
    {
      radar_points[i] = msg.radPoint[i];
    }
  }
}

void sendFusionResult(std::vector<std::vector<double>> fusion_result)
{
  if (!fusion_result.empty())
  {
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::PointXYZI temp;
    for (size_t i = 0; i < fusion_result.size(); i++)
    {
      if (fusion_result[i][0] != 0 && fusion_result[i][1] != 0)
      {
        camera_objects.objects[i].radarInfo.imgPoint60.x = fusion_result[i][0];
        camera_objects.objects[i].radarInfo.imgPoint60.y = fusion_result[i][1];

        //delphi x front y right
        temp.x = fusion_result[i][0];
        temp.y = -fusion_result[i][1];
        cloud->points.push_back(temp);
        sensor_msgs::PointCloud2 msgtemp;
        pcl::toROSMsg(*cloud, msgtemp);
        msgtemp.header = camera_objects.header;
        msgtemp.header.frame_id = "base_link";
        fusion_result_pointcloud_pub.publish(msgtemp);
      }
    }
    fusion_result_pub.publish(camera_objects);
  }
}

void Merge(std::vector<msgs::PointXYZV> &Array, int front, int mid, int end)
{
  msgs::PointXYZV Max;
  Max.x = 1000;
  std::vector<msgs::PointXYZV> LeftSub(Array.begin()+front, Array.begin()+mid+1),
                     RightSub(Array.begin()+mid+1, Array.begin()+end+1);
  LeftSub.insert(LeftSub.end(), Max);     
  RightSub.insert(RightSub.end(), Max);    
  int idxLeft = 0, idxRight = 0;
  for (int i = front; i <= end; i++) 
  {
    if (LeftSub[idxLeft].x <= RightSub[idxRight].x ) 
    {
      Array[i] = LeftSub[idxLeft];
      idxLeft++;
    }
    else
    {
      Array[i] = RightSub[idxRight];
      idxRight++;
    }
  }
}

void MergeSort(std::vector<msgs::PointXYZV> &array, int front, int end)
{
  if (front < end)
  {            
    int mid = (front+end)/2;    
    MergeSort(array, front, mid);   
    MergeSort(array, mid+1, end); 
    Merge(array, front, mid, end);  
  }
}

std::vector<std::vector<double>> fusion()
{
  std::vector<std::vector<double>> empty;
  if (!radar_points.empty() && !camera_objects.objects.empty())
  {
    //preprocessing
    MergeSort(radar_points, 0, radar_points.size() - 1);
    //start fusion
    std::vector<std::vector<double>> fusion_result(camera_objects.objects.size(), std::vector<double> (5, 0));
    size_t camera_index = 0;
    for (size_t i = 0; i < radar_points.size(); i++)
    {
      if (camera_index >= camera_objects.objects.size())
      {
        break;
      }
      ///////////////////////////////
      //camera world y front x right
      //radar input y front x right
      ///////////////////////////////
      //delphi x front y right
      std::vector<int> result = projector.project(radar_points[i].y, radar_points[i].x, 0, 0);
      //mingtai y front x left
      //std::vector<int> result = projector.project(-radar_points[i].x, radar_points[i].y, 0, -7);
      double V_max = -1;
      int index = -1, classId = -1;
      for (size_t j = 0; j < camera_objects.objects.size(); j++)
      {
        double scale_u = 608.0 / 1920.0;
        double scale_v = 384.0 / 1208.0;
        double u = camera_objects.objects[j].camInfo[0].u;
        double v = camera_objects.objects[j].camInfo[0].v;
        double width = camera_objects.objects[j].camInfo[0].width;
        double height = camera_objects.objects[j].camInfo[0].height;
        double center_u = (u + width / 2) * scale_u;
        double center_v = (v + height / 2) * scale_v;
        double v_max = (v + height) * scale_v;
        double radius = sqrt(pow(width / 2 * scale_u, 2) + pow(height / 2 * scale_v, 2)) * 1.2;
        double distance = sqrt(pow(result[0] - center_u, 2) + pow(result[1] - center_v, 2));
        if (distance <= radius && v_max > V_max)
        {
          V_max = v_max;
          index = j;
        }
      }
      if (index != -1)
      {
        classId = camera_objects.objects[index].classId;
        //delphi x front y right
        fusion_result[index][0] = radar_points[i].x;
        fusion_result[index][1] = radar_points[i].y;
        fusion_result[index][2] = result[0];
        fusion_result[index][3] = result[1];
        fusion_result[index][4] = classId;
        if (classId == 2 || classId == 3)
        {
          i--;
        }
        //camera_objects.erase(camera_objects.begin() + index);
        camera_index++;
      }
    }
    return fusion_result;
  }
  return empty;
}


void detection(int argc, char** argv)
{
  ros::init(argc, argv, "LidFrontTop_idsXC2_fusion");
  ros::NodeHandle n;

  projector.init();

  image_transport::ImageTransport it(n);
  image_transport::Subscriber sub_image2 = it.subscribe("/cam/front_bottom_60/detect_image", 1, callbackCamera);

  ros::Subscriber CameraObjectsSub = n.subscribe("/cam_obj/front_bottom_60", 1, callbackCameraObjects);

  ros::Subscriber RadFrontSub = n.subscribe("/RadFront", 1, callbackRadar);

  fusion_result_pub = n.advertise<msgs::DetectedObjectArray>("/cam_obj/front_bottom_60/radar_fusion", 1);
  fusion_result_pointcloud_pub = n.advertise<sensor_msgs::PointCloud2>("/cam_obj/front_bottom_60/radar_fusion/pointcloud", 1);

  std::vector<double> camera_angle, radar_angle;
  while (ros::ok())
  {
    //0:1, 1:0.1, 2:0.1, 3:0.1, 4:1, 5:1
    cv::Mat M_MID_temp;
    M_MID.copyTo(M_MID_temp);
    cv::resize(M_MID_temp, M_MID_temp, cv::Size(608, 384), 0, 0, cv::INTER_LINEAR);
    //double scaleFactor = M_MID_temp.rows / 384;
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
    std::vector<std::vector<double>> fusion_result = fusion();
    if (!fusion_result.empty())
    {
      for (size_t i = 0; i < fusion_result.size(); i++)
      {
        if (fusion_result[i][2] >= 0 && fusion_result[i][3] >= 0 && fusion_result[i][2] < M_MID_temp.cols && fusion_result[i][3] < M_MID_temp.rows)
        {
          int red_int = 255, gre_int = 0, blu_int = 0;
          double depths_float = fusion_result[i][0];
          std::stringstream ss;
          ss << std::setprecision(3) << abs(depths_float);
          string distance = ss.str();
          cv::circle(M_MID_temp, cv::Point(fusion_result[i][2], fusion_result[i][3]), 6, CV_RGB(red_int, gre_int, blu_int), -1, 8, 0);
          cv::putText(M_MID_temp, distance, cv::Point(fusion_result[i][2] + 3, fusion_result[i][3]), cv::FONT_HERSHEY_DUPLEX, 0.6, CV_RGB(red_int, gre_int, blu_int), 2);
        }
      }
      sendFusionResult(fusion_result);
    }
/*
    for (size_t i = 0; i < radar_points.size(); i++)
    {
      //delphi x front y right
      std::vector<int> result = projector.project(radar_points[i].y, radar_points[i].x, 0, 0);
      //mingtai y front x left
      //std::vector<int> result = projector.project(-radar_points[i].x, radar_points[i].y, 0, -7);
      if (result[0] >= 0 && result[1] >= 0 && result[0] < M_MID_temp.cols && result[1] < M_MID_temp.rows)
      {
        int red_int = 255, gre_int = 0, blu_int = 0;
        double depths_float = (double)radar_points[i].x;
        std::stringstream ss;
        ss << std::setprecision(3) << abs(depths_float);
        string distance = ss.str();
        cv::circle(M_MID_temp, cv::Point(result[0], result[1]), 6, CV_RGB(red_int, gre_int, blu_int), -1, 8, 0);
        cv::putText(M_MID_temp, distance, cv::Point(result[0] + 3, result[1]), cv::FONT_HERSHEY_DUPLEX, 0.6, CV_RGB(red_int, gre_int, blu_int), 2);
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
