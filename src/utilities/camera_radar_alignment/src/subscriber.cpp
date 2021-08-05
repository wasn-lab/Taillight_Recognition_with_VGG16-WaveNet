#include "UI/QtViewer.h"
#include "all_header.h"
#include "projector.h"
#include "tracker.h"
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
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>
#include <cmath>

Projector g_projector;
cv::Mat g_m_mid(cvSize(1920, 1208), CV_8UC3, cvScalar(0));
std::vector<msgs::PointXYZV> g_radar_points;
msgs::DetectedObjectArray g_camera_objects;
ros::Publisher g_tracking_result_pub;
ros::Publisher g_fusion_result_pub;
ros::Publisher g_fusion_result_pointcloud_pub;

void callbackCameraObjects(const msgs::DetectedObjectArray& msg)
{
    g_camera_objects = msg;
}

void callbackCamera(const sensor_msgs::Image::ConstPtr& msg)
{
  cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
  g_m_mid = cv_ptr->image;
}

void callbackRadar(const msgs::Rad& msg)
{
  if (!msg.radPoint.empty())
  {
    g_radar_points.resize(msg.radPoint.size());
    for (size_t i = 0; i < msg.radPoint.size(); i++)
    {
      g_radar_points[i] = msg.radPoint[i];
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
        g_camera_objects.objects[i].radarInfo.imgPoint60.x = fusion_result[i][0];
        g_camera_objects.objects[i].radarInfo.imgPoint60.y = fusion_result[i][1];

        //delphi x front y right
        temp.x = fusion_result[i][0];
        temp.y = -fusion_result[i][1];
        cloud->points.push_back(temp);
        sensor_msgs::PointCloud2 msgtemp;
        pcl::toROSMsg(*cloud, msgtemp);
        msgtemp.header = g_camera_objects.header;
        msgtemp.header.frame_id = "base_link";
        g_fusion_result_pointcloud_pub.publish(msgtemp);
      }
    }
    g_fusion_result_pub.publish(g_camera_objects);
  }
}

void Merge(std::vector<msgs::PointXYZV> &Array, int front, int mid, int end)
{
  msgs::PointXYZV max;
  max.x = 1000;
  std::vector<msgs::PointXYZV> left_sub(Array.begin()+front, Array.begin()+mid+1),
                     right_sub(Array.begin()+mid+1, Array.begin()+end+1);
  left_sub.insert(left_sub.end(), max);     
  right_sub.insert(right_sub.end(), max);    
  int idx_left = 0, idx_right = 0;
  for (int i = front; i <= end; i++) 
  {
    if (left_sub[idx_left].x <= right_sub[idx_right].x ) 
    {
      Array[i] = left_sub[idx_left];
      idx_left++;
    }
    else
    {
      Array[i] = right_sub[idx_right];
      idx_right++;
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
  if (!g_radar_points.empty() && !g_camera_objects.objects.empty())
  {
    //preprocessing
    MergeSort(g_radar_points, 0, g_radar_points.size() - 1);
    //start fusion
    std::vector<std::vector<double>> fusion_result(g_camera_objects.objects.size(), std::vector<double> (5, 0));
    size_t camera_index = 0;
    for (size_t i = 0; i < g_radar_points.size(); i++)
    {
      if (camera_index >= g_camera_objects.objects.size())
      {
        break;
      }
      ///////////////////////////////
      //camera world y front x right
      //radar input y front x right
      ///////////////////////////////
      //delphi x front y right
      std::vector<int> result = g_projector.project(g_radar_points[i].y, g_radar_points[i].x, 0, 0);
      //mingtai y front x left
      //std::vector<int> result = g_projector.project(-g_radar_points[i].x, g_radar_points[i].y, 0, -7);
      double v_max = -1;
      int index = -1, class_id = -1;
      for (size_t j = 0; j < g_camera_objects.objects.size(); j++)
      {
        double scale_u = 608.0 / 1920.0;
        double scale_v = 384.0 / 1208.0;
        double u = g_camera_objects.objects[j].camInfo[0].u;
        double v = g_camera_objects.objects[j].camInfo[0].v;
        double width = g_camera_objects.objects[j].camInfo[0].width;
        double height = g_camera_objects.objects[j].camInfo[0].height;
        double center_u = (u + width / 2) * scale_u;
        double center_v = (v + height / 2) * scale_v;
        double v_max_temp = (v + height) * scale_v;
        double radius = sqrt(pow(width / 2 * scale_u, 2) + pow(height / 2 * scale_v, 2)) * 1.2;
        double distance = sqrt(pow(result[0] - center_u, 2) + pow(result[1] - center_v, 2));
        if (distance <= radius && v_max_temp > v_max)
        {
          v_max = v_max_temp;
          index = j;
        }
      }
      if (index != -1)
      {
        class_id = g_camera_objects.objects[index].classId;
        //delphi x front y right
        fusion_result[index][0] = g_radar_points[i].x;
        fusion_result[index][1] = g_radar_points[i].y;
        fusion_result[index][2] = result[0];
        fusion_result[index][3] = result[1];
        fusion_result[index][4] = class_id;
        if (class_id == 2 || class_id == 3)
        {
          i--;
        }
        //g_camera_objects.erase(g_camera_objects.begin() + index);
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

  g_projector.init();

  image_transport::ImageTransport it(n);
  image_transport::Subscriber sub_image2 = it.subscribe("/cam/front_bottom_60/detect_image", 1, callbackCamera);

  ros::Subscriber camera_objects_sub = n.subscribe("/cam_obj/front_bottom_60", 1, callbackCameraObjects);

  ros::Subscriber rad_front_sub = n.subscribe("/RadFront", 1, callbackRadar);

  g_fusion_result_pub = n.advertise<msgs::DetectedObjectArray>("/cam_obj/front_bottom_60/radar_fusion", 1);
  g_fusion_result_pointcloud_pub = n.advertise<sensor_msgs::PointCloud2>("/cam_obj/front_bottom_60/radar_fusion/pointcloud", 1);

  std::vector<double> camera_angle, radar_angle;
  while (ros::ok())
  {
    //0:1, 1:0.1, 2:0.1, 3:0.1, 4:1, 5:1
    cv::Mat m_mid_temp;
    g_m_mid.copyTo(m_mid_temp);
    cv::resize(m_mid_temp, m_mid_temp, cv::Size(608, 384), 0, 0, cv::INTER_LINEAR);
    //double scaleFactor = m_mid_temp.rows / 384;
    cv::circle(m_mid_temp, cv::Point(GlobalVariable::UI_PARA[0], GlobalVariable::UI_PARA[1] * 10), 3, CV_RGB(255, 0, 0), -1, 8, 0);
    if (GlobalVariable::UI_PARA[5] < 0 && GlobalVariable::UI_TESTING_BUTTOM)
    {
      GlobalVariable::UI_TESTING_BUTTOM = false;
      //double h_camera, double x_p, double y_p, double x_cw, double y_cw, double z_cw
      camera_angle = g_projector.calculateCameraAngle(1.94, GlobalVariable::UI_PARA[0], GlobalVariable::UI_PARA[1] * 10, GlobalVariable::UI_PARA[2]/10, GlobalVariable::UI_PARA[3]/10,  GlobalVariable::UI_PARA[4]/100, false);
    }else if (GlobalVariable::UI_PARA[5] > 0 && GlobalVariable::UI_TESTING_BUTTOM)
    {
      GlobalVariable::UI_TESTING_BUTTOM = false;
      //double camera_alpha, double camera_beta, double h_camera, double h_r, double x_p, double y_p, double x_r, double y_r, double L_x, double L_y
      radar_angle = g_projector.calculateRadarAngle(camera_angle[0], camera_angle[1], 1.96, 0.86, GlobalVariable::UI_PARA[4]/100, GlobalVariable::UI_PARA[0], GlobalVariable::UI_PARA[1] * 10, GlobalVariable::UI_PARA[2]/10, GlobalVariable::UI_PARA[3]/10, 0, -7, false);
    }
/*
    if (!radar_angle.empty()) 
    {
      for (size_t i = 0; i < g_radar_points.size(); i++)
      {
        std::vector<int> result = g_projector.project(camera_angle[0], camera_angle[1], 1.96, radar_angle[0], radar_angle[1], 0.86, GlobalVariable::UI_PARA[4]/100, -g_radar_points[i].x, g_radar_points[i].y, 0, -7, false);
        cv::circle(m_mid_temp, cv::Point(result[0], result[1]), 3, CV_RGB(0, 255, 0), -1, 8, 0);
        std::vector<int> test = g_projector.project(camera_angle[0], camera_angle[1], 1.96, radar_angle[0], radar_angle[1], 0.86, GlobalVariable::UI_PARA[4]/100, GlobalVariable::UI_PARA[2]/10, GlobalVariable::UI_PARA[3]/10, 0, -7, false);
        cv::circle(m_mid_temp, cv::Point(test[0], test[1]), 3, CV_RGB(0, 0, 255), -1, 8, 0);
      }
    }
*/
    std::vector<std::vector<double>> fusion_result = fusion();
    if (!fusion_result.empty())
    {
      for  (auto & result : fusion_result)
      {
        if (result[2] >= 0 && result[3] >= 0 && result[2] < m_mid_temp.cols && result[3] < m_mid_temp.rows)
        {
          int red_int = 255, gre_int = 0, blu_int = 0;
          double depths_float = result[0];
          std::stringstream ss;
          ss << std::setprecision(3) << abs(depths_float);
          string distance = ss.str();
          cv::circle(m_mid_temp, cv::Point(result[2], result[3]), 6, CV_RGB(red_int, gre_int, blu_int), -1, 8, 0);
          cv::putText(m_mid_temp, distance, cv::Point(result[2] + 3, result[3]), cv::FONT_HERSHEY_DUPLEX, 0.6, CV_RGB(red_int, gre_int, blu_int), 2);
        }
      }
      sendFusionResult(fusion_result);
    }
/*
    for (size_t i = 0; i < g_radar_points.size(); i++)
    {
      //delphi x front y right
      std::vector<int> result = g_projector.project(g_radar_points[i].y, g_radar_points[i].x, 0, 0);
      //mingtai y front x left
      //std::vector<int> result = g_projector.project(-g_radar_points[i].x, g_radar_points[i].y, 0, -7);
      if (result[0] >= 0 && result[1] >= 0 && result[0] < m_mid_temp.cols && result[1] < m_mid_temp.rows)
      {
        int red_int = 255, gre_int = 0, blu_int = 0;
        double depths_float = (double)g_radar_points[i].x;
        std::stringstream ss;
        ss << std::setprecision(3) << abs(depths_float);
        string distance = ss.str();
        cv::circle(m_mid_temp, cv::Point(result[0], result[1]), 6, CV_RGB(red_int, gre_int, blu_int), -1, 8, 0);
        cv::putText(m_mid_temp, distance, cv::Point(result[0] + 3, result[1]), cv::FONT_HERSHEY_DUPLEX, 0.6, CV_RGB(red_int, gre_int, blu_int), 2);
      }
    }
*/
    cv::namedWindow("Alignment_view", cv::WINDOW_NORMAL);
    cv::imshow("Alignment_view", m_mid_temp);
    cv::waitKey(1);

    ros::spinOnce();
  }
}

void tracking_test(int argc, char** argv)
{
  ros::init(argc, argv, "LidFrontTop_idsXC2_fusion");
  ros::NodeHandle n;

  image_transport::ImageTransport it(n);
  image_transport::Subscriber sub_image2 = it.subscribe("/cam/front_bottom_60/detect_image", 1, callbackCamera);

  ros::Subscriber camera_objects_sub = n.subscribe("/cam_obj/front_bottom_60", 1, callbackCameraObjects);


  g_tracking_result_pub = n.advertise<msgs::DetectedObjectArray>("/cam_obj/front_bottom_60/tracking_result", 1);
  Tracker tracker;
  ros::Rate r(10);
  while (ros::ok())
  {
    cv::Mat m_mid_temp;
    g_m_mid.copyTo(m_mid_temp);
    cv::resize(m_mid_temp, m_mid_temp, cv::Size(608, 384), 0, 0, cv::INTER_LINEAR);
    std::cout << "before tracking: " << g_camera_objects.objects.size() << std::endl;
    msgs::DetectedObjectArray tracking_result = tracker.tracking(g_camera_objects);
    g_camera_objects.objects.clear();
    std::cout << "after tracking: " << tracking_result.objects.size() << std::endl;
    double u_mag = 608.0 / 1280.0;
    double v_mag = 384.0 / 720.0;
    for (auto & object : tracking_result.objects)
    {
      double u = object.camInfo[0].u;
      double v = object.camInfo[0].v;
      double width = object.camInfo[0].width;
      double height = object.camInfo[0].height;
      cv::Point pt1(u * u_mag, v * v_mag);
      cv::Point pt2((u + width) * u_mag, (v + height) * v_mag);
      cv::rectangle(m_mid_temp, pt1, pt2, cv::Scalar(0, 0, 255),2,1,0);
      std::stringstream ss;
      ss << abs(object.camInfo[0].id);
      string id = ss.str();
      cv::putText(m_mid_temp, id, cv::Point(u * u_mag, v * v_mag), cv::FONT_HERSHEY_DUPLEX,
                  0.6,  cv::Scalar(0, 0, 255), 2);
    }
    g_tracking_result_pub.publish(tracking_result);
    cv::namedWindow("Alignment_view", cv::WINDOW_NORMAL);
    cv::imshow("Alignment_view", m_mid_temp);
    cv::waitKey(1);

    r.sleep();
    ros::spinOnce();
  }
}

int main(int argc, char** argv)
{
  thread thead_detection(tracking_test, argc, argv);
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
