#include "sync_message.h"

cv::Mat getSpecificTimeCameraMessage(message_filters::Cache<sensor_msgs::Image>& cache_image, ros::Time target_time,
                                     const ros::Duration& duration_time)
{
  ros::Time begin_time = target_time - duration_time;
  ros::Time end_time = target_time + duration_time;
  std::vector<sensor_msgs::Image::ConstPtr> images = cache_image.getInterval(begin_time, end_time);
  cv::Mat out_mat;
  if (!images.empty())
  {
    std::vector<ros::Time> images_time(images.size());
    for (size_t index = 0; index < images.size(); index++)
    {
      images_time[index] = images[index]->header.stamp;
    }
    std::vector<ros::Time>::iterator it;
    it = std::find(images_time.begin(), images_time.end(), target_time);
    if (it != images_time.end())
    {
      int time_index = std::distance(images_time.begin(), it);
      cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvCopy(images[time_index], sensor_msgs::image_encodings::BGR8);
      out_mat = cv_ptr->image;
    }
    else if (images.size() == 1)
    {
      cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvCopy(images[0], sensor_msgs::image_encodings::BGR8);
      out_mat = cv_ptr->image;
    }
    else
    {
      std::cout << "Not found the same timestamp in camera buffer." << std::endl;
    }
  }
  else
  {
    std::cout << "Not found any message in camera buffer." << std::endl;
  }
  return out_mat;
}

pcl::PointCloud<pcl::PointXYZI>::Ptr
getSpecificTimeLidarMessage(message_filters::Cache<pcl::PointCloud<pcl::PointXYZI>>& cache_lidar, ros::Time target_time,
                            const ros::Duration& duration_time)
{
  ros::Time begin_time = target_time - duration_time;
  ros::Time end_time = target_time + duration_time;
  std::vector<pcl::PointCloud<pcl::PointXYZI>::ConstPtr> lidars = cache_lidar.getInterval(begin_time, end_time);
  pcl::PointCloud<pcl::PointXYZI>::Ptr lidar_ptr(new pcl::PointCloud<pcl::PointXYZI>);

  if (!lidars.empty())
  {
    std::vector<ros::Time> lidars_time;
    for (const auto& msg : lidars)
    {
      ros::Time header_time;
      pcl_conversions::fromPCL(msg->header.stamp, header_time);
      lidars_time.push_back(header_time);
    }
    std::vector<ros::Time>::iterator it;
    it = std::find(lidars_time.begin(), lidars_time.end(), target_time);
    if (it != lidars_time.end())
    {
      int time_index = std::distance(lidars_time.begin(), it);
      *lidar_ptr = *lidars[time_index];
    }
    else if (lidars.size() == 1)
    {
      *lidar_ptr = *lidars[0];
    }
    else
    {
      std::cout << "Not found the same timestamp in lidar buffer." << std::endl;
    }
  }
  else
  {
    std::cout << "Not found any message in lidar buffer." << std::endl;
  }
  return lidar_ptr;
}
pcl::PointCloud<pcl::PointXYZIL>::Ptr
getSpecificTimeLidarMessage(message_filters::Cache<pcl::PointCloud<pcl::PointXYZIL>>& cache_lidar,
                            ros::Time target_time, const ros::Duration& duration_time)
{
  ros::Time begin_time = target_time - duration_time;
  ros::Time end_time = target_time + duration_time;
  std::vector<pcl::PointCloud<pcl::PointXYZIL>::ConstPtr> lidars = cache_lidar.getInterval(begin_time, end_time);
  pcl::PointCloud<pcl::PointXYZIL>::Ptr lidar_ptr(new pcl::PointCloud<pcl::PointXYZIL>);

  if (!lidars.empty())
  {
    std::vector<ros::Time> lidars_time;
    for (const auto& msg : lidars)
    {
      ros::Time header_time;
      pcl_conversions::fromPCL(msg->header.stamp, header_time);
      lidars_time.push_back(header_time);
    }
    std::vector<ros::Time>::iterator it;
    it = std::find(lidars_time.begin(), lidars_time.end(), target_time);
    if (it != lidars_time.end())
    {
      int time_index = std::distance(lidars_time.begin(), it);
      *lidar_ptr = *lidars[time_index];
    }
    else if (lidars.size() == 1)
    {
      *lidar_ptr = *lidars[0];
    }
    else
    {
      std::cout << "Not found the same timestamp in lidar buffer." << std::endl;
    }
  }
  else
  {
    std::cout << "Not found any message in lidar buffer." << std::endl;
  }
  return lidar_ptr;
}