
#include "all_header.h"
#include "pointcloud_format_conversion.h"
#include "msgs/CompressedPointCloud.h"
#include <sensor_msgs/PointCloud2.h>

//---------------------------- Publisher
// no-filter
ros::Publisher pub_LidarFrontLeft;
ros::Publisher pub_LidarFrontRight;
ros::Publisher pub_LidarFrontTop;

mutex L_Lock;
mutex R_Lock;
mutex T_Lock;

//------------------------------ Callback
void cloud_cb_LidarFrontLeft(msgs::CompressedPointCloud msg)
{
  L_Lock.lock();
  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloudOut(new pcl::PointCloud<pcl::PointXYZRGBA>());
  
  stringstream msg_data_ss;

  msg_data_ss << msg.data; 

  pcl::io::OctreePointCloudCompression<pcl::PointXYZRGBA>* PointCloudDecoder;
  PointCloudDecoder = new pcl::io::OctreePointCloudCompression<pcl::PointXYZRGBA>();
  PointCloudDecoder->decodePointCloud(msg_data_ss, cloudOut);

  pcl::PointCloud<pcl::PointXYZIR>::Ptr XYZIR_tmp(new pcl::PointCloud<pcl::PointXYZIR>);
  *XYZIR_tmp = XYZRBGA_to_XYZIR(cloudOut);

  pcl_conversions::toPCL(msg.header, XYZIR_tmp->header);

  sensor_msgs::PointCloud2::Ptr sensor_pc2 (new sensor_msgs::PointCloud2);

  pcl::toROSMsg(*XYZIR_tmp, *sensor_pc2);


  pub_LidarFrontLeft.publish(*sensor_pc2);

  msg_data_ss.clear();
  msg_data_ss.str("");


  delete (PointCloudDecoder);
  L_Lock.unlock();
}


void cloud_cb_LidarFrontRight(msgs::CompressedPointCloud msg)
{
  R_Lock.lock();
  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloudOut(new pcl::PointCloud<pcl::PointXYZRGBA>());
  
  stringstream msg_data_ss;

  msg_data_ss << msg.data; 

  pcl::io::OctreePointCloudCompression<pcl::PointXYZRGBA>* PointCloudDecoder;
  PointCloudDecoder = new pcl::io::OctreePointCloudCompression<pcl::PointXYZRGBA>();
  PointCloudDecoder->decodePointCloud(msg_data_ss, cloudOut);

  pcl::PointCloud<pcl::PointXYZIR>::Ptr XYZIR_tmp(new pcl::PointCloud<pcl::PointXYZIR>);
  *XYZIR_tmp = XYZRBGA_to_XYZIR(cloudOut);

  pcl_conversions::toPCL(msg.header, XYZIR_tmp->header);

  sensor_msgs::PointCloud2::Ptr sensor_pc2 (new sensor_msgs::PointCloud2);

  pcl::toROSMsg(*XYZIR_tmp, *sensor_pc2);

  pub_LidarFrontRight.publish(*sensor_pc2);

  msg_data_ss.clear();
  msg_data_ss.str("");


  delete (PointCloudDecoder);
  R_Lock.unlock();
}

void cloud_cb_LidarFrontTop(msgs::CompressedPointCloud msg)
{
  T_Lock.lock();
  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloudOut(new pcl::PointCloud<pcl::PointXYZRGBA>());
  
  stringstream msg_data_ss;

  msg_data_ss << msg.data; 

  pcl::io::OctreePointCloudCompression<pcl::PointXYZRGBA>* PointCloudDecoder;
  PointCloudDecoder = new pcl::io::OctreePointCloudCompression<pcl::PointXYZRGBA>();
  PointCloudDecoder->decodePointCloud(msg_data_ss, cloudOut);

  pcl::PointCloud<pcl::PointXYZIR>::Ptr XYZIR_tmp(new pcl::PointCloud<pcl::PointXYZIR>);
  *XYZIR_tmp = XYZRBGA_to_XYZIR(cloudOut);
  pcl_conversions::toPCL(msg.header, XYZIR_tmp->header);

  sensor_msgs::PointCloud2::Ptr sensor_pc2 (new sensor_msgs::PointCloud2);
  pcl::toROSMsg(*XYZIR_tmp, *sensor_pc2);


  pub_LidarFrontTop.publish(*sensor_pc2);
  msg_data_ss.clear();
  msg_data_ss.str("");


  delete (PointCloudDecoder);
  T_Lock.unlock();
}


int main(int argc, char** argv)
{
  ros::init(argc, argv, "lidars_decoder");
  ros::NodeHandle n;

  // subscriber
  ros::Subscriber sub_LidarFrontLeft =
      n.subscribe<msgs::CompressedPointCloud>("/LidarFrontLeft/Compressed", 1, cloud_cb_LidarFrontLeft);
  ros::Subscriber sub_LidarFrontRight =
      n.subscribe<msgs::CompressedPointCloud>("/LidarFrontRight/Compressed", 1, cloud_cb_LidarFrontRight);
  ros::Subscriber sub_LidarFrontTop =
      n.subscribe<msgs::CompressedPointCloud>("/LidarFrontTop/Compressed", 1, cloud_cb_LidarFrontTop);

  // publisher
  pub_LidarFrontLeft =
      n.advertise<const sensor_msgs::PointCloud2>("/LidarFrontLeft/Decompressed", 1);
  pub_LidarFrontRight =
      n.advertise<const sensor_msgs::PointCloud2>("/LidarFrontRight/Decompressed", 1);
  pub_LidarFrontTop =
      n.advertise<const sensor_msgs::PointCloud2>("/LidarFrontTop/Decompressed", 1);


  ros::AsyncSpinner spinner(3);
  spinner.start();

  // TheadDetection_Pub.join();
  ros::waitForShutdown();

  return 0;
}
