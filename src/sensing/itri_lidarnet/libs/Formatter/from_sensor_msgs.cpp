#include "from_sensor_msgs.h"





pcl::PointCloud<pcl::PointXYZIR> get_ring_pcl_from_sensor_msgs(const sensor_msgs::PointCloud2 & cloud_msg)
{
  pcl::PointCloud<pcl::PointXYZIR> cloud;

  // Get the field structure of this point cloud
  int pointBytes = cloud_msg.point_step;
  int offset_x;
  int offset_y;
  int offset_z;
  int offset_int;
  int offset_ring;
  for (int f=0; f<cloud_msg.fields.size(); ++f)
  {
    if (cloud_msg.fields[f].name == "x")
      offset_x = cloud_msg.fields[f].offset;
    if (cloud_msg.fields[f].name == "y")
      offset_y = cloud_msg.fields[f].offset;
    if (cloud_msg.fields[f].name == "z")
      offset_z = cloud_msg.fields[f].offset;
    if (cloud_msg.fields[f].name == "intensity")
      offset_int = cloud_msg.fields[f].offset;
    if (cloud_msg.fields[f].name == "ring")
      offset_ring = cloud_msg.fields[f].offset;
  }

  // populate point cloud object
  for (int p=0; p< (cloud_msg.width * cloud_msg.height); ++p)
  {
      pcl::PointXYZIR newPoint;

      newPoint.x = *(float*)(&cloud_msg.data[0] + (pointBytes*p) + offset_x);
      newPoint.y = *(float*)(&cloud_msg.data[0] + (pointBytes*p) + offset_y);
      newPoint.z = *(float*)(&cloud_msg.data[0] + (pointBytes*p) + offset_z);
      newPoint.intensity = *(float*)(&cloud_msg.data[0] + (pointBytes*p) + offset_int);
      newPoint.ring = *(unsigned char*)(&cloud_msg.data[0] + (pointBytes*p) + offset_ring);

      cloud.points.push_back(newPoint);
  }

  pcl_conversions::toPCL(cloud_msg.header, cloud.header);

  return cloud;
}


pcl::RangeImage PointCloudtoRangeImage(pcl::PointCloud<pcl::PointXYZIR>::Ptr input_cloud, std::string lidar_brand, int ring_num)
{
  // pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud(new pcl::PointCloud<pcl::PointXYZ>);
  // pcl::copyPointCloud(*input, *input_cloud);
  
  if (lidar_brand == "ouster")
  {
    if (ring_num == 64)
    {
      input_cloud->width = input_cloud->size() / 64;
      input_cloud->height = 64;  

      // We now want to create a range image from the above point cloud, with a 1deg angular resolution
      float angularResolution = (float) ( 0.4f * (M_PI/180.0f));  //  1.0 degree in radians
      float maxAngleWidth    = (float) (360.0f * (M_PI/180.0f));  // 360.0 degree in radians
      float maxAngleHeight    = (float) (33.0f * (M_PI/180.0f));  // 180.0 degree in radians
      Eigen::Affine3f sensorPose = (Eigen::Affine3f)Eigen::Translation3f(0.0f, 0.0f, 0.0f);
      pcl::RangeImage::CoordinateFrame coordinate_frame = pcl::RangeImage::LASER_FRAME;
      float noiseLevel=0.00;
      float minRange = 0.3f;
      int borderSize = 1;

      pcl::RangeImage rangeImage;
      rangeImage.createFromPointCloud(*input_cloud, angularResolution, maxAngleWidth, maxAngleHeight,
                                      sensorPose, coordinate_frame, noiseLevel, minRange, borderSize);
      
      rangeImage.header = input_cloud->header;

      std::cout << rangeImage << "\n";
      return rangeImage;

    }
  }
  if (lidar_brand == "velodyne")
  {
    if (ring_num == 32)
    {
      input_cloud->width = input_cloud->size() /32 ;
      input_cloud->height = 32;
      // We now want to create a range image from the above point cloud, with a 1deg angular resolution
      float angularResolution = (float) ( 0.01f * (M_PI/180.0f));  //  1.0 degree in radians
      float maxAngleWidth    = (float) (360.0f * (M_PI/180.0f));  // 360.0 degree in radians
      float maxAngleHeight    = (float) (40.0f * (M_PI/180.0f));  // 180.0 degree in radians
      Eigen::Affine3f sensorPose = (Eigen::Affine3f)Eigen::Translation3f(0.0f, 0.0f, 0.0f);
      pcl::RangeImage::CoordinateFrame coordinate_frame = pcl::RangeImage::LASER_FRAME;
      float noiseLevel=0.00;
      float minRange = 0.4f;
      int borderSize = 1;

      pcl::RangeImage rangeImage;
      rangeImage.createFromPointCloud(*input_cloud, angularResolution, maxAngleWidth, maxAngleHeight,
                                      sensorPose, coordinate_frame, noiseLevel, minRange, borderSize);
      rangeImage.header = input_cloud->header;
      std::cout << rangeImage << "\n";
      return rangeImage;  

    }
  }
}
  


