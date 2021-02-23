#include "pointcloud_format_conversion.h"

pcl::PointCloud<pcl::PointXYZIR> SensorMsgs_to_XYZIR(const sensor_msgs::PointCloud2& cloud_msg, string brand)
{
  pcl::PointCloud<pcl::PointXYZIR> cloud;

  // Get the field structure of this point cloud
  int pointBytes = cloud_msg.point_step;
  int offset_x;
  int offset_y;
  int offset_z;
  int offset_int;
  int offset_ring;
  for (int f = 0; f < cloud_msg.fields.size(); ++f)
  {
    if (cloud_msg.fields[f].name == "x")
    {
      offset_x = cloud_msg.fields[f].offset;
    }
    if (cloud_msg.fields[f].name == "y")
    {
      offset_y = cloud_msg.fields[f].offset;
    }
    if (cloud_msg.fields[f].name == "z")
    {
      offset_z = cloud_msg.fields[f].offset;
    }
    if (cloud_msg.fields[f].name == "intensity")
    {
      offset_int = cloud_msg.fields[f].offset;
    }
    if (cloud_msg.fields[f].name == "ring")
    {
      offset_ring = cloud_msg.fields[f].offset;
    }
  }

  // populate point cloud object
  for (int p = 0, bound = cloud_msg.width * cloud_msg.height; p < bound; ++p)
  {
    pcl::PointXYZIR newPoint;

    newPoint.x = *(float*)(&cloud_msg.data[0] + (pointBytes * p) + offset_x);
    newPoint.y = *(float*)(&cloud_msg.data[0] + (pointBytes * p) + offset_y);
    newPoint.z = *(float*)(&cloud_msg.data[0] + (pointBytes * p) + offset_z);
    newPoint.intensity = *(float*)(&cloud_msg.data[0] + (pointBytes * p) + offset_int);
    newPoint.ring = *(unsigned char*)(&cloud_msg.data[0] + (pointBytes * p) + offset_ring);

    // if (brand == "ouster")
    // {
    //   newPoint.intensity = newPoint.intensity * 255 / 500;  // normalize

    //   if (newPoint.intensity > 255)
    //   {
    //     newPoint.intensity = 255;
    //   }
    // }
    cloud.points.push_back(newPoint);
  }

  pcl_conversions::toPCL(cloud_msg.header, cloud.header);

  return cloud;
}

//----------------------RGB
pcl::PointCloud<pcl::PointXYZRGB> XYZIR_to_XYZRGB(pcl::PointCloud<pcl::PointXYZIR>::Ptr input_cloud)
{
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr output_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);

  for (int p = 0; p < (input_cloud->size()); ++p)
  {
    pcl::PointXYZRGB new_point;
    new_point.x = input_cloud->points[p].x;
    new_point.y = input_cloud->points[p].y;
    new_point.z = input_cloud->points[p].z;

    new_point.r = (uint8_t)input_cloud->points[p].intensity;
    new_point.g = (uint8_t)input_cloud->points[p].ring;
    new_point.b = 0;

    output_cloud->points.push_back(new_point);
  }

  output_cloud->header = input_cloud->header;
  return *output_cloud;
}

pcl::PointCloud<pcl::PointXYZIR> XYZRGB_to_XYZIR(pcl::PointCloud<pcl::PointXYZRGB>::Ptr input_cloud)
{
  pcl::PointCloud<pcl::PointXYZIR>::Ptr output_cloud(new pcl::PointCloud<pcl::PointXYZIR>);

  for (int p = 0; p < (input_cloud->size()); ++p)
  {
    pcl::PointXYZIR new_point;
    new_point.x = input_cloud->points[p].x;
    new_point.y = input_cloud->points[p].y;
    new_point.z = input_cloud->points[p].z;

    uint8_t new_r = input_cloud->points[p].r;
    uint8_t new_g = input_cloud->points[p].g;
    uint8_t new_b = input_cloud->points[p].b;

    new_point.intensity = unsigned(new_r);
    new_point.ring = unsigned(new_g);

    output_cloud->points.push_back(new_point);
  }
  output_cloud->header = input_cloud->header;

  return *output_cloud;
}
