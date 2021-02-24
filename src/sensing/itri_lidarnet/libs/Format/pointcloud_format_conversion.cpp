#include <pcl_conversions/pcl_conversions.h>
#include "pointcloud_format_conversion.h"

pcl::PointCloud<pcl::PointXYZIR> SensorMsgs_to_XYZIR(const sensor_msgs::PointCloud2& cloud_msg, lidar::Hardware /*brand*/)
{
  pcl::PointCloud<pcl::PointXYZIR> cloud;

  // Get the field structure of this point cloud
  int point_bytes = cloud_msg.point_step;
  int offset_x = 0;
  int offset_y = 0;
  int offset_z = 0;
  int offset_int = 0;
  int offset_ring = 0;
  for (size_t f = 0; f < cloud_msg.fields.size(); ++f)
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
  for (size_t p = 0, bound = cloud_msg.width * cloud_msg.height, point_offset = 0; p < bound;
       ++p, point_offset += point_bytes)
  {
    pcl::PointXYZIR new_point;

    new_point.x = *(float*)(&cloud_msg.data[0] + point_offset + offset_x);
    new_point.y = *(float*)(&cloud_msg.data[0] + point_offset + offset_y);
    new_point.z = *(float*)(&cloud_msg.data[0] + point_offset + offset_z);
    new_point.intensity = *(float*)(&cloud_msg.data[0] + point_offset + offset_int);
    new_point.ring = *(unsigned char*)(&cloud_msg.data[0] + point_offset + offset_ring);

    // if (brand == "ouster")
    // {
    //   new_point.intensity = new_point.intensity * 255 / 500;  // normalize

    //   if (new_point.intensity > 255)
    //   {
    //     new_point.intensity = 255;
    //   }
    // }
    cloud.points.emplace_back(new_point);
  }

  pcl_conversions::toPCL(cloud_msg.header, cloud.header);

  return cloud;
}

//----------------------RGB
pcl::PointCloud<pcl::PointXYZRGB> XYZIR_to_XYZRGB(pcl::PointCloud<pcl::PointXYZIR>::Ptr input_cloud)
{
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr output_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);

  for (size_t p = 0; p < input_cloud->size(); ++p)
  {
    pcl::PointXYZRGB new_point;
    new_point.x = input_cloud->points[p].x;
    new_point.y = input_cloud->points[p].y;
    new_point.z = input_cloud->points[p].z;

    new_point.r = (uint8_t)input_cloud->points[p].intensity;
    new_point.g = (uint8_t)input_cloud->points[p].ring;
    new_point.b = 0;

    output_cloud->points.emplace_back(new_point);
  }

  output_cloud->header = input_cloud->header;
  return *output_cloud;
}

pcl::PointCloud<pcl::PointXYZIR> XYZRGB_to_XYZIR(pcl::PointCloud<pcl::PointXYZRGB>::Ptr input_cloud)
{
  pcl::PointCloud<pcl::PointXYZIR>::Ptr output_cloud(new pcl::PointCloud<pcl::PointXYZIR>);

  for (size_t p = 0; p < input_cloud->size(); ++p)
  {
    pcl::PointXYZIR new_point;
    new_point.x = input_cloud->points[p].x;
    new_point.y = input_cloud->points[p].y;
    new_point.z = input_cloud->points[p].z;

    uint8_t new_r = input_cloud->points[p].r;
    uint8_t new_g = input_cloud->points[p].g;
    // uint8_t new_b = input_cloud->points[p].b;

    new_point.intensity = unsigned(new_r);
    new_point.ring = unsigned(new_g);

    output_cloud->points.emplace_back(new_point);
  }
  output_cloud->header = input_cloud->header;

  return *output_cloud;
}
