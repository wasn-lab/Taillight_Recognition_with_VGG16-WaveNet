#include <pcl_conversions/pcl_conversions.h>
#include "pointcloud_format_conversion.h"

pcl::PointCloud<pcl::PointXYZIR>::Ptr SensorMsgs_to_XYZIR(const sensor_msgs::PointCloud2& cloud_msg, lidar::Hardware /*brand*/)
{
  pcl::PointCloud<pcl::PointXYZIR>::Ptr cloud_ptr{new pcl::PointCloud<pcl::PointXYZIR>};
  auto& cloud = *cloud_ptr;

  // Get the field structure of this point cloud
  int point_bytes = cloud_msg.point_step;
  int offset_x = 0;
  int offset_y = 0;
  int offset_z = 0;
  int offset_int = 0;
  int offset_ring = 0;
  for (const auto& field: cloud_msg.fields)
  {
    if (field.name == "x")
    {
      offset_x = field.offset;
    }
    else if (field.name == "y")
    {
      offset_y = field.offset;
    }
    else if (field.name == "z")
    {
      offset_z = field.offset;
    }
    else if (field.name == "intensity")
    {
      offset_int = field.offset;
    }
    else if (field.name == "ring")
    {
      offset_ring = field.offset;
    }
  }

  // populate point cloud object
  cloud.points.resize(cloud_msg.width * cloud_msg.height);
  for (size_t p = 0, bound = cloud_msg.width * cloud_msg.height, point_offset = 0; p < bound;
       ++p, point_offset += point_bytes)
  {
    const auto base_addr = &cloud_msg.data[0] + point_offset;
    cloud.points[p].x = *(float*)(base_addr + offset_x);
    cloud.points[p].y = *(float*)(base_addr + offset_y);
    cloud.points[p].z = *(float*)(base_addr + offset_z);
    cloud.points[p].intensity = *(float*)(base_addr + offset_int);
    cloud.points[p].ring = *(unsigned char*)(base_addr + offset_ring);

    // if (brand == "ouster")
    // {
    //   new_point.intensity = new_point.intensity * 255 / 500;  // normalize

    //   if (new_point.intensity > 255)
    //   {
    //     new_point.intensity = 255;
    //   }
    // }
  }

  pcl_conversions::toPCL(cloud_msg.header, cloud.header);

  return cloud_ptr;
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
