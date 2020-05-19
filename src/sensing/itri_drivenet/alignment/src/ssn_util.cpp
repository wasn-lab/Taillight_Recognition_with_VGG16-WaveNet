#include "ssn_util.h"

/// namespace
using namespace DriveNet;

int transferCommonLabelToSSNLabel(common_type_id label_id)
{
  int nn_label_id = -1;
  switch (label_id)
  {
    case common_type_id::person:
      nn_label_id = nnClassID::Person;  // Person
      break;
    case common_type_id::bicycle:
      nn_label_id = nnClassID::Motobike;  // bicycle
      break;
    case common_type_id::motorbike:
      nn_label_id = nnClassID::Motobike;  // motobike
      break;
    case common_type_id::car:
      nn_label_id = nnClassID::Car;  // car
      break;
    case common_type_id::bus:
      nn_label_id = nnClassID::Car;  // bus
      break;
    case common_type_id::truck:
      nn_label_id = nnClassID::Car;  // truck
      break;
    default:
      break;
  }
  return nn_label_id;
}
pcl::PointCloud<pcl::PointXYZI>::Ptr getClassObjectPoint(const pcl::PointCloud<pcl::PointXYZIL>::Ptr& points_ptr,
                                                         common_type_id label_id)
{
  pcl::PointCloud<pcl::PointXYZI>::Ptr out_points_ptr(new pcl::PointCloud<pcl::PointXYZI>);
  int nn_label_id = -1;
  nn_label_id = transferCommonLabelToSSNLabel(label_id);
  for (size_t index = 0; index < points_ptr->size(); index++)
  {
    if (points_ptr->points[index].label == nn_label_id)
    {
      pcl::PointXYZI point;
      point.x = points_ptr->points[index].x;
      point.y = points_ptr->points[index].y;
      point.z = points_ptr->points[index].z;
      point.intensity = points_ptr->points[index].intensity;
      out_points_ptr->push_back(point);
    }
  }
  return out_points_ptr;
}
pcl::PointCloud<pcl::PointXYZI>::Ptr getClassObjectPoint(const pcl::PointCloud<pcl::PointXYZIL>::Ptr& points_ptr,
                                                         nnClassID label_id)
{
  pcl::PointCloud<pcl::PointXYZI>::Ptr out_points_ptr(new pcl::PointCloud<pcl::PointXYZI>);
  int nn_label_id = -1;
  nn_label_id = label_id;
  for (size_t index = 0; index < points_ptr->size(); index++)
  {
    if (points_ptr->points[index].label == nn_label_id)
    {
      pcl::PointXYZI point;
      point.x = points_ptr->points[index].x;
      point.y = points_ptr->points[index].y;
      point.z = points_ptr->points[index].z;
      point.intensity = points_ptr->points[index].intensity;
      out_points_ptr->push_back(point);
    }
  }
  return out_points_ptr;
}