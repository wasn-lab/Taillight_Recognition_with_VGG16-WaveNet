#include "roi_fusion.h"
#include "drivenet/object_label_util.h"

using namespace sensor_msgs_itri;

std::vector<std::vector<sensor_msgs::RegionOfInterest>>
RoiFusion::getLidar2DROI(const std::vector<std::vector<DriveNet::MinMax2D>>& cam_pixels_obj)
{
  std::vector<std::vector<sensor_msgs::RegionOfInterest>> object_roi(cam_pixels_obj.size());
  for (size_t cam_order = 0; cam_order < cam_pixels_obj.size(); cam_order++)
  {
    for (const auto& pixel_obj : cam_pixels_obj[cam_order])
    {
      sensor_msgs::RegionOfInterest roi;
      roi.x_offset = pixel_obj.p_min.u;
      roi.y_offset = pixel_obj.p_min.v;
      roi.width = pixel_obj.p_max.u - pixel_obj.p_min.u;
      roi.height = std::abs(pixel_obj.p_max.v - pixel_obj.p_min.v);
      object_roi[cam_order].push_back(roi);
    }
  }
  return object_roi;
}
std::vector<std::vector<sensor_msgs::RegionOfInterest>>
RoiFusion::getCam2DROI(const std::vector<msgs::DetectedObjectArray>& objects_array)
{
  std::vector<std::vector<sensor_msgs::RegionOfInterest>> object_roi(objects_array.size());
  for (size_t cam_order = 0; cam_order < objects_array.size(); cam_order++)
  {
    for (const auto& obj : objects_array[cam_order].objects)
    {
      std::vector<DriveNet::PixelPosition> pixel_positions(2);
      pixel_positions[0].u = obj.camInfo[0].u;
      pixel_positions[0].v = obj.camInfo[0].v;
      pixel_positions[1].u = obj.camInfo[0].u + obj.camInfo[0].width;
      pixel_positions[1].v = obj.camInfo[0].v + obj.camInfo[0].height;
      transferPixelScaling(pixel_positions);

      sensor_msgs::RegionOfInterest roi;
      roi.x_offset = pixel_positions[0].u;
      roi.y_offset = pixel_positions[0].v;
      roi.width = pixel_positions[1].u - pixel_positions[0].u;
      roi.height = pixel_positions[1].v - pixel_positions[0].v;
      object_roi[cam_order].push_back(roi);
    }
  }
  return object_roi;
}
std::vector<std::vector<std::pair<int, int>>>
RoiFusion::getRoiFusionResult(const std::vector<std::vector<sensor_msgs::RegionOfInterest>>& object_camera_roi,
                              const std::vector<std::vector<sensor_msgs::RegionOfInterest>>& object_lidar_roi,
                              const std::vector<std::vector<DetectedObjectClassId>>& object_camera_class_id,
                              const std::vector<std::vector<DetectedObjectClassId>>& object_lidar_class_id)
{
  std::vector<std::vector<std::pair<int, int>>> fusion_index(object_camera_roi.size());
  for (size_t cam_order = 0; cam_order < object_camera_roi.size(); cam_order++)
  {
    for (size_t cam_index = 0; cam_index < object_camera_roi[cam_order].size(); cam_index++)
    {
      int max_iou_cam_index = -1;
      int max_iou_lidar_index = -1;
      double max_iou = 0.0;
      for (size_t lidar_index = 0; lidar_index < object_lidar_roi[cam_order].size(); lidar_index++)
      {
        if(object_camera_class_id[cam_order][cam_index] == object_lidar_class_id[cam_order][lidar_index])
        {
          double iou(0.0), iou_x(0.0), iou_y(0.0);
          if (roi_fusion_nodelet.use_iou_)
          {
            iou = roi_fusion_nodelet.calcIoU(object_lidar_roi[cam_order][lidar_index],
                                            object_camera_roi[cam_order][cam_index]);
          }
          if (roi_fusion_nodelet.use_iou_x_)
          {
            iou_x = roi_fusion_nodelet.calcIoUX(object_lidar_roi[cam_order][lidar_index],
                                                object_camera_roi[cam_order][cam_index]);
          }
          if (roi_fusion_nodelet.use_iou_y_)
          {
            iou_y = roi_fusion_nodelet.calcIoUY(object_lidar_roi[cam_order][lidar_index],
                                                object_camera_roi[cam_order][cam_index]);
          }
          if (max_iou < iou + iou_x + iou_y)
          {
            max_iou_lidar_index = lidar_index;
            max_iou_cam_index = cam_index;
            max_iou = iou + iou_x + iou_y;
          }
        }
      }

      if (roi_fusion_nodelet.iou_threshold_ < max_iou)
      {
        fusion_index[cam_order].push_back(std::pair<int, int>(max_iou_cam_index, max_iou_lidar_index));
      }
    }
  }
  return fusion_index;
}
void RoiFusion::getFusionCamObj(const std::vector<msgs::DetectedObjectArray>& objects_array,
                                const std::vector<std::vector<std::pair<int, int>>> fusion_index,
                                std::vector<std::vector<DriveNet::MinMax2D>>& cam_pixels_obj)
{
  for (size_t cam_order = 0; cam_order < objects_array.size(); cam_order++)
  {
    for (size_t pair_index = 0; pair_index < fusion_index[cam_order].size(); pair_index++)
    {
      int camera_index = fusion_index[cam_order][pair_index].first;

      std::vector<DriveNet::PixelPosition> pixel_positions(2);
      pixel_positions[0].u = objects_array[cam_order].objects[camera_index].camInfo[0].u;
      pixel_positions[0].v = objects_array[cam_order].objects[camera_index].camInfo[0].v;
      pixel_positions[1].u = objects_array[cam_order].objects[camera_index].camInfo[0].u +
                             objects_array[cam_order].objects[camera_index].camInfo[0].width;
      pixel_positions[1].v = objects_array[cam_order].objects[camera_index].camInfo[0].v +
                             objects_array[cam_order].objects[camera_index].camInfo[0].height;
      transferPixelScaling(pixel_positions);

      DriveNet::MinMax2D min_max_2d_bbox;
      min_max_2d_bbox.p_min.u = pixel_positions[0].u;
      min_max_2d_bbox.p_min.v = pixel_positions[0].v;
      min_max_2d_bbox.p_max.u = pixel_positions[1].u;
      min_max_2d_bbox.p_max.v = pixel_positions[1].v;
      cam_pixels_obj[cam_order].push_back(min_max_2d_bbox);
    }
  }
}

std::vector<std::vector<DetectedObjectClassId>>
RoiFusion::getCamObjSpecialClassId(const std::vector<msgs::DetectedObjectArray>& objects_array)
{
  std::vector<std::vector<DetectedObjectClassId>> object_class_id(objects_array.size());
  DetectedObjectClassId class_id = DetectedObjectClassId::Unknown; // LidarNet SpecialClassId: Person, Motobike, Car
  for (size_t cam_order = 0; cam_order < objects_array.size(); cam_order++)
  {
    for (const auto& obj : objects_array[cam_order].objects)
    {
      switch (obj.classId)
      {
        case (DetectedObjectClassId::Person):
          class_id = DetectedObjectClassId::Person;
          break;
        case (DetectedObjectClassId::Bicycle):
          class_id = DetectedObjectClassId::Motobike;
          break;
        case (DetectedObjectClassId::Motobike):
          class_id = DetectedObjectClassId::Motobike;
          break;
        case (DetectedObjectClassId::Car):
          class_id = DetectedObjectClassId::Car;
          break;
        case (DetectedObjectClassId::Bus):
          class_id = DetectedObjectClassId::Car;
          break;
        case (DetectedObjectClassId::Truck):
          class_id = DetectedObjectClassId::Car;
          break;
      }
      object_class_id[cam_order].push_back(class_id);
    }
  }
  return object_class_id;
}

std::vector<std::vector<DetectedObjectClassId>>
RoiFusion::getLidarObjSpecialClassId(const std::vector<std::vector<msgs::DetectedObject>>& objects_array)
{
  std::vector<std::vector<DetectedObjectClassId>> object_class_id(objects_array.size());
  for (size_t cam_order = 0; cam_order < objects_array.size(); cam_order++)
  {
    for (const auto& obj : objects_array[cam_order])
    {
      object_class_id[cam_order].push_back(static_cast<DetectedObjectClassId>(obj.classId));
    }
  }
  return object_class_id;
}