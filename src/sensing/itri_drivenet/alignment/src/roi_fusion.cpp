#include "roi_fusion.h"
#include "drivenet/object_label_util.h"

std::vector<sensor_msgs::RegionOfInterest>
RoiFusion::getLidar2DROI(const std::vector<DriveNet::MinMax2D>& cam_pixels_obj)
{
  std::vector<sensor_msgs::RegionOfInterest> object_roi;
  for (const auto& pixel_obj : cam_pixels_obj)
  {
    sensor_msgs::RegionOfInterest roi;
    roi.x_offset = pixel_obj.p_min.u;
    roi.y_offset = pixel_obj.p_min.v;
    roi.width = pixel_obj.p_max.u - pixel_obj.p_min.u;
    roi.height = std::abs(pixel_obj.p_max.v - pixel_obj.p_min.v);
    object_roi.push_back(roi);
  }
  return object_roi;
}
std::vector<sensor_msgs::RegionOfInterest> RoiFusion::getCam2DROI(const msgs::DetectedObjectArray& objects_array)
{
  std::vector<sensor_msgs::RegionOfInterest> object_roi;
  for (const auto& obj : objects_array.objects)
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
    object_roi.push_back(roi);
  }
  return object_roi;
}
std::vector<std::pair<int, int>>
RoiFusion::getRoiFusionResult(const std::vector<sensor_msgs::RegionOfInterest>& object_camera_roi,
                              const std::vector<sensor_msgs::RegionOfInterest>& object_lidar_roi)
{
  std::vector<std::pair<int, int>> fusion_index;
  for (size_t cam_index = 0; cam_index < object_camera_roi.size(); cam_index++)
  {
    int max_iou_cam_index = -1;
    int max_iou_lidar_index = -1;
    double max_iou = 0.0;
    for (size_t lidar_index = 0; lidar_index < object_lidar_roi.size(); lidar_index++)
    {
      double iou(0.0), iou_x(0.0), iou_y(0.0);
      if (roi_fusion_nodelet.use_iou_)
      {
        iou = roi_fusion_nodelet.calcIoU(object_lidar_roi[lidar_index], object_camera_roi[cam_index]);
      }
      if (roi_fusion_nodelet.use_iou_x_)
      {
        iou_x = roi_fusion_nodelet.calcIoUX(object_lidar_roi[lidar_index], object_camera_roi[cam_index]);
      }
      if (roi_fusion_nodelet.use_iou_y_)
      {
        iou_y = roi_fusion_nodelet.calcIoUY(object_lidar_roi[lidar_index], object_camera_roi[cam_index]);
      }
      if (max_iou < iou + iou_x + iou_y)
      {
        max_iou_lidar_index = lidar_index;
        max_iou_cam_index = cam_index;
        max_iou = iou + iou_x + iou_y;
      }
    }

    if (roi_fusion_nodelet.iou_threshold_ < max_iou)
    {
      fusion_index.push_back(std::pair<int, int>(max_iou_cam_index, max_iou_lidar_index));
    }
  }
  return fusion_index;
}
void RoiFusion::getFusionCamObj(const msgs::DetectedObjectArray& objects_array,
                                const std::vector<std::pair<int, int>> fusion_index,
                                std::vector<DriveNet::MinMax2D>& cam_pixels_obj)
{
  for (size_t pair_index = 0; pair_index < fusion_index.size(); pair_index++)
  {
    int camera_index = fusion_index[pair_index].first;

    std::vector<DriveNet::PixelPosition> pixel_positions(2);
    pixel_positions[0].u = objects_array.objects[camera_index].camInfo[0].u;
    pixel_positions[0].v = objects_array.objects[camera_index].camInfo[0].v;
    pixel_positions[1].u =
        objects_array.objects[camera_index].camInfo[0].u + objects_array.objects[camera_index].camInfo[0].width;
    pixel_positions[1].v =
        objects_array.objects[camera_index].camInfo[0].v + objects_array.objects[camera_index].camInfo[0].height;
    transferPixelScaling(pixel_positions);

    DriveNet::MinMax2D min_max_2d_bbox;
    min_max_2d_bbox.p_min.u = pixel_positions[0].u;
    min_max_2d_bbox.p_min.v = pixel_positions[0].v;
    min_max_2d_bbox.p_max.u = pixel_positions[1].u;
    min_max_2d_bbox.p_max.v = pixel_positions[1].v;
    cam_pixels_obj.push_back(min_max_2d_bbox);
  }
}