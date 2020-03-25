#ifndef DISTANCEESTIMATION_B1_V2_H_
#define DISTANCEESTIMATION_B1_V2_H_

// ROS message
#include "camera_params.h"  // include camera topic name
#include "distance_estimation.h"
#include <msgs/BoxPoint.h>
#include <msgs/PointXYZ.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>
#include <string>
#include <fstream>
#include <jsoncpp/json/json.h>

class DistanceEstimation
{
private:
  DisEstiParams* arr_params;
  CheckArea* ShrinkArea;
  cv::Point3d** align_FC60 /*, align_FL60, align_FR60*/;

  float Lidar_offset_x = 0;
  float Lidar_offset_y = 0;
  float Lidar_offset_z = -3;
  const int img_h = camera::raw_image_height;
  const int img_w = camera::raw_image_width;

  const int img_al_h = camera::image_height;
  const int img_al_w = camera::image_width;

  int de_mode = 0;

  void initParams();
  void initShrinkArea();
  void initDetectArea();

  int ReadDistanceFromJson(const std::string& filename, cv::Point3d** dist_in_cm, const int rows, const int cols);
  float ComputeObjectXDist(int piexl_loc, std::vector<int> regionHeight, std::vector<float> regionDist);
  float ComputeObjectXDistWithSlope(int piexl_loc_x, int piexl_loc_y, std::vector<int> regionHeight,
                                    std::vector<float> regionHeightSlope_x, std::vector<float> regionDist);
  float ComputeObjectYDist(int piexl_loc_y, int piexl_loc_x, std::vector<int> regionHeight,
                           std::vector<float> regionHeightSlope_y, std::vector<float> regionDist, int img_h);
  msgs::PointXYZ GetPointDist(int x, int y, camera::id cam_id);
  int BoxShrink(camera::id cam_id, std::vector<int> Points_src, std::vector<int>& Points_dst);
  float RatioDefine(camera::id cam_id, int cls);

public:
  DistanceEstimation();
  ~DistanceEstimation();

  void init(std::string pkgPath, int mode);
  msgs::BoxPoint Get3dBBox(int x1, int y1, int x2, int y2, int class_id, camera::id cam_id);
  msgs::BoxPoint Get3dBBox(msgs::PointXYZ p0, msgs::PointXYZ p3, int class_id, camera::id cam_id);
  int CheckPointInArea(CheckArea area, int object_x1, int object_y2);

  CheckArea* area;
};

#endif /*DISTANCEESTIMATION_B1_V2_H_*/
