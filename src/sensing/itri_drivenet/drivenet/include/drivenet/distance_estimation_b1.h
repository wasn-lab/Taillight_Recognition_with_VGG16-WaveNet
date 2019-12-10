#ifndef DISTANCEESTIMATION_H_
#define DISTANCEESTIMATION_H_

// ROS message
#include "camera_params.h"  // include camera topic name
#include <msgs/BoxPoint.h>
#include <msgs/PointXYZ.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
struct DisEstiParams
{
  std::vector<int> regionHeight_x;
  std::vector<float> regionDist_x;
  std::vector<int> regionHeight_y;
  std::vector<float> regionDist_y;
  std::vector<float> regionHeightSlope_x;
  std::vector<float> regionHeightSlope_y;
};

struct CheckArea
{
  cv::Point LeftLinePoint1;
  cv::Point LeftLinePoint2;
  cv::Point RightLinePoint1;
  cv::Point RightLinePoint2;
};

class DistanceEstimation
{
private:
  DisEstiParams camFR60, camFC60, camFL60, camFT120, camRF120, camRB120, camLF120, camLB120, camBT120;
  CheckArea ShrinkArea_camFR60, ShrinkArea_camFT120, ShrinkArea_camBT120;
  /// camId: 0 = camFR60
  /// camId: 1 = camFC60
  /// camId: 2 = camFL60
  /// camId: 4 = camFT120
  /// camId: 5 = camRF120
  /// camId: 6 = camRB120
  /// camId: 8 = camLF120
  /// camId: 9 = camLB120
  /// camId: 10 = camBT120

  float Lidar_offset_x = 0;
  float Lidar_offset_y = 0;
  float Lidar_offset_z = -3;
  int carId = 1;

  void initParams();
  void initShrinkArea();
  void initDetectArea();

  float ComputeObjectXDist(int piexl_loc, std::vector<int> regionHeight, std::vector<float> regionDist);
  float ComputeObjectXDistWithSlope(int piexl_loc_y, int piexl_loc_x, std::vector<int> regionHeight,
                                    std::vector<float> regionHeightSlope_x, std::vector<float> regionDist);
  float ComputeObjectYDist(int piexl_loc_y, int piexl_loc_x, std::vector<int> regionHeight,
                           std::vector<float> regionHeightSlope_y, std::vector<float> regionDist, int img_h);
  msgs::PointXYZ GetPointDist(int x, int y, int cam_id);
  int BoxShrink(int cam_id, std::vector<int> Points_src, std::vector<int>& Points_dst);
  int CheckPointInArea(CheckArea area, int object_x1, int object_y2);
  float RatioDefine(int cam_id, int cls);

public:
  void init(int carId);
  msgs::BoxPoint Get3dBBox(int x1, int y1, int x2, int y2, int class_id, int cam_id);
  msgs::BoxPoint Get3dBBox(msgs::PointXYZ p0, msgs::PointXYZ p3, int class_id, int cam_id);

  /// camId:0
  // Front right 60 range:
  // x axis: 7 ~ 50 meters
  // y axis: -10 ~ 10 meters

  /// camId: 1
  // Front center 60 range:
  // x axis: 7 ~ 50 meters
  // y axis: -10 ~ 10 meters

  /// camId: 4
  // Front top 120 range:
  // x axis: 0 ~ 7 meters
  // y axis: -9 ~ 6 meters

  /// camId: 10
  // Back top 120 range:
  // x axis: 8 ~ 20 meters
  // y axis: -3 ~ 3 meters

  CheckArea camFR60_area, camFC60_area, camFT120_area, camBT120_area;
};

#endif /*DISTANCEESTIMATION_H_*/
