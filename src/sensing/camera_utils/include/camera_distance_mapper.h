#ifndef __CAMERA_DISTANCE_MAPPER_H__
#define __CAMERA_DISTANCE_MAPPER_H__

#include <string>
#include <memory>
#include <opencv2/core/mat.hpp>
#include "camera_params.h"

class CameraDistanceMapper
{
private:
  const camera::id cam_id_;
  std::unique_ptr<float[][camera::raw_image_cols]> x_in_meters_ptr_;
  std::unique_ptr<float[][camera::raw_image_cols]> y_in_meters_ptr_;
  std::unique_ptr<float[][camera::raw_image_cols]> z_in_meters_ptr_;

  // private functions
  std::string get_json_filename();
  int read_dist_from_json();

public:
  CameraDistanceMapper(const camera::id cam_id);
  ~CameraDistanceMapper();
  cv::Mat remap_distance_in_undistorted_image();
  int get_distance_raw_1920x1208(const int im_x, const int im_y, float* spatial_x, float* spatial_y,
                                 float* spatial_z) const;
};

#endif  // __CAMERA_DISTANCE_MAPPER_H__
