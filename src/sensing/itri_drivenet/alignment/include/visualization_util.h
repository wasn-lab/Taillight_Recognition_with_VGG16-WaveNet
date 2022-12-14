#ifndef VISUALIZATION_UTIL_H_
#define VISUALIZATION_UTIL_H_

/// ros
#include <msgs/DetectedObjectArray.h>
#include <msgs/DetectedObject.h>

/// opencv
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

/// utils
#include <camera_params.h>
#include "drivenet/image_preprocessing.h"
#include "drivenet/object_label_util.h"
#include "point_preprocessing.h"

class Visualization
{
private:
  int image_w_ = camera::image_width;
  int image_h_ = camera::image_height;

public:
  void drawPointCloudOnImage(cv::Mat& m_src, int point_u, int point_v, float point_x);
  void drawPointCloudOnImage(cv::Mat& m_src, int point_u, int point_v, int index);
  void drawBoxOnImage(cv::Mat& m_src, const std::vector<msgs::DetectedObject>& objects);
  void drawBoxOnImage(cv::Mat& m_src, const std::vector<DriveNet::MinMax2D>& min_max_2d_bbox, int source_id);
  void drawCubeOnImage(cv::Mat& m_src, std::vector<std::vector<DriveNet::PixelPosition>>& cube_2d_bbox);
  cv::Scalar getDistColor(float distance_in_meters);
  MinMax3D getDistLinePoint(float x_dist, float y_dist, float z_dist);
};

#endif
