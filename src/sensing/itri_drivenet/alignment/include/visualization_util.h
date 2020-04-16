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

class Visualization
{
private:
  int image_w_ = camera::image_width;
  int image_h_ = camera::image_height;

public:
  void drawPointCloudOnImage(cv::Mat& m_src, int point_u, int point_v, float point_x);
  void drawBoxOnImage(cv::Mat& m_src, std::vector<msgs::DetectedObject>& objects);
  cv::Scalar getDistColor(float distance_in_meters);
};

#endif
