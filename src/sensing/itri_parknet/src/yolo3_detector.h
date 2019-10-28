/*
   CREATER: ICL U300
   DATE: Feb, 2019
 */

#ifndef __YOLO3_DETECTOR_H__
#define __YOLO3_DETECTOR_H__

#include <vector>
#include <chrono>
#include <mutex>

#include <image_transport/image_transport.h>
#include <rect_class_score.h>
#include <opencv2/opencv.hpp>

extern "C" {
#undef __cplusplus
#include "box.h"
#include "image.h"
#include "network.h"
#include "detection_layer.h"
#include "parser.h"
#include "region_layer.h"
#include "utils.h"
#define __cplusplus
}

namespace darknet
{
class Yolo3Detector
{
private:
  double score_thresh_, nms_thresh_;
  network* darknet_network_;
  std::vector<box> darknet_boxes_;
  std::mutex darknet_network_mutex_;

  int forward(std::vector<RectClassScore<float> >& out_detections, image& in_darknet_image);

public:
  Yolo3Detector() : darknet_network_(nullptr)
  {
  }

  bool is_initialized() const
  {
    return bool(darknet_network_);
  }

  void load(std::string& in_model_file, std::string& in_trained_file, double in_score_thresh, double in_nms_thresh);

  ~Yolo3Detector();

  int detect(std::vector<RectClassScore<float> >& out_detections, image& in_darknet_image);
  uint32_t get_network_height();
  uint32_t get_network_width();
};
}  // namespace darknet

#endif  // __YOLO3_DETECTOR_H__
