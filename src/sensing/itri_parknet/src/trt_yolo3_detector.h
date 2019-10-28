#ifndef __TRT_YOLO3_DETECTOR_H__
#define __TRT_YOLO3_DETECTOR_H__

#include "parknet.h"
#if USE(TENSORRT)
#include <vector>
#include <mutex>
#include <memory>
#include "ds_image.h"
#include "rect_class_score.h"
#include "parknet_logging.h"
#include "parknet_camera.h"
#include "nppdefs.h"
class Yolo;

namespace parknet
{
class TRTYolo3Detector
{
private:
  std::unique_ptr<Yolo> inferNet_;
  std::vector<DsImage> dsImages[parknet::camera::num_cams_e];
  std::mutex mu_;
  int get_detections(std::vector<RectClassScore<float> >& out_detections);

public:
  TRTYolo3Detector();
  ~TRTYolo3Detector();

  bool is_initialized() const
  {
    return bool(inferNet_);
  }
  int detect(std::vector<RectClassScore<float> >& out_detections, cv::Mat& in_cv_mat_image, const int cam_id);
  int detect(std::vector<RectClassScore<float> >& out_detections, const Npp32f* npp32f_ptr_369664_rgb,
             const int cam_id);
};  // class TRTYolo3Detector
}  // namespace parknet

#endif  // USE(TENSORRT)
#endif  // __TRT_YOLO3_DETECTOR_H__
