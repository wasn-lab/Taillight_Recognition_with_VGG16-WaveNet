/*
   CREATER: ICL U300
   DATE: Feb, 2019
 */

#ifndef __PARKNET_IMAGE_MANAGER_H__
#define __PARKNET_IMAGE_MANAGER_H__

#include "nppdefs.h"
#include "parknet_camera.h"
#include "opencv2/core/mat.hpp"
#include <mutex>
#include <memory>
#include <vector>
#include "rect_class_score.h"
class NPPRemapper;
class NPPResizer;
class CameraDistanceMapper;

class ParknetImageManager
{
private:
  cv::Mat raw_image_[parknet::camera::num_cams_e];
  cv::Mat preprocessed_image_[parknet::camera::num_cams_e];
  cv::Mat displayed_image_[parknet::camera::num_cams_e];
  std::unique_ptr<NPPRemapper> npp_remapper_ptr_;  // thread-safe
  std::unique_ptr<NPPResizer> npp_resizer_ptr_;

  Npp8u* npp8u_ptr_1920x1208_bgr_distorted_[parknet::camera::num_cams_e];
  Npp8u* npp8u_ptr_1920x1208_bgr_undistorted_[parknet::camera::num_cams_e];
  Npp8u* npp8u_ptr_608x608_bgr_undistorted_[parknet::camera::num_cams_e];
  Npp8u* npp8u_ptr_608x608_rgb_undistorted_[parknet::camera::num_cams_e];
  Npp32f* npp32f_ptr_608x608_rgb_undistorted_[parknet::camera::num_cams_e];
  Npp32f* npp32f_ptr_369664_rgb_[parknet::camera::num_cams_e];

  // Detectio Results
  std::vector<RectClassScore<float> > detections_[parknet::camera::num_cams_e];

  // vars for threading
  std::mutex resizer_mutex_;  // guard npp_resizer_ptr_

  // private functions
  void release_image_if_necessary(cv::Mat& img_in);
  int image_processing_pipeline(const Npp8u* npp8u_ptr, const int cam_id);
  int image_processing_pipeline_yolov3(const Npp8u* npp8u_608x608, const int cam_id);
  void gen_displayed_image_yolov3(const int cam_id, const CameraDistanceMapper* dist_mapper_ptr,
                                  cv::Mat& annotated_image);
  void drawpred_1(const RectClassScore<float>& corner, const CameraDistanceMapper* dist_mapper_ptr, cv::Mat& frame,
                  int border_width);

public:
  ParknetImageManager();
  ~ParknetImageManager();
  int get_raw_image(const int cam_id, cv::Mat& img_out);
  int get_preprocessed_image(const int cam_id, cv::Mat& img_out);
  int get_annotated_image(const int cam_id, const CameraDistanceMapper* dist_mapper_ptr, cv::Mat& img_out);

  const Npp32f* get_blob(const int cam_id);
  int get_displayed_image(const int cam_id, cv::Mat& img_out);
  int set_raw_image(const cv::Mat& img_in, const int cam_id);
  int set_raw_image(const Npp8u* npp8u_ptr, const int cam_id, const int num_bytes);
  int set_detection(const std::vector<RectClassScore<float> >& in_detection, const int cam_id);
  const std::vector<RectClassScore<float> > get_detection(const int cam_id) const;
  int get_num_detections(const int cam_id) const;
};

#endif  // __PARKNET_IMAGE_MANAGER_H__
