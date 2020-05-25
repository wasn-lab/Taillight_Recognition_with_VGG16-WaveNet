#pragma once
#include <opencv2/core/mat.hpp>
#include <memory>
#include <tensorflow/c/c_api.h>
#include "tf_utils.hpp"

namespace deeplab
{
class DeeplabSegmenterImpl
{
private:
  std::unique_ptr<TF_Graph, void (*)(TF_Graph*)> tf_graph_;
  std::unique_ptr<TF_Status, void (*)(TF_Status*)> tf_status_;
  std::unique_ptr<TF_SessionOptions, void (*)(TF_SessionOptions*)> tf_sess_options_;
  std::unique_ptr<TF_Session, void (*)(TF_Session*)> tf_sess_;
  TF_Tensor* input_tensor_;
  TF_Tensor* output_tensor_;

  int32_t preprocess_for_input_tensor(const cv::Mat& img_in, cv::Mat& img_rgb);
  int32_t postprocess_with_labels(const int64_t* labels, const cv::Mat& img_rgb, cv::Mat& img_out);
  int32_t inference(const cv::Mat& img_rgb);

public:
  DeeplabSegmenterImpl();
  ~DeeplabSegmenterImpl();
  int32_t segment(const cv::Mat& img_in, cv::Mat& img_out);
  int32_t segment_into_labels(const cv::Mat& img_in, uint8_t* labels);
};
};  // namespace deeplab
