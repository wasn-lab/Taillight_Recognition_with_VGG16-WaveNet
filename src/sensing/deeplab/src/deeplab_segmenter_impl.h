#pragma once
#include <opencv2/core/mat.hpp>
#include <memory>
#include <tensorflow/c/c_api.h>
#include "tf_utils.hpp"


namespace deeplab {
class DeeplabSegmenterImpl {
  private:
    std::unique_ptr<TF_Graph, void (*)(TF_Graph*)> tf_graph_;
    std::unique_ptr<TF_Status, void (*)(TF_Status*)> tf_status_;
    std::unique_ptr<TF_SessionOptions, void (*)(TF_SessionOptions*)> tf_sess_options_;
    std::unique_ptr<TF_Session, void (*)(TF_Session*)> tf_sess_;

  public:
    DeeplabSegmenterImpl();
    ~DeeplabSegmenterImpl();
    int segment(const cv::Mat& img_in, cv::Mat& img_out);
};
}; // namespace deeplab
