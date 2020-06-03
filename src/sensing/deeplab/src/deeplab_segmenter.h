#pragma once

#include <opencv2/core/mat.hpp>
#include <memory>

namespace deeplab {
class DeeplabSegmenterImpl;

class DeeplabSegmenter {
  private:
    std::unique_ptr<DeeplabSegmenterImpl> nn_ptr_;
  public:
    DeeplabSegmenter();
    ~DeeplabSegmenter();
    int32_t segment(const cv::Mat& img_in, cv::Mat& img_out);
};
}; // namespace deeplab
