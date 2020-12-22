#include "deeplab_segmenter.h"
#include "deeplab_segmenter_impl.h"

namespace deeplab
{
DeeplabSegmenter::DeeplabSegmenter()
{
  nn_ptr_.reset(new DeeplabSegmenterImpl);
}

DeeplabSegmenter::~DeeplabSegmenter() = default;

int32_t DeeplabSegmenter::segment(const cv::Mat& img_in, cv::Mat& img_out)
{
  return nn_ptr_->segment(img_in, img_out);
}

int32_t DeeplabSegmenter::segment_with_labels(const cv::Mat& img_in, cv::Mat& img_out, uint8_t* labels)
{
  return nn_ptr_->segment_with_labels(img_in, img_out, labels);
}

};  // namespace deeplab
