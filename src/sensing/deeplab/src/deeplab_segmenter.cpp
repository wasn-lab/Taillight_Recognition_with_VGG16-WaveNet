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

int32_t DeeplabSegmenter::segment_into_labels(const cv::Mat& img_in, uint8_t* labels)
{
  return nn_ptr_->segment_into_labels(img_in, labels);
}

};  // namespace deeplab
