#include "deeplab_segmenter.h"
#include "deeplab_segmenter_impl.h"

namespace deeplab {
DeeplabSegmenter::DeeplabSegmenter()
{
  nn_ptr_.reset(new DeeplabSegmenterImpl);
}

DeeplabSegmenter::~DeeplabSegmenter() = default;


int DeeplabSegmenter::segment(const cv::Mat& img_in, cv::Mat& img_out)
{
  nn_ptr_->segment(img_in, img_out);
  return 0;
}
}; // namespace deeplab
