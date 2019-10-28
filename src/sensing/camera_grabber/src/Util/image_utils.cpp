#include "image_utils.h"

namespace SensingSubSystem
{
// Util function:  copy from cv::Mat to cv::Mat
void copyBufferByIndex(const std::vector<cv::Mat>& in, std::vector<cv::Mat>& out, std::vector<size_t>& num)
{
  out.clear();
  out.resize(num.size());
  for (size_t i = 0; i < num.size(); ++i)
  {
    out[i] = in[num[i]].clone();
  }
}
};  // namespace
