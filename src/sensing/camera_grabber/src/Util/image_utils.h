#ifndef __GRABBER_IMAGE_UTILS_H__
#define __GRABBER_IMAGE_UTILS_H__

#include <vector>
#include "opencv2/core/mat.hpp"

namespace SensingSubSystem
{
// Util function:  copy from cv::Mat to cv::Mat
void copyBufferByIndex(const std::vector<cv::Mat>& in, std::vector<cv::Mat>& out, std::vector<size_t>& num);
};      // namespace
#endif  // __GRABBER_IMAGE_UTILS_H__
