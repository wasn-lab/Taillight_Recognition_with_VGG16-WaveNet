#pragma once
#include "opencv2/core/mat.hpp"
#include "image_compressor.h"

namespace image_compressor
{
double compress(const cv::Mat& in_img, std::vector<uint8_t>& cmpr_data, const compression_format format,
                const int32_t quality);
int decompress(const std::vector<uint8_t>& cmpr_data, cv::Mat& out_img);
};  // namespace image_compressor
