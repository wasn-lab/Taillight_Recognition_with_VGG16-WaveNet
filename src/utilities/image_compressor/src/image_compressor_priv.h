#pragma once
#include "opencv2/core/mat.hpp"

namespace image_compressor
{
int compress_internal(const cv::Mat& in_img, std::vector<uint8_t>& cmpr_data, const std::string format,
                      const std::vector<int>& params);
int compress_by_jpg(const cv::Mat& in_img, std::vector<uint8_t>& cmpr_data);
int compress_by_png(const cv::Mat& in_img, std::vector<uint8_t>& cmpr_data);
int decompress(const std::vector<uint8_t>& cmpr_data, cv::Mat& out_img);
};  // namespace image_compressor
