#include <opencv2/imgcodecs.hpp>
#include <glog/logging.h>
#include "image_compressor_priv.h"
#include "image_compressor_args_parser.h"

#define NO_UNUSED_VAR_CHECK(x) ((void)(x))

namespace image_compressor
{
static std::vector<int> get_encoding_params(const compression_format format, const int32_t quality)
{
  if (format == compression_format::jpg)
  {
    return std::vector<int>{ cv::IMWRITE_JPEG_QUALITY, quality, cv::IMWRITE_JPEG_OPTIMIZE, 1 };
  }
  // IMWRITE_PNG_COMPRESSION: 1-fastest, 9-slowest
  return std::vector<int>{
    cv::IMWRITE_PNG_COMPRESSION,
    quality,
    cv::IMWRITE_PNG_STRATEGY,
    cv::IMWRITE_PNG_STRATEGY_RLE,
  };
}

double compress(const cv::Mat& in_img, std::vector<uint8_t>& cmpr_data, const compression_format format,
                const int32_t quality)
{
  CHECK(cmpr_data.empty()) << "cmpr_data is expected to be empty.";
  cmpr_data.reserve(in_img.total());
  std::string fmt_str = ".jpg";
  if (format == compression_format::png)
  {
    fmt_str = ".png";
  }
  cv::imencode(fmt_str, in_img, cmpr_data, get_encoding_params(format, quality));

  return double(in_img.total() * in_img.elemSize()) / cmpr_data.size();
}

int decompress(const std::vector<uint8_t>& cmpr_data, cv::Mat& out_img)
{
  out_img = cv::imdecode(cmpr_data, cv::IMREAD_COLOR);
  CHECK(out_img.data != nullptr) << "Error: cannot decompress image data (jpg)";
  return EXIT_SUCCESS;
}

};  // namespace image_compressor
