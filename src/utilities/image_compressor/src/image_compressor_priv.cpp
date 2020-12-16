#include <opencv2/imgcodecs.hpp>
#include <glog/logging.h>
#include "image_compressor_priv.h"
#include "image_compressor_args_parser.h"

#define NO_UNUSED_VAR_CHECK(x) ((void)(x))

namespace image_compressor
{
const std::vector<int> jpg_params{
  cv::IMWRITE_JPEG_QUALITY,
  85,
  cv::IMWRITE_JPEG_OPTIMIZE,
  1,
};
const std::vector<int> png_params{
  // IMWRITE_PNG_COMPRESSION: 1-fastest, 9-slowest
  cv::IMWRITE_PNG_COMPRESSION,
  1,
  cv::IMWRITE_PNG_STRATEGY,
  cv::IMWRITE_PNG_STRATEGY_RLE,
};

int compress_internal(const cv::Mat& in_img, std::vector<uint8_t>& cmpr_data, const std::string& format,
                      const std::vector<int>& params)
{
  CHECK(cmpr_data.empty()) << "cmpr_data is expected to be empty.";
  cmpr_data.reserve(in_img.total());
  cv::imencode(format, in_img, cmpr_data, params);
  if (is_verbose())
  {
    const uint64_t org_len = in_img.total() * in_img.elemSize();
    const uint64_t cmpr_len = cmpr_data.size();
    std::string prefix = format;
    if (format == ".jpg")
    {
      prefix += " (quality " + std::to_string(params[1]) + ")";
    }
    if (format == ".png")
    {
      prefix += " (level " + std::to_string(params[1]) + ")";
    }
    LOG(INFO) << prefix << " compression rate : " << cmpr_len << "/" << org_len << " = " << double(cmpr_len) / org_len;
  }
  return EXIT_SUCCESS;
}

int compress_by_jpg(const cv::Mat& in_img, std::vector<uint8_t>& cmpr_data)
{
  return compress_internal(in_img, cmpr_data, ".jpg", jpg_params);
}

int compress_by_png(const cv::Mat& in_img, std::vector<uint8_t>& cmpr_data)
{
  return compress_internal(in_img, cmpr_data, ".png", png_params);
}

int decompress(const std::vector<uint8_t>& cmpr_data, cv::Mat& out_img)
{
  out_img = cv::imdecode(cmpr_data, cv::IMREAD_COLOR);
  CHECK(out_img.data != nullptr) << "Error: cannot decompress image data (jpg)";
  return EXIT_SUCCESS;
}

};  // namespace image_compressor
