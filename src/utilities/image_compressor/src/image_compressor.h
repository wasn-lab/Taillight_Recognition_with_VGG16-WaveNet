#pragma once
#include <string>
#include <sensor_msgs/CompressedImage.h>
#include <sensor_msgs/Image.h>

namespace image_compressor
{
enum compression_format
{
  jpg,  // 0
  png,  // 1
};

sensor_msgs::CompressedImageConstPtr compress_msg(sensor_msgs::ImageConstPtr msg,
                                                  const compression_format fmt = compression_format::jpg);
sensor_msgs::ImageConstPtr decompress_msg(sensor_msgs::CompressedImageConstPtr cmpr_msg);
};  // namespace image_compressor
