#include <unistd.h>
#include <cstdio>
#include <cv_bridge/cv_bridge.h>
#include <glog/logging.h>
#include "image_compressor.h"
#include "image_compressor_args_parser.h"
#include "image_compressor_priv.h"

#define NO_UNUSED_VAR_CHECK(x) ((void)(x))

namespace image_compressor
{
sensor_msgs::CompressedImageConstPtr compress_msg(sensor_msgs::ImageConstPtr msg, const compression_format fmt)
{
  cv_bridge::CvImageConstPtr cv_ptr;

  try
  {
    cv_ptr = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::BGR8);
  }
  catch (cv_bridge::Exception& e)
  {
    LOG(ERROR) << "cv_bridge exception: " << e.what();
    return nullptr;
  }

  sensor_msgs::CompressedImagePtr cmpr_msg{ new sensor_msgs::CompressedImage };
  cmpr_msg->header = msg->header;
  CHECK(fmt == compression_format::jpg || fmt == compression_format::png);
  if (fmt == compression_format::jpg)
  {
    compress_by_jpg(cv_ptr->image, cmpr_msg->data);
    cmpr_msg->format = "jpeg";
  }
  else
  {
    compress_by_png(cv_ptr->image, cmpr_msg->data);
    cmpr_msg->format = "png";
  }

  return cmpr_msg;
}

sensor_msgs::ImageConstPtr decompress_msg(sensor_msgs::CompressedImageConstPtr cmpr_msg)
{
  cv_bridge::CvImage cv_img;
  int ret = decompress(cmpr_msg->data, cv_img.image);
  CHECK(ret == EXIT_SUCCESS);

  cv_img.header = cmpr_msg->header;
  cv_img.encoding = sensor_msgs::image_encodings::BGR8;

  return cv_img.toImageMsg();
}

};  // namespace image_compressor
