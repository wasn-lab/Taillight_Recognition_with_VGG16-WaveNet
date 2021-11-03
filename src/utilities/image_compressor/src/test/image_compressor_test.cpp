#include <cstdlib>
#include <memory>
#include <opencv2/opencv.hpp>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <cv_bridge/cv_bridge.h>
#include "image_compressor.h"
#include "image_compressor_priv.h"
#include "image_compressor_def.h"
#include "image_compressor_args_parser.h"

constexpr int NUM_PERF_LOOPS = 100;

cv::Mat g_img;
sensor_msgs::ImagePtr g_msg_ptr;

static void init_test_data()
{
  if (g_img.empty())
  {
    LOG(INFO) << "initialize g_img";
    const std::string img_path = std::string{ IMAGE_COMPRESSOR_TEST_DATA_DIR } + "/608x384.png";
    g_img = cv::imread(img_path);
  }
  if (!g_msg_ptr)
  {
    LOG(INFO) << "initialize g_msg_ptr";
    cv_bridge::CvImage cv_img;
    cv_img.header.seq = 31415;
    cv_img.header.stamp.sec = 1608000995;
    cv_img.header.stamp.nsec = 10007;
    cv_img.header.frame_id = "test";

    cv_img.encoding = sensor_msgs::image_encodings::BGR8;
    cv_img.image = g_img;
    g_msg_ptr = cv_img.toImageMsg();

    EXPECT_EQ(g_msg_ptr->header.seq, 31415);
    EXPECT_EQ(g_msg_ptr->header.stamp.sec, 1608000995);
    EXPECT_EQ(g_msg_ptr->header.stamp.nsec, 10007);
    EXPECT_EQ(g_msg_ptr->header.frame_id, "test");
    EXPECT_EQ(g_msg_ptr->width, 608);
    EXPECT_EQ(g_msg_ptr->height, 384);
    EXPECT_EQ(g_msg_ptr->data.size(), 608 * 384 * 3);
  }
}

TEST(ImageCompressorTest, cmpr_msg_jpg)
{
  init_test_data();
  // Test jpg compression
  sensor_msgs::CompressedImageConstPtr cmpr_msg_ptr = image_compressor::compress_msg(g_msg_ptr);
  EXPECT_EQ(cmpr_msg_ptr->header.seq, 31415);
  EXPECT_EQ(cmpr_msg_ptr->header.stamp.sec, 1608000995);
  EXPECT_EQ(cmpr_msg_ptr->header.stamp.nsec, 10007);
  EXPECT_EQ(cmpr_msg_ptr->header.frame_id, "test");
  EXPECT_EQ(cmpr_msg_ptr->format, "jpeg");
  EXPECT_TRUE(cmpr_msg_ptr->data.size() > 0);

  sensor_msgs::ImageConstPtr decmpr_msg_ptr = image_compressor::decompress_msg(cmpr_msg_ptr);
  EXPECT_EQ(decmpr_msg_ptr->header.seq, 31415);
  EXPECT_EQ(decmpr_msg_ptr->header.stamp.sec, 1608000995);
  EXPECT_EQ(decmpr_msg_ptr->header.stamp.nsec, 10007);
  EXPECT_EQ(decmpr_msg_ptr->header.frame_id, "test");
  EXPECT_EQ(decmpr_msg_ptr->width, 608);
  EXPECT_EQ(decmpr_msg_ptr->height, 384);
}

TEST(ImageCompressorTest, cmpr_msg_jpg_perf)
{
  init_test_data();
  for (int i = 0; i < NUM_PERF_LOOPS; i++)
  {
    auto cmpr_img_ptr = image_compressor::compress_msg(g_msg_ptr);
  }
}

TEST(ImageCompressorTest, decmpr_msg_jpg_perf)
{
  init_test_data();
  sensor_msgs::CompressedImageConstPtr cmpr_msg_ptr = image_compressor::compress_msg(g_msg_ptr);
  for (int i = 0; i < NUM_PERF_LOOPS; i++)
  {
    auto decmpr_msg_ptr = image_compressor::decompress_msg(cmpr_msg_ptr);
  }
}

TEST(ImageCompressorTest, cmpr_msg_png)
{
  init_test_data();
  auto cmpr_msg_ptr = image_compressor::compress_msg(g_msg_ptr, image_compressor::compression_format::png);
  EXPECT_EQ(cmpr_msg_ptr->header.seq, 31415);
  EXPECT_EQ(cmpr_msg_ptr->header.stamp.sec, 1608000995);
  EXPECT_EQ(cmpr_msg_ptr->header.stamp.nsec, 10007);
  EXPECT_EQ(cmpr_msg_ptr->header.frame_id, "test");
  EXPECT_EQ(cmpr_msg_ptr->format, "png");
  EXPECT_TRUE(cmpr_msg_ptr->data.size() > 0);

  sensor_msgs::ImageConstPtr decmpr_msg_ptr = image_compressor::decompress_msg(cmpr_msg_ptr);
  EXPECT_EQ(g_msg_ptr->header, decmpr_msg_ptr->header);
  EXPECT_EQ(g_msg_ptr->height, decmpr_msg_ptr->height);
  EXPECT_EQ(g_msg_ptr->width, decmpr_msg_ptr->width);
  EXPECT_EQ(g_msg_ptr->encoding, decmpr_msg_ptr->encoding);
  EXPECT_EQ(g_msg_ptr->is_bigendian, decmpr_msg_ptr->is_bigendian);
  EXPECT_EQ(g_msg_ptr->step, decmpr_msg_ptr->step);
  EXPECT_EQ(g_msg_ptr->data, decmpr_msg_ptr->data);
}

TEST(ImageCompressorTest, cmpr_msg_png_perf)
{
  init_test_data();
  for (int i = 0; i < NUM_PERF_LOOPS; i++)
  {
    auto cmpr_img_ptr = image_compressor::compress_msg(g_msg_ptr, image_compressor::compression_format::png);
  }
}

TEST(ImageCompressorTest, decmpr_msg_png_perf)
{
  init_test_data();
  sensor_msgs::CompressedImageConstPtr cmpr_msg_ptr =
      image_compressor::compress_msg(g_msg_ptr, image_compressor::compression_format::png);
  for (int i = 0; i < NUM_PERF_LOOPS; i++)
  {
    auto decmpr_msg_ptr = image_compressor::decompress_msg(cmpr_msg_ptr);
  }
}

TEST(ImageCompressorTest, compress_by_jpg_1)
{
  init_test_data();
  std::vector<uint8_t> cmpr_data;
  const int32_t quality = 85;
  image_compressor::compress(g_img, cmpr_data, image_compressor::compression_format::jpg, quality);

  cv::Mat decmpr_img;
  int ret = image_compressor::decompress(cmpr_data, decmpr_img);
  EXPECT_EQ(ret, EXIT_SUCCESS);
  EXPECT_EQ(decmpr_img.cols, 608);
  EXPECT_EQ(decmpr_img.rows, 384);
}

TEST(ImageCompressorTest, compress_by_jpg_perf)
{
  init_test_data();
  std::vector<uint8_t> cmpr_data;

  const int32_t quality = 85;
  for (int i = 0; i < NUM_PERF_LOOPS; i++)
  {
    cmpr_data.clear();
    image_compressor::compress(g_img, cmpr_data, image_compressor::compression_format::jpg, quality);
  }
}

TEST(ImageCompressorTest, decompress_jpg_perf)
{
  init_test_data();
  std::vector<uint8_t> cmpr_data;
  const int32_t quality = 85;
  image_compressor::compress(g_img, cmpr_data, image_compressor::compression_format::jpg, quality);
  for (int i = 0; i < NUM_PERF_LOOPS; i++)
  {
    cv::Mat decmpr_img;
    image_compressor::decompress(cmpr_data, decmpr_img);
  }
}

TEST(ImageCompressorTest, compress_by_png_1)
{
  init_test_data();
  std::vector<uint8_t> cmpr_data;
  const int32_t quality = 1;
  image_compressor::compress(g_img, cmpr_data, image_compressor::compression_format::png, quality);

  cv::Mat decmpr_img;
  int ret = image_compressor::decompress(cmpr_data, decmpr_img);
  EXPECT_EQ(ret, EXIT_SUCCESS);
  EXPECT_EQ(decmpr_img.cols, 608);
  EXPECT_EQ(decmpr_img.rows, 384);

  // png is lossless
  const int data_len = g_img.total() * g_img.elemSize();
  for (int i = 0; i < data_len; i++)
  {
    EXPECT_EQ(decmpr_img.data[i], g_img.data[i]);
  }
  // cv::imwrite("kk.png", decmpr_img);
}

TEST(ImageCompressorTest, compress_by_png_perf)
{
  init_test_data();
  std::vector<uint8_t> cmpr_data;
  const int32_t quality = 1;
  for (int i = 0; i < NUM_PERF_LOOPS; i++)
  {
    cmpr_data.clear();
    image_compressor::compress(g_img, cmpr_data, image_compressor::compression_format::png, quality);
  }
}

TEST(ImageCompressorTest, decompress_png_perf)
{
  init_test_data();
  std::vector<uint8_t> cmpr_data;
  const int32_t quality = 1;
  image_compressor::compress(g_img, cmpr_data, image_compressor::compression_format::png, quality);
  for (int i = 0; i < NUM_PERF_LOOPS; i++)
  {
    cv::Mat decmpr_img;
    image_compressor::decompress(cmpr_data, decmpr_img);
  }
}
