#include <gtest/gtest.h>
#include <glog/logging.h>
#include <opencv2/opencv.hpp>
#include "camera_utils_test.h"
#include "camera_utils.h"
#include "camera_params.h"
#include "npp_resizer.h"

const cv::Mat g_img_in = cv::imread(CAMERA_UTILS_TEST_DATA_DIR "/front_120_raw_10000.jpg");

TEST(CameraUtilsTest, test_has_yolov3_image_size)
{
  cv::Mat img;
  img.create(608, 608, CV_8UC3);
  EXPECT_TRUE(camera::has_yolov3_image_size(img));
  img.release();

  img.create(608, 384, CV_8UC3);
  EXPECT_FALSE(camera::has_yolov3_image_size(img));
}

TEST(CameraUtilsTest, test_fit_yolov3_image_size)
{
  cv::Mat yolov3_img;
  camera::fit_yolov3_image_size(g_img_in, yolov3_img);
  EXPECT_EQ(608, yolov3_img.cols);
  EXPECT_EQ(608, yolov3_img.rows);
}

TEST(CameraUtilsTest, test_fit_yolov3_image_size_with_resizer)
{
  cv::Mat yolov3_img;
  NPPResizer resizer(camera::raw_image_height, camera::raw_image_width, camera::image_ratio_on_yolov3,
                     camera::image_ratio_on_yolov3);
  camera::fit_yolov3_image_size(g_img_in, yolov3_img, resizer);
  EXPECT_EQ(608, yolov3_img.cols);
  EXPECT_EQ(608, yolov3_img.rows);
}

TEST(CameraUtilsTest, perf_fit_yolov3_image_size_with_resizer)
{
  cv::Mat yolov3_img;
  NPPResizer resizer(camera::raw_image_height, camera::raw_image_width, camera::image_ratio_on_yolov3,
                     camera::image_ratio_on_yolov3);
  for (int i = 0; i < NUM_PERF_LOOPS; i++)
  {
    camera::fit_yolov3_image_size(g_img_in, yolov3_img, resizer);
  }
}

TEST(CameraUtilsTest, perf_fit_yolov3_image_size_with_resizer_only)
{
  cv::Mat yolov3_img;
  NPPResizer resizer(camera::raw_image_height, camera::raw_image_width, camera::image_ratio_on_yolov3,
                     camera::image_ratio_on_yolov3);
  for (int i = 0; i < NUM_PERF_LOOPS; i++)
  {
    resizer.resize_to_letterbox_yolov3(g_img_in, yolov3_img);
  }
}

TEST(CameraUtilsTest, test_calc_yolov3_image_to_raw_size)
{
  cv::Mat yolov3_img = cv::imread(CAMERA_UTILS_TEST_DATA_DIR "/front_120_yolov3_10000.jpg");
  cv::Mat scaled_img;
  auto ret = camera::scale_yolov3_image_to_raw_size(yolov3_img, scaled_img);
  EXPECT_EQ(ret, 0);
  EXPECT_EQ(scaled_img.rows, camera::raw_image_height);
  EXPECT_EQ(scaled_img.cols, camera::raw_image_width);
}

TEST(CameraUtilsTest, test_camera_to_yolov3_xy)
{
  int cam_image_x = 0, cam_image_y = 0, yolov3_x = -1, yolov3_y = -1;

  camera::camera_to_yolov3_xy(cam_image_x, cam_image_y, &yolov3_x, &yolov3_y);
  EXPECT_EQ(yolov3_x, camera::left_border);
  EXPECT_EQ(yolov3_y, camera::top_border);

  cam_image_x = 1920 - 1;
  cam_image_y = 1208 - 1;
  camera::camera_to_yolov3_xy(cam_image_x, cam_image_y, &yolov3_x, &yolov3_y);
  EXPECT_EQ(yolov3_x, 607);
  EXPECT_LE(std::abs(yolov3_y - 494), 1);
}

TEST(CameraUtilsTest, test_yolov3_to_camera_xy)
{
  int cam_image_x = 0, cam_image_y = 0;
  int yolov3_x = camera::left_border, yolov3_y = camera::top_border;

  camera::yolov3_to_camera_xy(yolov3_x, yolov3_y, &cam_image_x, &cam_image_y);
  EXPECT_EQ(cam_image_x, 0);
  EXPECT_EQ(cam_image_y, 0);

  yolov3_x = 607;
  yolov3_y = 494;
  camera::yolov3_to_camera_xy(yolov3_x, yolov3_y, &cam_image_x, &cam_image_y);
  // TODO: Rounding error
  EXPECT_LE(std::abs(cam_image_x - 1920), 4);
  EXPECT_LE(std::abs(cam_image_y - 1208), 5);

  yolov3_x = int(377.787) + int(17.5906) / 2;
  yolov3_y = int(384.484) + int(29.4605) / 2;
  camera::yolov3_to_camera_xy(yolov3_x, yolov3_y, &cam_image_x, &cam_image_y);
  LOG(INFO) << "yolov3_xy (" << yolov3_x << ", " << yolov3_y << ") -> camera (" << cam_image_x << ", " << cam_image_y
            << ")";
  yolov3_x = int(378.447) + int(18.4501) / 2;
  yolov3_y = int(383.658) + int(31.1359) / 2;
  camera::yolov3_to_camera_xy(yolov3_x, yolov3_y, &cam_image_x, &cam_image_y);
  LOG(INFO) << "yolov3_xy (" << yolov3_x << ", " << yolov3_y << ") -> camera (" << cam_image_x << ", " << cam_image_y
            << ")";
}

TEST(CameraUtilsTest, test_calc_cvmat_checksum)
{
  cv::Mat cvmat1 = cv::imread(CAMERA_UTILS_TEST_DATA_DIR "/front_120_raw_10000.jpg");
  EXPECT_EQ(camera::calc_cvmat_checksum(cvmat1), 486308059);
}

TEST(CameraUtilsTest, test_cvmats_are_equal)
{
  cv::Mat cvmat1 = cv::imread(CAMERA_UTILS_TEST_DATA_DIR "/front_120_raw_10000.jpg");
  cv::Mat cvmat2 = cv::imread(CAMERA_UTILS_TEST_DATA_DIR "/front_120_raw_10000.jpg");
  EXPECT_TRUE(camera::cvmats_are_equal(cvmat1, cvmat2));
  EXPECT_EQ(camera::calc_cvmat_checksum(cvmat1), camera::calc_cvmat_checksum(cvmat2));

  cvmat1.at<uchar>(30, 30) = 255;
  EXPECT_FALSE(camera::cvmats_are_equal(cvmat1, cvmat2));
}

TEST(CameraUtilsTest, test_is_black_image)
{
  cv::Mat img(cv::Mat::zeros(8, 16, CV_8UC3));
  EXPECT_TRUE(camera::is_black_image(img));

  img.at<uint8_t>(1,2) = 1;
  EXPECT_FALSE(camera::is_black_image(img));
}
