#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <gtest/gtest.h>
#include "parknet.h"
#include "parknet_camera.h"
#include "parknet_image_utils.h"
#include "camera_utils.h"

#if USE(DARKNET)
TEST(ParknetImageUtilsTest, test_cvmat_to_darknet_image)
{
  auto yolo3_img = cv::imread(PARKNET_TEST_DATA_DIR "/10428_yolo3.ppm");

  auto img_darknet = parknet::convert_to_darknet_image(yolo3_img);
  EXPECT_EQ(yolo3_img.cols, img_darknet.w);
  EXPECT_EQ(yolo3_img.rows, img_darknet.h);
}
#endif

TEST(ParknetImageUtilsTest, test_draw_parking_slot)
{
  auto img = cv::imread(PARKNET_TEST_DATA_DIR "/front_120_preprocessed_10428.jpg");
  msgs::PointXY points[4];
  points[0].x = 117;
  points[0].y = 160;
  points[1].x = 216;
  points[1].y = 75;
  points[2].x = 886;
  points[2].y = 66;
  points[3].x = 963;
  points[3].y = 115;

  auto ret = parknet::draw_parking_slot(img, points);
  EXPECT_EQ(ret, 0);

  cv::resize(img, img, cv::Size(), 0.25, 0.25);
  auto expected = cv::imread(PARKNET_TEST_DATA_DIR "/front_120_preprocessed_10428_draw_parking_slot.ppm");
  EXPECT_TRUE(camera::cvmats_are_equal(img, expected));
}
