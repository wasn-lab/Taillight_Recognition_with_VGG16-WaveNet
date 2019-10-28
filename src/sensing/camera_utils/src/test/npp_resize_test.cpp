#include <gtest/gtest.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include "npp.h"
#include "npp_wrapper.h"
#include "camera_utils_test.h"
#include "camera_params.h"
#include "npp_resizer.h"
#include "camera_utils.h"
#include "npp_utils.h"

namespace npp_wrapper_test
{
const cv::Mat g_img_in = cv::imread(CAMERA_UTILS_TEST_DATA_DIR "/front_120_raw_10000.jpg");

TEST(NPPWrapperTest, test_resize)
{
  cv::Mat resized;
  npp_wrapper::resize(g_img_in, resized, 0.2, 0.2);
  //  imwrite("front_120_raw_10000_npp_resized.ppm", resized);

  cv::Mat expected = cv::imread(CAMERA_UTILS_TEST_DATA_DIR "/front_120_raw_10000_npp_resized.ppm");
  EXPECT_TRUE(camera::cvmats_are_equal(resized, expected));
}

TEST(NPPWrapperTest, test_resizer_to_yolov3)
{
  cv::Mat img_out;
  NPPResizer resizer(g_img_in.rows, g_img_in.cols, camera::yolov3_image_rows, camera::yolov3_image_cols);
  resizer.resize(g_img_in, img_out);
  EXPECT_EQ(img_out.cols, 608);
  EXPECT_EQ(img_out.rows, 608);
}

TEST(NPPWrapperTest, test_resizer_to_letterbox_yolov3)
{
  cv::Mat img_out;
  NPPResizer resizer(g_img_in.rows, g_img_in.cols, camera::yolov3_image_rows, camera::yolov3_image_cols);
  resizer.resize_to_letterbox_yolov3(g_img_in, img_out);

  // cv::imwrite("letterbox_yolov3_cv2cv.jpg", img_out);
  EXPECT_EQ(img_out.cols, 608);
  EXPECT_EQ(img_out.rows, 608);
}

TEST(NPPWrapperTest, test_resizer_to_letterbox_yolov3_gpu2gpu)
{
  cv::Mat img_out;
  int dummy;
  Npp8u* npp8u_ptr_in = nppiMalloc_8u_C3(g_img_in.rows, g_img_in.cols, &dummy);
  Npp8u* npp8u_ptr_out = nppiMalloc_8u_C3(camera::yolov3_image_rows, camera::yolov3_image_cols, &dummy);

  NPPResizer resizer(g_img_in.rows, g_img_in.cols, camera::yolov3_image_rows, camera::yolov3_image_cols);

  npp_wrapper::cvmat_to_npp8u_ptr(g_img_in, npp8u_ptr_in);

  resizer.resize_to_letterbox_yolov3(npp8u_ptr_in, npp8u_ptr_out);

  npp_wrapper::npp8u_ptr_to_cvmat(npp8u_ptr_out, camera::num_yolov3_image_bytes, img_out, camera::yolov3_image_rows,
                                  camera::yolov3_image_cols);

  // cv::imwrite("letterbox_yolov3_gpu2gpu.jpg", img_out);
  EXPECT_EQ(img_out.cols, 608);
  EXPECT_EQ(img_out.rows, 608);
  nppiFree(npp8u_ptr_in);
  nppiFree(npp8u_ptr_out);
}

TEST(NPPWrapperTest, test_resizer)
{
  cv::Mat img_out;
  const auto rows = g_img_in.rows;
  const auto cols = g_img_in.cols;
  int dummy;

  Npp8u* npp8u_ptr = nppiMalloc_8u_C3(cols, rows, &dummy);

  NPPResizer resizer(rows, cols, 0.25, 0.25);
  resizer.resize(npp8u_ptr, img_out);

  EXPECT_EQ(img_out.cols, cols / 4);
  EXPECT_EQ(img_out.rows, rows / 4);

  nppiFree(npp8u_ptr);
}

TEST(NPPWrapperTest, test_npp_resize_perf)
{
  cv::Mat resized;
  NPPResizer resizer(g_img_in.rows, g_img_in.cols, 0.2, 0.2);
  for (int i = 0; i < NUM_PERF_LOOPS; i++)
  {
    resizer.resize(g_img_in, resized);
  }
}

TEST(NPPWrapperTest, test_cv_resize_perf)
{
  cv::Mat resized;
  for (int i = 0; i < NUM_PERF_LOOPS; i++)
  {
    cv::resize(g_img_in, resized, cv::Size(), 0.2, 0.2);
  }
}
};  // namespace
