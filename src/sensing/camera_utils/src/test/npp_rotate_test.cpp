#include <gtest/gtest.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include "npp_wrapper.h"
#include "camera_utils_test.h"

namespace npp_wrapper_test
{
const cv::Mat g_img_in = cv::imread(CAMERA_UTILS_TEST_DATA_DIR "/front_120_raw_10000.jpg");

TEST(NPPWrapperTest, test_npp_rotate_perf)
{
  cv::Mat out;
  for (int i = 0; i < NUM_PERF_LOOPS; i++)
  {
    npp_wrapper::rotate(g_img_in, out, 90);
  }
}

TEST(NPPWrapperTest, test_cv_rotate_perf)
{
  cv::Mat out;
  for (int i = 0; i < NUM_PERF_LOOPS; i++)
  {
    cv::rotate(g_img_in, out, cv::ROTATE_90_CLOCKWISE);
  }
}
};  // namespace
