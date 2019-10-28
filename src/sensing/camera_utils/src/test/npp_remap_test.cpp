#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include "npp_wrapper.h"
#include "camera_utils_test.h"
#include "camera_utils_defs.h"
#include "npp_remapper.h"
#include "npp_utils.h"
#include "camera_params.h"
#include "camera_utils.h"
#include "npp.h"

namespace camera
{
const cv::Mat g_img_in = cv::imread(CAMERA_UTILS_TEST_DATA_DIR "/front_120_raw_10000.jpg");
cv::Mat g_mapx, g_mapy;

TEST(NPPWrapperTest, test_remap)
{
  camera::get_undistortion_maps(g_mapx, g_mapy);
  cv::Mat out;

  auto ret = npp_wrapper::remap(g_img_in, out, g_mapx, g_mapy);
  // cv::imwrite("front_120_raw_10000_undistorted.jpg", out);
  // EXPECT_EQ(camera::calc_cvmat_checksum(out), 1585756982);  // not portable.
  EXPECT_EQ(ret, NPP_SUCCESS);
  EXPECT_EQ(out.rows, camera::raw_image_height);
  EXPECT_EQ(out.cols, camera::raw_image_width);
}

TEST(NPPWrapperTest, test_remapper)
{
  cv::Mat out, out2;
  int dummy;
  NPPRemapper remapper(g_img_in.rows, g_img_in.cols);

  remapper.remap(g_img_in, out);
  auto checksum1 = camera::calc_cvmat_checksum(out);

  Npp8u* npp8u_ptr = nppiMalloc_8u_C3(g_img_in.rows, g_img_in.cols, &dummy);
  npp_wrapper::cvmat_to_npp8u_ptr(g_img_in, npp8u_ptr);
  remapper.remap(npp8u_ptr, out2);
  auto checksum2 = camera::calc_cvmat_checksum(out2);
  nppiFree(npp8u_ptr);

  EXPECT_EQ(checksum1, checksum2);
}

TEST(NPPWrapperTest, test_npp_remap_perf)
{
  const cv::Mat expected = cv::imread(CAMERA_UTILS_TEST_DATA_DIR "/front_120_raw_10000_undistorted.ppm");
  NPPRemapper remapper(g_img_in.rows, g_img_in.cols);
  cv::Mat out;
  remapper.set_mapxy(g_mapx, g_mapy);
  for (int i = 0; i < NUM_PERF_LOOPS; i++)
  {
    remapper.remap(g_img_in, out);
    // EXPECT_TRUE(npp_wrapper::cvmats_are_equal(out, expected));
  }
}

TEST(NPPWrapperTest, test_cv_remap_32fc1_perf)
{
  cv::Mat out;
  for (int i = 0; i < NUM_PERF_LOOPS; i++)
  {
    cv::remap(g_img_in, out, g_mapx, g_mapy, cv::INTER_LINEAR);
  }
}

};  // namespace
