#include <gtest/gtest.h>
#include "alignment_params.h"

TEST(AlignmentParamsTest, test_inv_r_t)
{
  const int cam_sn = 5;
  const auto& mat = alignment::get_invR_T(cam_sn);

  EXPECT_EQ(mat.rows, 3);
  EXPECT_EQ(mat.cols, 3);
  EXPECT_EQ(mat.at<double>(0, 0), -0.07728790000000001);
}

TEST(AlignmentParamsTest, test_inv_t_t)
{
  const int cam_sn = 5;
  const auto& mat = alignment::get_invT_T(cam_sn);

  EXPECT_EQ(mat.rows, 1);
  EXPECT_EQ(mat.cols, 3);
  EXPECT_EQ(mat.at<double>(0, 2), 3.9395786409569);
}

TEST(AlignmentParamsTest, test_g_alignment_dist_coeff_mat)
{
  const int cam_sn = 5;
  const auto& mat = alignment::get_alignment_dist_coeff_mat(cam_sn);
  EXPECT_EQ(mat.rows, 1);
  EXPECT_EQ(mat.cols, 5);
  EXPECT_EQ(mat.at<double>(0, 2), 0.0100694);
}

TEST(AlignmentParamsTest, test_g_alignment_camera_mat)
{
  const int cam_sn = 5;
  const auto& mat = alignment::get_alignment_camera_mat(cam_sn);
  EXPECT_EQ(mat.rows, 3);
  EXPECT_EQ(mat.cols, 3);
  EXPECT_EQ(mat.at<double>(1, 1), 1327.62);
}
