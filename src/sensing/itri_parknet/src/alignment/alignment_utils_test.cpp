#include <gtest/gtest.h>
#include <memory>
#include "parknet.h"
#include "parknet_pcd_manager.h"
#include "alignment_utils.h"

TEST(AlignmentUtilsTest, test_alignment_result_center)
{
  int x = 864, y = 509;
  PointCloud pcd;
  parknet::read_pcd_file(PARKNET_TEST_DATA_DIR "/1556730036.pcd", pcd);
  double* xyz;

  xyz = alignment::get_dist_by_image_point(pcd, x, y, 5);  // 5: maps to front 120

  // (864, 509) -> (5.19194, 0.589009, -2.70523)
  EXPECT_TRUE(xyz[0] - 5.19194 < 0.00001);
  EXPECT_TRUE(xyz[1] - 0.589009 < 0.00001);
}

TEST(AlignmentUtilsTest, test_map_pcd_point_to_image_point)
{
  pcl::PointXYZ pcd_point;
  pcd_point.x = 2.90047;
  pcd_point.y = 6.28646;
  pcd_point.z = -3.25145;

  auto ret = alignment::map_pcd_point_to_image_point(pcd_point, 5);
  EXPECT_EQ(ret.x, 10);
  EXPECT_EQ(ret.y, 883);
}

TEST(AlignmentUtilsTest, test_map_pcd_ground_point_to_image_point)
{
  pcl::PointXYZ pcd_point;
  pcd_point.x = 2.90047;
  pcd_point.y = 6.28646;
  pcd_point.z = -3.25145;
  auto ret = alignment::map_pcd_ground_point_to_image_point(pcd_point, 5);
  EXPECT_EQ(ret.x, 10);
  EXPECT_EQ(ret.y, 883);

  pcd_point.x = 2.90047;
  pcd_point.y = 6.28646;
  pcd_point.z = -4.25145;
  ret = alignment::map_pcd_ground_point_to_image_point(pcd_point, 5);
  EXPECT_EQ(ret.x, -1);
  EXPECT_EQ(ret.y, -1);
}

TEST(AlignmentUtilsTest, test_map_pcd_to_image_points)
{
  PointCloud pcd;
  parknet::read_pcd_file(PARKNET_TEST_DATA_DIR "/1556730036.pcd", pcd);

  auto ret = alignment::map_pcd_to_image_points(pcd, 5);  // 5: maps to front 120
  EXPECT_EQ(ret, 26625);
}

TEST(AlignmentUtilsTest, test_map_pcd_ground_to_image_points)
{
  PointCloud pcd;
  parknet::read_pcd_file(PARKNET_TEST_DATA_DIR "/1556730036.pcd", pcd);

  auto ret = alignment::map_pcd_ground_to_image_points(pcd, 5);  // 5: maps to front 120
  EXPECT_EQ(ret, 5212);
}
