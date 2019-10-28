#include <gtest/gtest.h>
#include <glog/logging.h>
#include "parknet_camera.h"
#include "camera_params.h"
#include "camera_distance_mapper.h"
#include "parknet_advertise_utils.h"
#include "rect_class_score.h"
#include "msgs/ParkingSlot.h"

namespace parknet
{
TEST(ParknetAdvertiseUtilsTest, test_convert_2_marking_points_to_parking_slot_success)
{
  msgs::MarkingPoint mps[2];
  msgs::ParkingSlot pslot;

  mps[0].x = 4.7;
  mps[0].y = 1.2;
  mps[1].x = 4.7;
  mps[1].y = -2.4;
  auto ret = convert_2_marking_points_to_parking_slot(mps, &pslot);
  EXPECT_TRUE(ret);
  EXPECT_NEAR(pslot.marking_points[0].x, 4.7, 0.001);
  EXPECT_NEAR(pslot.marking_points[0].y, -2.4, 0.001);
  EXPECT_NEAR(pslot.marking_points[1].x, 4.7, 0.001);
  EXPECT_NEAR(pslot.marking_points[1].y, 1.2, 0.001);
  EXPECT_NEAR(pslot.marking_points[2].x, -4.3, 0.001);
  EXPECT_NEAR(pslot.marking_points[2].y, 1.2, 0.001);
  EXPECT_NEAR(pslot.marking_points[3].x, -4.3, 0.001);
  EXPECT_NEAR(pslot.marking_points[3].y, -2.4, 0.001);
  for (int i = 0; i < 4; i++)
  {
    VLOG(2) << "mp " << i << ":" << pslot.marking_points[i].x << " " << pslot.marking_points[i].y;
  }
}

TEST(ParknetAdvertiseUtilsTest, test_convert_2_marking_points_to_parking_slot_failed)
{
  msgs::MarkingPoint mps[2];
  msgs::ParkingSlot pslot;

  mps[0].x = 4.7;
  mps[0].y = 1.2;
  mps[1].x = 4.6;
  mps[1].y = -2;
  auto ret = convert_2_marking_points_to_parking_slot(mps, &pslot);
  EXPECT_FALSE(ret);
}

TEST(ParknetAdvertiseUtilsTest, test_convert_3_marking_points_to_parking_slot_success)
{
  msgs::MarkingPoint mps[3];
  msgs::ParkingSlot pslot;

  mps[0].x = 4.7;
  mps[0].y = 1.2;
  mps[1].x = 1;
  mps[1].y = 1.2;
  mps[2].x = 1;
  mps[2].y = 10.2;

  auto ret = convert_3_marking_points_to_parking_slot(mps, &pslot);
  EXPECT_TRUE(ret);
  for (int i = 0; i < 4; i++)
  {
    VLOG(2) << "mp " << i << ":" << pslot.marking_points[i].x << " " << pslot.marking_points[i].y;
  }
}

TEST(ParknetAdvertiseUtilsTest, test_convert_3_marking_points_to_parking_slot_failed)
{
  msgs::MarkingPoint mps[3];
  msgs::ParkingSlot pslot;

  mps[0].x = 1;
  mps[0].y = 2;
  mps[1].x = 3;
  mps[1].y = 4;
  mps[2].x = 5;
  mps[2].y = 6;

  auto ret = convert_3_marking_points_to_parking_slot(mps, &pslot);
  EXPECT_FALSE(ret);
}

TEST(ParknetAdvertiseUtilsTest, test_infer_4th_marking_point_found)
{
  msgs::MarkingPoint mps[3];
  msgs::MarkingPoint mp4;

  mps[0].x = 4.7;
  mps[0].y = 1.2;
  mps[1].x = 1;
  mps[1].y = 1.2;
  mps[2].x = 1;
  mps[2].y = 10.2;

  auto ret = infer_4th_marking_point(mps, &mp4);

  EXPECT_TRUE(ret);
  EXPECT_NEAR(mp4.x, 4.7, 0.001);
  EXPECT_NEAR(mp4.y, 10.2, 0.001);
}

TEST(ParknetAdvertiseUtilsTest, test_infer_4th_marking_point_not_found)
{
  msgs::MarkingPoint mps[3];
  msgs::MarkingPoint mp4;

  mps[0].x = 1;
  mps[0].y = 2;
  mps[1].x = 3;
  mps[1].y = 4;
  mps[2].x = 5;
  mps[2].y = 6;

  auto ret = infer_4th_marking_point(mps, &mp4);

  EXPECT_FALSE(ret);
  EXPECT_EQ(mp4.x, 0);
  EXPECT_EQ(mp4.y, 0);
}

TEST(ParknetAdvertiseUtilsTest, test_is_valid_parking_slot)
{
  msgs::ParkingSlot pslot;
  msgs::MarkingPoint mps[4];

  mps[0].x = 1;
  mps[0].y = 10.2;
  mps[1].x = 1;
  mps[1].y = 1.2;
  mps[2].x = 4.6;
  mps[2].y = 1.2;
  mps[3].x = 4.6;
  mps[3].y = 10.2;
  for (int i = 0; i < 4; i++)
  {
    pslot.marking_points.push_back(mps[i]);
  }
  EXPECT_TRUE(is_valid_parking_slot(pslot));

  pslot.marking_points.clear();
  mps[0].x = 1;
  mps[0].y = 10.2;
  mps[1].x = 1;
  mps[1].y = 1.2;
  mps[2].x = 4.3;
  mps[2].y = 1.2;
  mps[3].x = 4.3;
  mps[3].y = 10.2;
  for (int i = 0; i < 4; i++)
  {
    pslot.marking_points.push_back(mps[i]);
  }
  EXPECT_FALSE(is_valid_parking_slot(pslot));
}

TEST(ParknetAdvertiseUtilsTest, test_is_valid_edge_length)
{
  EXPECT_FALSE(is_valid_edge_length(3.34));
  EXPECT_TRUE(is_valid_edge_length(3.35));
  EXPECT_TRUE(is_valid_edge_length(3.6));
  EXPECT_TRUE(is_valid_edge_length(3.85));
  EXPECT_FALSE(is_valid_edge_length(3.86));

  EXPECT_FALSE(is_valid_edge_length(8.74));
  EXPECT_TRUE(is_valid_edge_length(8.75));
  EXPECT_TRUE(is_valid_edge_length(9));
  EXPECT_TRUE(is_valid_edge_length(9.25));
  EXPECT_FALSE(is_valid_edge_length(9.26));
}

TEST(ParknetAdvertiseUtilsTest, test_area_of_parking_slot)
{
  msgs::ParkingSlot pslot;
  msgs::MarkingPoint mps[4];

  mps[0].x = 1;
  mps[0].y = 10.2;
  mps[1].x = 1;
  mps[1].y = 1.2;
  mps[2].x = 4.6;
  mps[2].y = 1.2;
  mps[3].x = 4.6;
  mps[3].y = 10.2;
  for (int i = 0; i < 4; i++)
  {
    pslot.marking_points.push_back(mps[i]);
  }
  EXPECT_TRUE(std::abs(area_of_parking_slot(pslot) - 9 * 3.6) <= 0.0001);
}

TEST(ParknetAdvertiseUtilsTest, test_is_marking_point_near_image_border)
{
  msgs::MarkingPoint mp;

  mp.x = -8.0;
  mp.y = -6.24;
  mp.z = -3;
  EXPECT_TRUE(is_marking_point_near_image_border(camera::right_120_e, mp));

  mp.x = 0;
  EXPECT_TRUE(is_marking_point_near_image_border(camera::right_120_e, mp));
}

TEST(ParknetAdvertiseUtilsTest, test_sort_corners_couterclockwise)
{
  msgs::ParkingSlot pslot;
  msgs::MarkingPoint mps[4];

  mps[0].x = 1;
  mps[0].y = 1.2;
  mps[1].x = 4.6;
  mps[1].y = 10.2;
  mps[2].x = 4.6;
  mps[2].y = 1.2;
  mps[3].x = 1;
  mps[3].y = 10.2;
  for (int i = 0; i < 4; i++)
  {
    pslot.marking_points.push_back(mps[i]);
  }

  const auto sorted_pslot = sort_corners_couterclockwise(pslot);
  const int sorted_indexes[] = { 1, 3, 2, 0 };  // map to origin index
  for (int i = 0; i < 4; i++)
  {
    auto si = sorted_indexes[i];
    EXPECT_NEAR(sorted_pslot.marking_points[si].x, pslot.marking_points[i].x, 0.001);
    EXPECT_NEAR(sorted_pslot.marking_points[si].y, pslot.marking_points[i].y, 0.001);
    EXPECT_NEAR(sorted_pslot.marking_points[si].z, pslot.marking_points[i].z, 0.001);
  }
}

TEST(ParknetAdvertiseUtilsTest, test_convert_corner_to_yolov3_image_xy)
{
  RectClassScore<float> corner;
  corner.x = 100;
  corner.y = 150;
  corner.w = 20;
  corner.h = 24;

  int im_x = 0, im_y = 0;
  parknet::convert_corner_to_yolov3_image_xy(corner, &im_x, &im_y);
  EXPECT_EQ(im_x, 110);
  EXPECT_EQ(im_y, 162);
}

}  // parknet
