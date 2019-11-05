#include "car_model.h"
#if CAR_MODEL_IS_HINO
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
#if 1
const CameraDistanceMapper g_dist_mapper_front_120(::camera::id::front_120);
const CameraDistanceMapper g_dist_mapper_right_120(::camera::id::right_120);

TEST(ParknetAdvertiseUtilsTest, test_convert_2_corners_to_parking_slot_result_in_front_120_success)
{
  std::vector<RectClassScore<float> > corners;

  RectClassScore<float> corner;
  corner.x = 369;
  corner.y = 50 + ::camera::top_border;
  corner.w = 26;
  corner.h = 26;
  corners.emplace_back(corner);
  corner.x = 393;
  corner.y = 120 + ::camera::top_border;
  corners.emplace_back(corner);

  msgs::ParkingSlotResult psr =
      convert_2_corners_to_parking_slot_result_in_front_120(camera::front_120_e, corners, g_dist_mapper_front_120);
  EXPECT_EQ(psr.parking_slots.size(), 1);
  EXPECT_TRUE(psr.parking_slots[0].marking_points[1].y < -11);
}

TEST(ParknetAdvertiseUtilsTest, test_convert_2_corners_to_parking_slot_result_in_front_120_2)
{
  std::vector<RectClassScore<float> > corners;

  // corners appear when car is backing into the slot.
  RectClassScore<float> corner;
  corner.x = 131;
  corner.y = 322 + ::camera::top_border;
  corner.w = 26;
  corner.h = 26;
  corners.emplace_back(corner);
  corner.x = 474;
  corner.y = 326 + ::camera::top_border;
  corners.emplace_back(corner);

  msgs::ParkingSlotResult psr =
      convert_2_corners_to_parking_slot_result_in_front_120(camera::front_120_e, corners, g_dist_mapper_front_120);
  EXPECT_EQ(psr.parking_slots.size(), 0);
}

TEST(ParknetAdvertiseUtilsTest, test_convert_2_corners_to_parking_slot_result_in_right_120_success)
{
  std::vector<RectClassScore<float> > corners;

  RectClassScore<float> corner;
  corner.x = 249;
  corner.y = 192 + ::camera::top_border;
  corner.w = 26;
  corner.h = 26;
  corners.emplace_back(corner);
  corner.x = 491;
  corner.y = 302 + ::camera::top_border;
  corners.emplace_back(corner);

  msgs::ParkingSlotResult psr =
      convert_2_corners_to_parking_slot_result_in_right_120(camera::right_120_e, corners, g_dist_mapper_right_120);
  EXPECT_EQ(psr.parking_slots.size(), 1);
  // mps[0..1] are roughly (-4.97, -2.04) (-1.51, -2.90)
  bool extend_to_y_neg = false;
  for (int i = 0; i < 4; i++)
  {
    if (psr.parking_slots[0].marking_points[i].y < -10)
    {
      extend_to_y_neg = true;
    }
  }
  EXPECT_TRUE(extend_to_y_neg);
}

TEST(ParknetAdvertiseUtilsTest, test_convert_4_corners_to_parking_slot_result_near_border)
{
  std::vector<RectClassScore<float> > corners;

  RectClassScore<float> corner;
  corner.x = 419;
  corner.y = 206 + ::camera::top_border;
  corner.w = 26;
  corner.h = 26;
  corners.emplace_back(corner);
  corner.x = 593;
  corner.y = 333 + ::camera::top_border;
  corners.emplace_back(corner);
  corner.x = 560;
  corner.y = 79 + ::camera::top_border;
  corners.emplace_back(corner);
  corner.x = 477;
  corner.y = 17 + ::camera::top_border;
  corners.emplace_back(corner);

  msgs::ParkingSlotResult psr =
      convert_4_corners_to_parking_slot_result(camera::right_120_e, corners, g_dist_mapper_right_120);
  EXPECT_EQ(psr.parking_slots.size(), 0);
}

TEST(ParknetAdvertiseUtilsTest, test_convert_corner_to_marking_point)
{
  RectClassScore<float> corner;
  corner.x = 100;
  corner.y = 150;
  corner.w = 20;
  corner.h = 24;

  const msgs::MarkingPoint mp = convert_corner_to_marking_point(corner, g_dist_mapper_front_120);
  EXPECT_EQ(int(mp.x * 100), 1018);
  EXPECT_EQ(int(mp.y * 100), 800);
  EXPECT_EQ(int(mp.z), -3);
}
#endif
}  // parknet
#endif // CAR_MODEL_IS_HINO
