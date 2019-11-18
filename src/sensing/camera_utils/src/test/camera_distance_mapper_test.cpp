#include "car_model.h"
#if CAR_MODEL_IS_HINO
#include <gtest/gtest.h>
#include "camera_params.h"
#include "camera_distance_mapper.h"
#include "glog/logging.h"

TEST(CameraUtilsTest, test_distance_mapper_init)
{
  CameraDistanceMapper dist_mapper(camera::id::front_120);
  float sx = 0, sy = 0, sz = 0;  // spatial coordinates
  auto ret = dist_mapper.get_distance_raw_1920x1208(1920 / 2, 1208 / 2, &sx, &sy, &sz);
  LOG(INFO) << "sx: " << sx << " sy: " << sy << " sz: " << sz;

  EXPECT_EQ(ret, 0);
  EXPECT_NEAR(sx, 4.1, 0.01);
  // z-axis of ground points should be between 0 and -4 meters.
  EXPECT_TRUE(sz > -4);
  EXPECT_TRUE(sz < 0);

  auto mat = dist_mapper.remap_distance_in_undistorted_image();

  LOG(INFO) << mat.at<float>(1208 / 2, 1920 / 2);
}
#endif // CAR_MODEL_IS_HINO
