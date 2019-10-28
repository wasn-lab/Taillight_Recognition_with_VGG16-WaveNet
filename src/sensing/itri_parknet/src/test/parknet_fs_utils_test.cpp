#include <gtest/gtest.h>
#include "parknet.h"
#include "parknet_fs_utils.h"

TEST(ParknetFsUtilsTest, test_is_file)
{
  std::string jpg_file(PARKNET_TEST_DATA_DIR "/front_120_preprocessed_10428.jpg");
  std::string jpg_dir(PARKNET_TEST_DATA_DIR);
  std::string not_existent_file(PARKNET_TEST_DATA_DIR "/parking_lotkkkkkkk.jpg");

  EXPECT_TRUE(parknet::is_file(jpg_file));
  EXPECT_FALSE(parknet::is_file(jpg_dir));
  EXPECT_FALSE(parknet::is_file(not_existent_file));
}

TEST(ParknetFsUtilsTest, test_get_trt_engine_fullpath)
{
  std::string weights_file = "/sensing/itri_parknet/weights/parknet_yolov3_190619_70963c.weights";
  std::string actual = parknet::get_trt_engine_fullpath(weights_file);
  EXPECT_EQ(actual, "/sensing/itri_parknet/weights/parknet_yolov3_190619_70963c.weights.tensorrt" +
                        std::to_string(TENSORRT_VERSION_MAJOR) + "." + std::to_string(TENSORRT_VERSION_MINOR) + "." +
                        std::to_string(TENSORRT_VERSION_PATCH) + ".engine");
}
