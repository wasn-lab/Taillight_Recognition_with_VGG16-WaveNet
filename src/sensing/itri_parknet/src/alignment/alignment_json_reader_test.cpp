#include <gtest/gtest.h>
#include "parknet.h"
#include "alignment_json_reader.h"

TEST(AlignmentJsonReaderTest, test_read_distance_from_json)
{
  std::string filename(PARKNET_TEST_DATA_DIR "/spatial_points_test.json");
  cv::Point3d** dist_in_cm;
  dist_in_cm = new cv::Point3d*[2];
  for (int i = 0; i < 2; i++)
  {
    dist_in_cm[i] = new cv::Point3d[3];
  }

  auto ret = alignment::read_distance_from_json(filename, dist_in_cm, 2, 3);
  EXPECT_EQ(ret, 0);
  EXPECT_EQ(dist_in_cm[1][2].x, 200);
  EXPECT_EQ(dist_in_cm[1][2].y, 300);

  for (int i = 0; i < 2; i++)
  {
    delete[] dist_in_cm[i];
  }
  delete[] dist_in_cm;
}
