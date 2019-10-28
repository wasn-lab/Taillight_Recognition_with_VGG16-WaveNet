#include <gtest/gtest.h>
#include "parknet.h"
#include "alignment_json_writer.h"
#include <iostream>
#include <memory>
#include <fstream>

TEST(AlignmentJsonWriterTest, test_jsonize_spatial_points)
{
  cv::Point3d** spatial_points;
  spatial_points = new cv::Point3d*[2];
  for (int i = 0; i < 2; i++)
  {
    spatial_points[i] = new cv::Point3d[3];
  }

  for (int row = 0; row < 2; row++)
  {
    for (int col = 0; col < 3; col++)
    {
      spatial_points[row][col].x = row + 1;
      spatial_points[row][col].y = col + 1;
      spatial_points[row][col].z = -3;
    }
  }
  auto jdata = alignment::jsonize_spatial_points(spatial_points, 2, 3);
  std::string expected;
  std::ifstream ifs(PARKNET_TEST_DATA_DIR "/spatial_points_test.json");
  if (ifs.is_open())
  {
    ifs >> expected;
    ifs.close();
  }

  EXPECT_EQ(jdata, expected);
  std::ofstream ofs("spatial_points_test.json");
  ofs << jdata;

  for (int i = 0; i < 2; i++)
  {
    delete[] spatial_points[i];
  }
  delete[] spatial_points;
}
