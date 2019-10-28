#include <gtest/gtest.h>
#include <memory>
#include "parknet.h"
#include "parknet_pcd_manager.h"
#include "parknet_logging.h"
#include "drivenet/Alignment.h"

class Alignment;

class ParknetPCDManagerTest : public testing::Test
{
protected:
  void SetUp() override
  {
    parknet::read_pcd_file(PARKNET_TEST_DATA_DIR "/1556730036.pcd", pcd);
    alignment_ptr.reset(new Alignment());
  }

  ParknetPCDManager pcd_manager;
  PointCloud pcd;
  std::unique_ptr<Alignment> alignment_ptr;
};

TEST_F(ParknetPCDManagerTest, test_set_pcd)
{
  const int cam_id = 1;
  EXPECT_EQ(pcd.points.size(), 90147);
  pcd_manager.set_pcd(pcd, cam_id);
  auto out_pcd = pcd_manager.get_pcd(cam_id);
  EXPECT_EQ(out_pcd.points.size(), pcd.points.size());
}

// Skip slow test case.
TEST_F(ParknetPCDManagerTest, DISABLED_test_alignment_results)
// TEST_F(ParknetPCDManagerTest, test_alignment_results)
{
  int x, y;
  double* xyz;
  int num_checked = 0;
  int num_nonzero = 0;

  for (x = 860; x < 860 + 50; x++)
  {
    for (y = 504; y < 504 + 50; y++)
    {
      num_checked++;
      xyz = alignment_ptr->value_distance_array(pcd, x, y, 5);  // 5: maps to front 120
      if ((!xyz[0]) && (!xyz[1]) && (!xyz[2]))
      {
        continue;
      }
      LOG_INFO << "(" << x << ", " << y << ") -> "
               << "(" << xyz[0] << ", " << xyz[1] << ", " << xyz[2] << ")";
      num_nonzero++;
    }
  }
  LOG_INFO << num_nonzero << "/" << num_checked << " map to points != (0,0,0).";
}

TEST_F(ParknetPCDManagerTest, test_alignment_result_center)
{
  int x = 864, y = 509;
  double* xyz;

  xyz = alignment_ptr->value_distance_array(pcd, x, y, 5);  // 5: maps to front 120
  // (864, 509) -> (5.19194, 0.589009, -2.70523)
  EXPECT_TRUE(xyz[0] - 5.19194 < 0.00001);
  EXPECT_TRUE(xyz[1] - 0.589009 < 0.00001);
}

TEST_F(ParknetPCDManagerTest, dump_matrix)
{
  int x = 864, y = 509;
  double* xyz;
  for (int i = 1; i <= 9; i++)
  {
    std::cout << "-----------------\n";
    std::cout << "i = " << i << "\n";
    xyz = alignment_ptr->value_distance_array(pcd, x, y, i);
  }
}
