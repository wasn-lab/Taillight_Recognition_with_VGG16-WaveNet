#include <gtest/gtest.h>
#include "parknet_alignment_node.h"
#include "parknet_logging.h"
#include <opencv2/opencv.hpp>
#include <memory>

namespace parknet_alignment_node_test
{
class ParknetAlignmentNodeTest : public testing::Test
{
protected:
  void SetUp() override
  {
  }

  std::unique_ptr<ParknetAlignmentNode> node;
};

TEST_F(ParknetAlignmentNodeTest, test_approx_nearest_points_if_necessary)
{
  node.reset(new ParknetAlignmentNode(3, 4));  // width=3, height=4
  const cv::Point3d validp(1, 3, 4);

  node->set_spatial_point(1, 2, validp);
  node->approx_nearest_points_if_necessary();

  for (int row = 0; row < node->image_height_; row++)
  {
    for (int col = 0; col < node->image_width_; col++)
    {
      EXPECT_EQ(node->get_spatial_point(row, col), validp);
    }
  }
}

TEST_F(ParknetAlignmentNodeTest, test_set_spatial_point)
{
  node.reset(new ParknetAlignmentNode(3, 4));  // width=3, height=4
  EXPECT_EQ(node->image_height_, 4);
  EXPECT_EQ(node->image_width_, 3);
  for (int row = 0; row < node->image_height_; row++)
  {
    for (int col = 0; col < node->image_width_; col++)
    {
      EXPECT_FALSE(node->spatial_point_is_valid(row, col));
    }
  }

  const cv::Point3d validp(1, 3, 4);
  node->set_spatial_point(1, 2, validp);
  EXPECT_TRUE(node->spatial_point_is_valid(1, 2));

  cv::Point valid_neighbor;
  EXPECT_TRUE(node->search_valid_neighbor(1, 1, &valid_neighbor));
  EXPECT_FALSE(node->search_valid_neighbor(1, 0, &valid_neighbor));
}
};  // namespace
