#include <gtest/gtest.h>
#include "deeplab_segmenter.h"

namespace deeplab
{

class DeeplabSegmenterTest : public testing::Test
{
protected:
  void SetUp() override
  {
  }
  void TearDown() override
  {
  }

  DeeplabSegmenter segmenter_;
};

TEST_F(DeeplabSegmenterTest, test_segment)
{
}

};  // namespace deeplab

