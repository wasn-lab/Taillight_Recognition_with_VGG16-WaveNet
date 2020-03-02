#include "car_model.h"
#include <gtest/gtest.h>

TEST(CarModelTestSuite, unique_car_model)
{
  EXPECT_EQ(CAR_MODEL_IS_B1 + CAR_MODEL_IS_B1_V2 + CAR_MODEL_IS_C + CAR_MODEL_IS_HINO, 1);
}


// Run all the tests that were declared with TEST()
int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
