#include <gtest/gtest.h>
#include <memory>
#include <glog/logging.h>
#include "parknet.h"
#include "parknet_logging.h"
#include "parknet_node_impl.h"
#include "parknet_test_params.h"
#include "camera_params.h"
#include "npp.h"
#include "npp_utils.h"

static const auto g_img_in = cv::imread(PARKNET_TEST_DATA_DIR "/front_120_raw_10000.jpg", CV_LOAD_IMAGE_COLOR);

class ParknetNodeImplTest : public testing::Test
{
protected:
  void SetUp() override
  {
    parknet_node_ptr_.reset(new ParknetNodeImpl());
    parknet_node_ptr_->on_init();
    parknet_node_ptr_->subscribe_and_advertise_topics();
    parknet_node_ptr_->set_unittest_mode(true);
  }
  std::unique_ptr<ParknetNodeImpl> parknet_node_ptr_;
};

TEST_F(ParknetNodeImplTest, test_on_inference_cv)
{
  std::vector<cv::Mat> mats;
  mats.emplace_back(g_img_in);
  mats.emplace_back(g_img_in);
  mats.emplace_back(g_img_in);
  auto num_detections = parknet_node_ptr_->on_inference(mats);
  EXPECT_GE(num_detections, 10);  // Detect at least 10 parking lot corners.
}

TEST_F(ParknetNodeImplTest, test_on_inference_gpu)
{
  int dummy;
  std::vector<Npp8u*> npp8u_ptrs_cuda;
  Npp8u* npp8u_ptrs[3];
  for (int i = 0; i < 3; i++)
  {
    npp8u_ptrs[i] = nppiMalloc_8u_C3(camera::raw_image_cols, camera::raw_image_rows, &dummy);
    npp_wrapper::cvmat_to_npp8u_ptr(g_img_in, npp8u_ptrs[i]);
    npp8u_ptrs_cuda.emplace_back(npp8u_ptrs[i]);
  }

  auto num_detections = parknet_node_ptr_->on_inference(npp8u_ptrs_cuda, camera::num_raw_image_bytes);
  EXPECT_GE(num_detections, 10);  // Detect at least 10 parking lot corners.
  for (int i = 0; i < 3; i++)
  {
    nppiFree(npp8u_ptrs[i]);
  }
}

TEST_F(ParknetNodeImplTest, perf_on_inference_cv)
{
  std::vector<cv::Mat> mats;
  mats.emplace_back(g_img_in);
  mats.emplace_back(g_img_in);
  mats.emplace_back(g_img_in);

  for (int i = 0; i < NUM_LOOPS; i++)
  {
    parknet_node_ptr_->on_inference(mats);
  }
}

TEST_F(ParknetNodeImplTest, perf_on_inference_gpu)
{
  int dummy;
  std::vector<Npp8u*> npp8u_ptrs_cuda;
  Npp8u* npp8u_ptrs[3];
  for (int i = 0; i < 3; i++)
  {
    npp8u_ptrs[i] = nppiMalloc_8u_C3(camera::raw_image_cols, camera::raw_image_rows, &dummy);
    npp_wrapper::cvmat_to_npp8u_ptr(g_img_in, npp8u_ptrs[i]);
    npp8u_ptrs_cuda.emplace_back(npp8u_ptrs[i]);
  }

  for (int i = 0; i < NUM_LOOPS; i++)
  {
    parknet_node_ptr_->on_inference(npp8u_ptrs_cuda, camera::num_raw_image_bytes);
  }

  for (int i = 0; i < 3; i++)
  {
    nppiFree(npp8u_ptrs[i]);
  }
}
