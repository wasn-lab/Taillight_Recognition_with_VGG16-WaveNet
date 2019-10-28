#include "parknet.h"
#if USE(TENSORRT)
#include "trt_yolo3_detector.h"
#include <gtest/gtest.h>
#include "parknet_image_utils.h"
#include "camera_utils.h"
#include "camera_params.h"
#include "npp.h"
#include "npp_utils.h"
#include "parknet_test_params.h"

namespace trtyolo3detector_test
{
const auto g_img_in = cv::imread(PARKNET_TEST_DATA_DIR "/front_120_preprocessed_10428.jpg", CV_LOAD_IMAGE_COLOR);

class TRTYolo3DetectorTest : public testing::Test
{
protected:
  void SetUp() override
  {
    int dummy;
    npp8u_ptr = nppiMalloc_8u_C3(camera::yolov3_image_cols, camera::yolov3_image_rows, &dummy);
    npp32f_ptr = nppiMalloc_32f_C3(camera::yolov3_image_cols, camera::yolov3_image_rows, &dummy);
    assert(npp8u_ptr);
    assert(npp32f_ptr);

    camera::fit_yolov3_image_size(g_img_in, yolov3_img);
  }
  void TearDown() override
  {
    nppiFree(npp8u_ptr);
    nppiFree(npp32f_ptr);
  }

  parknet::TRTYolo3Detector detector_;
  Npp8u* npp8u_ptr;
  Npp32f* npp32f_ptr;
  cv::Mat yolov3_img;
};

TEST_F(TRTYolo3DetectorTest, test_detect)
{
  std::vector<RectClassScore<float> > detections_cv;
  std::vector<RectClassScore<float> > detections_gpu;

  // Pass cv::Mat to detect()
  auto ret = detector_.detect(detections_cv, yolov3_img, parknet::camera::front_120_e);
  // NOTE: detections.size depends on detector's score_thresh
  EXPECT_EQ(ret, 0);
  EXPECT_GE(detections_cv.size(), 0);  // Currently we only check that there will be detections.

#if 0
  // Pass Npp8u* to detect()
  npp_wrapper::cvmat_to_npp8u_ptr(yolov3_img, npp8u_ptr);
  ret = detector_.detect(detections_gpu, npp8u_ptr, parknet::camera::front_120_e);
  EXPECT_EQ(detections_cv.size(), detections_gpu.size());
  for (size_t i = 0; i < detections_cv.size(); i++)
  {
    auto det_cv = detections_cv[i];
    auto det_gpu = detections_gpu[i];
    EXPECT_EQ(det_cv.toString(), det_gpu.toString());
  }
  EXPECT_EQ(ret, 0);
#endif
}

TEST_F(TRTYolo3DetectorTest, perf_detect_cv)
{
  for (int i = 0; i < NUM_LOOPS; i++)
  {
    std::vector<RectClassScore<float> > detections;
    detector_.detect(detections, yolov3_img, parknet::camera::front_120_e);
  }
}

#if 0
TEST_F(TRTYolo3DetectorTest, perf_detect_gpu)
{
  npp_wrapper::cvmat_to_npp8u_ptr(yolov3_img, npp8u_ptr);
  for (int i = 0; i < NUM_LOOPS; i++)
  {
    std::vector<RectClassScore<float> > detections;
    detector_.detect(detections, npp8u_ptr, parknet::camera::front_120_e);
  }
}
#endif

};  // namespace trtyolo3detector_test

#endif  // USE(TENSORRT)
