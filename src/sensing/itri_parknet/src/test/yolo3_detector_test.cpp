#include <gtest/gtest.h>
#include "parknet.h"
#include "yolo3_detector.h"
#include "parknet_image_utils.h"

#if USE(DARKNET)
namespace yolo3detector_test
{
class Yolo3DetectorTest : public testing::Test
{
protected:
  void SetUp() override
  {
    if (!yolo3_detector_.is_initialized())
    {
      init_detector();
    }
    img_in_cvmat = cv::imread(PARKNET_TEST_DATA_DIR "/front_120_preprocessed_10428.jpg", CV_LOAD_IMAGE_COLOR);
  }

  void init_detector()
  {
    std::string net_cfg_file = V3_NETWORK_CFG_FILE;
    std::string net_weights_file = V3_NETWORK_WEIGHTS_FILE;
    std::string obj_names_file = V3_OBJECT_NAMES_FILE;
    yolo3_detector_.load(net_cfg_file, net_weights_file, 0.05, 0.1);
  }
  darknet::Yolo3Detector yolo3_detector_;
  cv::Mat img_in_cvmat;
};

TEST_F(Yolo3DetectorTest, test_init_detector)
{
  EXPECT_EQ(yolo3_detector_.is_initialized(), true);
  EXPECT_EQ(608, yolo3_detector_.get_network_width());
  EXPECT_EQ(608, yolo3_detector_.get_network_height());
  EXPECT_EQ(1920, img_in_cvmat.cols);
  EXPECT_EQ(1208, img_in_cvmat.rows);
}

TEST_F(Yolo3DetectorTest, test_detect)
{
  auto yolo3_img = parknet::fit_yolo3_image_size(img_in_cvmat);
  auto img_in_darknet = parknet::convert_to_darknet_image(yolo3_img);
  std::vector<RectClassScore<float> > detections;
  auto ret = yolo3_detector_.detect(detections, img_in_darknet);
  // NOTE: detections.size depends on detector's score_thresh
  EXPECT_EQ(ret, 0);
  EXPECT_GT(detections.size(), 1);
}

};      // namespace yolo3detector_test
#endif  // USE(DARKNET)
