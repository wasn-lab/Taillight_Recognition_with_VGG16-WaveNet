#include <gtest/gtest.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/opencv.hpp>
#include "glog/logging.h"
#include "npp.h"
#include "npp_utils.h"
#include "npp_remapper.h"
#include "camera_utils_test.h"
#include "camera_utils.h"
#include "camera_params.h"

namespace npp_wrapper_test
{
static const cv::Mat g_img_in = cv::imread(CAMERA_UTILS_TEST_DATA_DIR "/front_120_raw_10000.jpg");
static const cv::Mat g_img_rgba = cv::imread(CAMERA_UTILS_TEST_DATA_DIR "/rgba_ex.png", cv::IMREAD_UNCHANGED);

TEST(NPPWrapperTest, test_cvmat_npp8u_ptr_conversion)
{
  cv::Mat img;
  int dummy;
  Npp8u* npp8u_ptr = nppiMalloc_8u_C3(g_img_in.cols, g_img_in.rows, &dummy);
  const size_t num_bytes = g_img_in.rows * g_img_in.cols * g_img_in.channels();

  npp_wrapper::cvmat_to_npp8u_ptr(g_img_in, npp8u_ptr);
  npp_wrapper::npp8u_ptr_to_cvmat(npp8u_ptr, num_bytes, img, g_img_in.rows, g_img_in.cols);
  EXPECT_EQ(camera::calc_cvmat_checksum(img), camera::calc_cvmat_checksum(g_img_in));

  nppiFree(npp8u_ptr);
}

TEST(NPPWrapperTest, test_npp8u_ptr_c4_to_c3)
{
  cv::Mat out_img;
  int dummy;
  const int rows = g_img_rgba.rows;
  const int cols = g_img_rgba.cols;
  Npp8u* npp8u_ptr_c3 = nppiMalloc_8u_C3(cols, rows, &dummy);
  Npp8u* npp8u_ptr_c4 = nppiMalloc_8u_C4(cols, rows, &dummy);
  npp_wrapper::cvmat_to_npp8u_ptr(g_img_rgba, npp8u_ptr_c4);
  npp_wrapper::npp8u_ptr_c4_to_c3(npp8u_ptr_c4, rows, cols, npp8u_ptr_c3);
  npp_wrapper::npp8u_ptr_to_cvmat(npp8u_ptr_c3, rows * cols * 3, out_img, rows, cols);
  EXPECT_EQ(out_img.rows, rows);
  EXPECT_EQ(out_img.cols, cols);
  EXPECT_EQ(out_img.channels(), 3);
  EXPECT_EQ(out_img.type(), 16);

  // cv::imwrite("npp8u_c3.jpg", out_img);
  nppiFree(npp8u_ptr_c3);
  nppiFree(npp8u_ptr_c4);
}

// Test NPP API only.
TEST(NPPWrapperTest, DISABLED_test_rgb2yuv)
{
  int dummy;
  const int rows = 2;
  const int cols = 3;
  Npp8u* npp8u_ptr_in = nppiMalloc_8u_C3(cols, rows, &dummy);
  Npp8u* npp8u_ptr_out = nppiMalloc_8u_C1(rows * cols, 3, &dummy);

  unsigned char grid_in[rows][cols][3] = {
    { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 } }, { { 10, 11, 12 }, { 13, 14, 15 }, { 16, 17, 18 } },
  };

  unsigned char grid_out[rows * cols * 3] = { 0 };
  NppiSize roi = {.width = cols, .height = rows };

  cudaMemcpy(npp8u_ptr_in, grid_in, rows * cols * 3, cudaMemcpyHostToDevice);

  NppStatus status;
  Npp8u* const aDst[3] = { npp8u_ptr_out, npp8u_ptr_out + cols * rows, npp8u_ptr_out + cols * rows * 2 };
  status = nppiCopy_8u_C3P3R(npp8u_ptr_in, cols * 3, aDst, cols, roi);
  LOG(INFO) << "status: " << status;

  cudaMemcpy(grid_out, npp8u_ptr_out, rows * cols * 3, cudaMemcpyDeviceToHost);
  for (int c = 0; c < 3; c++)
  {
    for (int ij = 0; ij < rows * cols; ij++)
    {
      std::cout << int(grid_out[c * rows * cols + ij]) << " ";
    }
    std::cout << "\n";
  }

  nppiFree(npp8u_ptr_in);
  nppiFree(npp8u_ptr_out);
}

TEST(NPPWrapperTest, test_blob_from_image)
{
  int dummy;
  const int rows = g_img_in.rows;
  const int cols = g_img_in.cols;
  // prepare expected result:
  cv::Mat mat32fc3;
  std::vector<cv::Mat> vec;
  g_img_in.convertTo(mat32fc3, CV_32FC3, 1.0);
  vec.emplace_back(mat32fc3);
  auto blob_cv = cv::dnn::blobFromImages(vec, 1.0, cv::Size(cols, rows), cv::Scalar(0.0, 0.0, 0.0), false, false);
  auto expected_checksum = camera::calc_cvmat_checksum(blob_cv);

  // prepare actual result:
  const size_t num_bytes_out = rows * cols * g_img_in.channels() * sizeof(float);
  unsigned char* bytes_out = new unsigned char[num_bytes_out];
  assert(bytes_out);

  Npp8u* npp8u_ptr = nppiMalloc_8u_C3(cols, rows, &dummy);
  Npp32f* npp32f_ptr_in = nppiMalloc_32f_C3(cols, rows, &dummy);
  Npp32f* npp32f_ptr_out = nppiMalloc_32f_C1(cols * rows, 3, &dummy);

  npp_wrapper::cvmat_to_npp8u_ptr(g_img_in, npp8u_ptr);

  auto status = nppiConvert_8u32f_C3R(npp8u_ptr, cols * 3, npp32f_ptr_in, cols * 3 * sizeof(float),
                                      {.width = cols, .height = rows });
  assert(status == NPP_SUCCESS);

  // generate blob data for NN inference.
  npp_wrapper::blob_from_image(npp32f_ptr_in, rows, cols, npp32f_ptr_out);

  cudaMemcpy(bytes_out, npp32f_ptr_out, num_bytes_out, cudaMemcpyDeviceToHost);
  EXPECT_EQ(camera::calc_bytes_checksum(bytes_out, num_bytes_out), expected_checksum);

  nppiFree(npp8u_ptr);
  nppiFree(npp32f_ptr_in);
  nppiFree(npp32f_ptr_out);
  delete[] bytes_out;
}

};  // namespace
