/*
   CREATER: ICL U300
   DATE: Aug, 2019
 */

#include <assert.h>
#include "opencv2/opencv.hpp"
#include "glog/logging.h"
#include "nppi_geometry_transforms.h"
#include "npp.h"
#include "npp_remapper_impl.h"
#include "camera_utils.h"

// Only accepts images of 3 channels(RGB) and each color is unsigned char.
NPPRemapperImpl::NPPRemapperImpl(const int rows, const int cols)
  : rows_(rows)
  , cols_(cols)
  , num_pixels_(rows * cols)
  , num_image_bytes_(rows * cols * 3)
  , image_line_steps_(cols * 3)
  , mapxy_line_steps_(cols * sizeof(Npp32f))
{
  int dummy;
  mapx_ptr_cuda_ = nppiMalloc_32f_C1(cols, rows, &dummy);
  CHECK(mapx_ptr_cuda_);
  mapy_ptr_cuda_ = nppiMalloc_32f_C1(cols, rows, &dummy);
  CHECK(mapy_ptr_cuda_);
  src_npp8u_ptr_cuda_ = nppiMalloc_8u_C3(cols, rows, &dummy);
  CHECK(src_npp8u_ptr_cuda_);
  dst_npp8u_ptr_cuda_ = nppiMalloc_8u_C3(cols, rows, &dummy);
  CHECK(dst_npp8u_ptr_cuda_);

  image_size_.width = cols;
  image_size_.height = rows;
  image_roi_.x = 0;
  image_roi_.y = 0;
  image_roi_.width = cols;
  image_roi_.height = rows;
  interpolation_mode_ = NPPI_INTER_LINEAR;

  cv::Mat mapx, mapy;
  camera::get_undistortion_maps(mapx, mapy);
  set_mapxy(mapx, mapy);
}

NPPRemapperImpl::~NPPRemapperImpl()
{
  nppiFree(mapx_ptr_cuda_);
  nppiFree(mapy_ptr_cuda_);
  nppiFree(src_npp8u_ptr_cuda_);
  nppiFree(dst_npp8u_ptr_cuda_);
}

int NPPRemapperImpl::remap(const cv::Mat& src, cv::Mat& dst)
{
  {
    std::lock_guard<std::mutex> lk(mu_);
    cudaMemcpyAsync(src_npp8u_ptr_cuda_, src.data, num_image_bytes_, cudaMemcpyHostToDevice, cudaStreamPerThread);
  }
  return remap(src_npp8u_ptr_cuda_, dst);
}

int NPPRemapperImpl::remap(const Npp8u* npp8u_ptr_in, cv::Mat& dst)
{
  assert(npp8u_ptr_in);
  remap(npp8u_ptr_in, dst_npp8u_ptr_cuda_);

  camera::release_cv_mat_if_necessary(dst);
  dst.create(rows_, cols_, CV_8UC3);
  cudaMemcpyAsync(dst.data, dst_npp8u_ptr_cuda_, num_image_bytes_, cudaMemcpyDeviceToHost, cudaStreamPerThread);
  return 0;
}

int NPPRemapperImpl::remap(const Npp8u* npp8u_ptr_in, Npp8u* npp8u_ptr_out)
{
  assert(npp8u_ptr_in);
  assert(npp8u_ptr_out);
  std::lock_guard<std::mutex> lk(mu_);
  NppStatus result = nppiRemap_8u_C3R(npp8u_ptr_in, image_size_, image_line_steps_, image_roi_, mapx_ptr_cuda_,
                                      mapxy_line_steps_, mapy_ptr_cuda_, mapxy_line_steps_, dst_npp8u_ptr_cuda_,
                                      image_line_steps_, image_size_, interpolation_mode_);
  assert(result == NPP_SUCCESS);
  cudaMemcpyAsync(npp8u_ptr_out, dst_npp8u_ptr_cuda_, num_image_bytes_, cudaMemcpyDeviceToDevice, cudaStreamPerThread);
  return result;
}

void NPPRemapperImpl::set_interpolation_mode(NppiInterpolationMode mode)
{
  interpolation_mode_ = mode;
}

int NPPRemapperImpl::set_mapxy(const cv::Mat& mapx, const cv::Mat& mapy)
{
  CHECK(mapx.rows == rows_);
  CHECK(mapx.cols == cols_);
  CHECK(mapy.rows == rows_);
  CHECK(mapy.cols == cols_);
  VLOG(1) << "mapx type: " << mapx.type() << ", mapy type: " << mapy.type();
  if ((mapx.type() == CV_16SC2) && (mapy.type() == CV_16UC1))
  {
    LOG(INFO) << "convert mapxy type to CV_32FC1.";
    cv::convertMaps(mapx, mapy, mapx_, mapy_, CV_32FC1);
  }
  else
  {
    VLOG(1) << "use default mapxy type.";
    mapx_ = mapx;
    mapy_ = mapy;
  }
  CHECK(mapx_.type() == CV_32FC1);
  CHECK(mapy_.type() == CV_32FC1);

  cudaMemcpyAsync(mapx_ptr_cuda_, mapx_.data, num_pixels_ * sizeof(Npp32f), cudaMemcpyHostToDevice,
                  cudaStreamPerThread);
  cudaMemcpyAsync(mapy_ptr_cuda_, mapy_.data, num_pixels_ * sizeof(Npp32f), cudaMemcpyHostToDevice,
                  cudaStreamPerThread);

  is_mapxy_initialized_ = true;
  mapx_.release();
  mapy_.release();
  return 0;
}
