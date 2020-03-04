/*
   CREATER: ICL U300
   DATE: Mar, 2020
 */
#include "opencv2/opencv.hpp"
#include "npp_transformer.h"
#include "camera_params.h"
#include "camera_utils.h"

NPPTransformer::NPPTransformer(const int src_rows, const int src_cols, const int dst_rows, const int dst_cols)
  : src_rows_(src_rows)
  , src_cols_(src_cols)
  , dst_rows_(dst_rows)
  , dst_cols_(dst_cols)
  , num_src_bytes_(src_rows * src_cols * 3)
  , num_dst_bytes_(dst_rows * dst_cols * 3)
{
}

NPPTransformer::~NPPTransformer()
{
}

void NPPTransformer::transform(const Npp8u* npp8u_ptr_in, cv::Mat& dst)
{
  camera::release_cv_mat_if_necessary(dst);
  dst.create(dst_rows_, dst_cols_, CV_8UC3);
  cudaMemcpyAsync(dst.data, npp8u_ptr_in, num_dst_bytes_, cudaMemcpyDeviceToHost, cudaStreamPerThread);

};  // namespace
