/*
   CREATER: ICL U300
   DATE: Aug, 2019
 */
#include <assert.h>
#include "npp_wrapper_dn.h"
#include "nppi_geometry_transforms.h"
#include "npp.h"

namespace DriveNet_npp
{
int resize(const cv::Mat& src, cv::Mat& dst, const double hscale, const double wscale, const int channel,
           const NppiInterpolationMode interpolation_mode)
{
  assert(hscale > 0);
  assert(wscale > 0);

  int dummy;
  const NppiSize src_size = {.width = src.cols, .height = src.rows };
  const NppiRect src_roi = {.x = 0, .y = 0, .width = src.cols, .height = src.rows };
  const int src_line_steps = src.cols * channel;
  const int num_src_bytes = src.cols * src.rows * channel;
  Npp8u* src_npp8u_ptr;
  if (channel == 3)
    src_npp8u_ptr = nppiMalloc_8u_C3(src.cols, src.rows, &dummy);
  if (channel == 1)
    src_npp8u_ptr = nppiMalloc_8u_C1(src.cols, src.rows, &dummy);

  assert(src_npp8u_ptr);
  assert(num_src_bytes > 0);

  // const int dst_rows = src.rows * hscale;
  const int dst_rows = hscale;
  // const int dst_cols = src.cols * wscale;
  const int dst_cols = wscale;
  const int dst_line_steps = dst_cols * channel;
  const int num_dst_bytes = dst_cols * dst_rows * channel;
  const NppiSize dst_size = {.width = dst_cols, .height = dst_rows };
  const NppiRect dst_roi = {.x = 0, .y = 0, .width = dst_cols, .height = dst_rows };
  Npp8u* dst_npp8u_ptr;
  if (channel == 3)
    dst_npp8u_ptr = nppiMalloc_8u_C3(dst_cols, dst_rows, &dummy);
  if (channel == 1)
    dst_npp8u_ptr = nppiMalloc_8u_C1(dst_cols, dst_rows, &dummy);

  assert(dst_npp8u_ptr);
  assert(num_dst_bytes > 0);

  cudaMemcpy(src_npp8u_ptr, src.data, num_src_bytes, cudaMemcpyHostToDevice);

  NppStatus result;
  if (channel == 3)
    result = nppiResize_8u_C3R(src_npp8u_ptr, src_line_steps, src_size, src_roi, dst_npp8u_ptr, dst_line_steps,
                               dst_size, dst_roi, interpolation_mode);

  if (channel == 1)
    result = nppiResize_8u_C1R(src_npp8u_ptr, src_line_steps, src_size, src_roi, dst_npp8u_ptr, dst_line_steps,
                               dst_size, dst_roi, interpolation_mode);

  assert(result == NPP_SUCCESS);

  if (!dst.empty())
  {
    dst.release();
  }
  if (channel == 3)
    dst = cv::Mat(dst_rows, dst_cols, CV_8UC3);

  if (channel == 1)
    dst = cv::Mat(dst_rows, dst_cols, CV_8UC1);

  cudaMemcpy(dst.data, dst_npp8u_ptr, num_dst_bytes, cudaMemcpyDeviceToHost);
  nppiFree(src_npp8u_ptr);
  nppiFree(dst_npp8u_ptr);
  return result;
}

int rotate(const cv::Mat& src, cv::Mat& dst, const int rotation_degree, const NppiInterpolationMode interpolation_mode)
{
  assert((rotation_degree == 90) || (rotation_degree == 180) || (rotation_degree == 270));

  int dummy = 0;
  const NppiSize src_size = {.width = src.cols, .height = src.rows };
  const NppiRect src_roi = {.x = 0, .y = 0, .width = src.cols, .height = src.rows };
  int src_line_steps = src.cols * 3;
  const int num_src_bytes = src.cols * src.rows * 3;
  Npp8u* src_npp8u_ptr = nppiMalloc_8u_C3(src.cols, src.rows, &dummy);

  assert(src_npp8u_ptr);
  assert(num_src_bytes > 0);

  double rotated_bounding_box[2][2];

  nppiGetRotateBound(src_roi, rotated_bounding_box, rotation_degree, 0, 0);

  // parameters for 90 / 270 degrees
  const int num_dst_bytes = src.rows * src.cols * 3;
  int dst_rows = src.cols;
  int dst_cols = src.rows;
  int dst_line_steps = dst_cols * 3;
  // NppiSize dst_size = {.width = dst_cols, .height = dst_rows };
  NppiRect dst_roi = {.x = 0, .y = 0, .width = dst_cols, .height = dst_rows };
  // paramters for 180 degree
  if (rotation_degree == 180)
  {
    dst_rows = src.rows;
    dst_cols = src.cols;
    dst_line_steps = dst_cols * 3;
    // dst_size.width = dst_cols;
    // dst_size.height = dst_rows;
    dst_roi.width = dst_cols;
    dst_roi.height = dst_rows;
  }

  Npp8u* dst_npp8u_ptr = nppiMalloc_8u_C3(dst_cols, dst_rows, &dummy);
  const double shift_x = -rotated_bounding_box[0][0];
  const double shift_y = -rotated_bounding_box[0][1];

  assert(dst_npp8u_ptr);
  assert(num_dst_bytes > 0);

  cudaMemcpy(src_npp8u_ptr, src.data, num_src_bytes, cudaMemcpyHostToDevice);

  NppStatus result = nppiRotate_8u_C3R(src_npp8u_ptr, src_size, src_line_steps, src_roi, dst_npp8u_ptr, dst_line_steps,
                                       dst_roi, rotation_degree, shift_x, shift_y, interpolation_mode);
  assert(result == NPP_SUCCESS);

  if (!dst.empty())
  {
    dst.release();
  }
  dst = cv::Mat(dst_rows, dst_cols, CV_8UC3);

  cudaMemcpy(dst.data, dst_npp8u_ptr, num_dst_bytes, cudaMemcpyDeviceToHost);
  nppiFree(src_npp8u_ptr);
  nppiFree(dst_npp8u_ptr);
  return result;
}
int cvmat_to_npp8u_ptr(const cv::Mat& src, Npp8u* out_npp8u_ptr)
{
  assert(out_npp8u_ptr);  // callers are responsible for malloc it with sufficient size.
  assert(src.rows > 0);
  assert(src.cols > 0);
  const size_t num_bytes = src.rows * src.cols * src.channels();

  cudaMemcpy(out_npp8u_ptr, src.data, num_bytes, cudaMemcpyHostToDevice);

  return 0;
}
int npp8u_ptr_to_cvmat(const Npp8u* in_npp8u_ptr, const size_t in_num_bytes, cv::Mat& out_img, const int rows,
                       const int cols)
{
  assert(in_npp8u_ptr);
  size_t dim_size = rows * cols;
  size_t channels = in_num_bytes / dim_size;

  // channels == 3 -> RGB/BGR images
  // TODO: extend to other channels like 1, 4
  assert(in_num_bytes % dim_size == 0);
  assert(channels == 3);

  if (!out_img.empty())
  {
    out_img.release();
  }
  out_img.create(rows, cols, CV_8UC3);

  cudaMemcpy(out_img.data, in_npp8u_ptr, in_num_bytes, cudaMemcpyDeviceToHost);

  return 0;
}

};  // namespace
