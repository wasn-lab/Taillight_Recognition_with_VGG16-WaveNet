/*
   CREATER: ICL U300
   DATE: Aug, 2019
 */
#include <assert.h>
#include "glog/logging.h"
#include "npp_utils.h"
#include "npp.h"
#include "camera_params.h"

namespace npp_wrapper
{
int cvmat_to_npp8u_ptr(const cv::Mat& src, Npp8u* out_npp8u_ptr)
{
  assert(out_npp8u_ptr);  // callers are responsible for malloc it with sufficient size.
  assert(src.rows > 0);
  assert(src.cols > 0);
  const size_t num_bytes = src.rows * src.cols * src.channels();

  cudaMemcpyAsync(out_npp8u_ptr, src.data, num_bytes, cudaMemcpyHostToDevice, cudaStreamPerThread);

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
  (void)(channels);

  if (!out_img.empty())
  {
    out_img.release();
  }
  out_img.create(rows, cols, CV_8UC3);

  cudaMemcpyAsync(out_img.data, in_npp8u_ptr, in_num_bytes, cudaMemcpyDeviceToHost, cudaStreamPerThread);

  return 0;
}

int npp8u_ptr_c4_to_c3(const Npp8u* npp8u_ptr_c4, const int rows, const int cols, Npp8u* npp8u_ptr_c3)
{
  assert(npp8u_ptr_c4);
  assert(npp8u_ptr_c3);
  assert(rows > 0);
  assert(cols > 0);
  NppiSize img_size = {.width = cols, .height = rows };
  // const int bgr_order[3] = {2, 1, 0};
  const int rgb_order[3] = { 0, 1, 2 };
  nppiSwapChannels_8u_C4C3R(npp8u_ptr_c4, cols * 4, npp8u_ptr_c3, cols * 3, img_size, rgb_order);
  return 0;
}

/**
 * NPP-version of cv::dnn::blobFromImage, which rearranges color channels from
 * pixels of (R0, G0, B0), (R1, G1, B1), ... (Rn, Gn, Bn) into
 * R0, R1, ... Rn, G0, G1, ... Gn, B0, B1, ... Bn
 *
 * Note: 1. Callers are responsible to pass valid pointers that can be read/written.
 *       2. To match the behavior of cv, the data type here is 32f (float), not 8u (unsigned char)
 *
 * @param[in] npp32f_ptr_in, created by nppiMalloc_32f_C3(cols, rows, &dummy);
 * @param[in] rows - The number of rows (image height).
 * @param[in] cols - The number of columns (image width).
 * @param[out] npp32f_ptr_out, created by nppiMalloc_32f_C1(cols * rows, 3, &dummy);
 */
int blob_from_image(const Npp32f* npp32f_ptr_in, const int rows, const int cols, Npp32f* npp32f_ptr_out)
{
  assert(npp32f_ptr_in);
  assert(npp32f_ptr_out);
  Npp32f* const aDst[3] = { npp32f_ptr_out, npp32f_ptr_out + rows * cols, npp32f_ptr_out + rows * cols * 2 };

  auto status = nppiCopy_32f_C3P3R(npp32f_ptr_in, cols * 3 * sizeof(float), aDst, cols * sizeof(float),
                                   {.width = cols, .height = rows });
  VLOG(2) << "nppiCopy_32f_C3P3R status: " << status;
  assert(status == NPP_SUCCESS);
  return 0;
}
};  // namespace
