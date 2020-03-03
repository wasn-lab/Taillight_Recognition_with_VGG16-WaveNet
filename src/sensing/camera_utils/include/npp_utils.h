/*
   CREATER: ICL U300
   DATE: Aug, 2019
 */

#ifndef __NPP_UTILS_H__
#define __NPP_UTILS_H__

#include <opencv2/core/mat.hpp>
#include <nppdefs.h>

namespace npp_wrapper
{
int cvmat_to_npp8u_ptr(const cv::Mat& src, Npp8u* out_npp8u_ptr);
int npp8u_ptr_to_cvmat(const Npp8u* in_npp8u_ptr, const size_t in_num_bytes, cv::Mat& out_img, const int rows,
                       const int cols);
int npp8u_ptr_c4_to_c3(const Npp8u* npp8u_ptr_c4, const int rows, const int cols, Npp8u* npp8u_ptr_c3);
int blob_from_image(const Npp32f* npp32f_ptr_in, const int rows, const int cols, Npp32f* npp32f_ptr_out);
} // namespace npp_wrapper

#endif  // __NPP_UTILS_H__
