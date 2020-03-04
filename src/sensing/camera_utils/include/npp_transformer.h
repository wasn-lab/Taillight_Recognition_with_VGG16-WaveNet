/*
   CREATER: ICL U300
   DATE: Mar, 2020
 */

#ifndef __CANONICAL_NPP_TRANSFORMER_H__
#define __CANONICAL_NPP_TRANSFORMER_H__

#include <memory>
#include <opencv2/core/mat.hpp>
#include <nppdefs.h>

class NPPTransformer
{
private:
  const int src_rows_, src_cols_;
  const int dst_rows_, dst_cols_;
  const int num_src_bytes_, num_dst_bytes_;

public:
  NPPTransformer(const int src_rows, const int src_cols, const int dst_rows, const int dst_cols);
  ~NPPTransformer();
  void transform(const Npp8u* npp8u_ptr_in, cv::Mat& dst);

};

#endif  // __CANONICAL_NPP_TRANSFORMER_H__
