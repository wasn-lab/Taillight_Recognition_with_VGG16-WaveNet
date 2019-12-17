#ifndef __IMAGE_H__
#define __IMAGE_H__

#include "trt_utils.h"
#include "npp_resizer_dn.h"

namespace DriveNet
{
struct BBoxInfo;

class DsImage
{
public:
  DsImage();
  ~DsImage();
  void init(int inputW, int inputH);
  float* preprocessing(const cv::Mat& srcImg, const int& inputH, const int& inputW);
  float* preprocessing(const cv::Mat& srcImg, const int& inputH, const int& inputW, int input_resize);

private:
  DriveNet_npp::NPPResizer* resizer;
  DriveNet_npp::NPPResizer* resizer2;

  int dummy = 0;
  int BGROrder[3];
  float* dstCudaCHW;
  Npp8u pixelArr[3];
  NppiSize nppSizeNet;
  NppiSize nppSizeResize;

  Npp8u* srcImg_npp8u_ptr;
  Npp8u* srcImg_npp8u_ptr1;
  Npp8u* srcImg_npp8u_ptr2;
  Npp8u* ResizeImg_npp8u_ptr1;
  Npp8u* ResizeImg_npp8u_ptr2;
  float* srcImg_32f_ptr;
  float* RGBImg_32f_ptr;
  float* CHWImg_32f_ptr;
  // Npp8u* dst;
  Npp8u* LetterBoxImg_npp8u_ptr;  // letterboxed Image given to the network as input

  cv::Mat Img8UC3;
  cv::Mat ImgFloat32C3;

  int m_InputWidth1;
  int m_InputHeight1;
  int m_InputWidth2;
  int m_InputHeight2;
  int m_Height;
  int m_Width;
  int m_Channel;
  int m_Size;
  int m_XOffset;
  int m_YOffset;
};
}
#endif
