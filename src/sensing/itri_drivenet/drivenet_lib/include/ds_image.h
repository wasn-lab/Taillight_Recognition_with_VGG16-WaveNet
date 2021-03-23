#ifndef __IMAGE_H__
#define __IMAGE_H__

#include "trt_utils.h"
#include "npp_resizer_dn.h"
#include "camera_params.h"  // include camera topic name

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
  float* preprocessing(const cv::Mat& srcImg, const int& inputH, const int& inputW, int input_resize, int crop_size,
                       int crop_offset);

private:
  DriveNet_npp::NPPResizer* resizer;

  int dummy = 0;
  int BGROrder[3];
  Npp8u pixelArr[3];
  NppiSize nppSizeNet;
  NppiSize nppSizeResize;
  NppiSize nppSizeCrop;

  Npp8u* srcImg_npp8u_ptr;
  Npp8u* ResizeImg_npp8u_ptr;
  float* srcImg_32f_ptr;
  float* RGBImg_32f_ptr;
  float* CHWImg_32f_ptr;
  // Npp8u* dst;
  Npp8u* CropImg_npp8u_ptr;
  Npp8u* LetterBoxImg_npp8u_ptr;  // letterboxed Image given to the network as input

  cv::Mat Img8UC3;
  cv::Mat ImgFloat32C3;

  int m_InputWidthRaw;
  int m_InputHeightRaw;
  int m_InputWidthResize;
  int m_InputHeightResize;
  int m_Height;
  int m_Width;
  int m_Channel;
  int m_Size;
  int m_XOffset;
  int m_YOffset;
};
}  // namespace DriveNet
#endif
