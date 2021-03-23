/**
MIT License

Copyright (c) 2018 NVIDIA CORPORATION. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*
*/
#include "ds_image.h"
#include "npp_wrapper_dn.h"
#include <experimental/filesystem>
#include "npp.h"
#include <nppdefs.h>
#include "nppi_data_exchange_and_initialization.h"
#include "nppi_arithmetic_and_logical_operations.h"
#include "cuda_pros_dn.h"
#include <cassert>
#include <chrono>

using namespace DriveNet_npp;

namespace DriveNet
{
DsImage::DsImage() : m_Height(0), m_Width(0), m_XOffset(0), m_YOffset(0)
{
}
void DsImage::init(int inputW, int inputH)
{
  m_InputWidthRaw = camera::raw_image_width;
  m_InputHeightRaw = camera::raw_image_height;
  m_InputWidthResize = camera::image_height;
  m_InputHeightResize = camera::image_width;
  nppSizeNet.width = inputW;
  nppSizeNet.height = inputH;

  pixelArr[0] = 128;
  pixelArr[1] = 128;
  pixelArr[2] = 128;  // letterBox color pixel
  BGROrder[0] = 2;
  BGROrder[1] = 1;
  BGROrder[2] = 0;  // [original channel] = swapped channel

  Img8UC3 = cv::Mat::zeros(cvSize(nppSizeNet.width, nppSizeNet.height), CV_8UC3);
  ImgFloat32C3 = cv::Mat::zeros(cvSize(nppSizeNet.width, nppSizeNet.height), CV_32FC3);

  cudaMalloc((float**)&srcImg_32f_ptr, nppSizeNet.width * nppSizeNet.height * 3 * sizeof(float));
  cudaMalloc((float**)&RGBImg_32f_ptr, nppSizeNet.width * nppSizeNet.height * 3 * sizeof(float));

  srcImg_npp8u_ptr = nppiMalloc_8u_C3(m_InputWidthRaw, m_InputHeightRaw, &dummy);

  ResizeImg_npp8u_ptr = nppiMalloc_8u_C3(m_InputWidthResize, m_InputHeightResize, &dummy);
  LetterBoxImg_npp8u_ptr = nppiMalloc_8u_C3(nppSizeNet.width, nppSizeNet.height, &dummy);

  resizer = new DriveNet_npp::NPPResizer(m_InputHeightRaw, m_InputWidthRaw, m_InputHeightResize, m_InputWidthResize);
}
DsImage::~DsImage()
{
  nppiFree(srcImg_npp8u_ptr);
  nppiFree(ResizeImg_npp8u_ptr);
  nppiFree(LetterBoxImg_npp8u_ptr);

  cudaFree(srcImg_32f_ptr);
  cudaFree(RGBImg_32f_ptr);
  cudaFree(CHWImg_32f_ptr);
}
float* DsImage::preprocessing(const cv::Mat& srcImg, const int& inputH, const int& inputW)
{
  if (!srcImg.data || srcImg.cols <= 0 || srcImg.rows <= 0)
  {
    std::cout << "Unable to read image : " << std::endl;
    assert(0);
  }

  if (srcImg.channels() != 3)
  {
    std::cout << "Non RGB images are not supported" << std::endl;
    assert(0);
  }

  // srcImg.copyTo(m_MarkedImage);
  m_Height = srcImg.rows;
  m_Width = srcImg.cols;
  m_Channel = srcImg.channels();
  m_Size = m_Height * m_Width * m_Channel;

  if (m_Height != camera::raw_image_height || m_Width != camera::raw_image_width)
  {
    std::cout << "Input size is not equal to " << std::to_string(camera::raw_image_width) << "x"
              << std::to_string(camera::raw_image_height) << std::endl;
    assert(0);

    // resize the DsImage with scale
    float dim = std::max(m_Height, m_Width);
    int resizeH = ((m_Height / dim) * nppSizeNet.height);
    int resizeW = ((m_Width / dim) * nppSizeNet.width);

    // Additional checks for images with non even dims
    if ((nppSizeNet.width - resizeW) % 2)
    {
      resizeW--;
    }
    if ((nppSizeNet.height - resizeH) % 2)
    {
      resizeH--;
    }
    assert((nppSizeNet.width - resizeW) % 2 == 0);
    assert((nppSizeNet.height - resizeH) % 2 == 0);

    nppSizeResize.width = resizeW;
    nppSizeResize.height = resizeH;

    m_XOffset = (nppSizeNet.width - resizeW) / 2;
    m_YOffset = (nppSizeNet.height - resizeH) / 2;
    assert(2 * m_YOffset + resizeH == nppSizeNet.height);
    assert(2 * m_XOffset + resizeW == inputW);

    // std::cout << "m_Width: " << m_Width << ", m_Height: " << m_Height <<std::endl;
    // std::cout << "resizeW: " << resizeW << ", resizeH: " << resizeH <<std::endl;
    // std::cout << "nppSizeNet.width: " << nppSizeNet.width << ", nppSizeNet.height: " << nppSizeNet.height
    // <<std::endl;

    assert(srcImg_npp8u_ptr);
    assert(LetterBoxImg_npp8u_ptr);
    assert(ResizeImg_npp8u_ptr);
    cudaMalloc((float**)&CHWImg_32f_ptr, nppSizeNet.width * nppSizeNet.height * 3 * sizeof(float));
    assert(srcImg_32f_ptr);

    // cv::Mat to Npp8u
    cvmat_to_npp8u_ptr(srcImg, srcImg_npp8u_ptr);
    // resizing
    resizer->resize(srcImg_npp8u_ptr, ResizeImg_npp8u_ptr);
    // letterboxing
    nppiCopyConstBorder_8u_C3R(ResizeImg_npp8u_ptr, resizeW * 3, nppSizeResize, LetterBoxImg_npp8u_ptr, Img8UC3.step,
                               nppSizeNet, m_YOffset, 0, pixelArr);

    // unsigned int to float
    nppiConvert_8u32f_C3R(LetterBoxImg_npp8u_ptr, Img8UC3.step, srcImg_32f_ptr, ImgFloat32C3.step, nppSizeNet);
    // BRG to RGB
    nppiSwapChannels_32f_C3R(srcImg_32f_ptr, ImgFloat32C3.step, RGBImg_32f_ptr, ImgFloat32C3.step, nppSizeNet,
                             BGROrder);
    // // HWC to CHW
    cudaReshape(CHWImg_32f_ptr, RGBImg_32f_ptr, nppSizeNet.width * nppSizeNet.height);

    // Cuda mem to cv:Mat
    // dst = nppiMalloc_8u_C3(nppSizeNet.width, nppSizeNet.height, &dummy);
    // nppiConvert_32f8u_C3R(CHWImg_32f_ptr, ImgFloat32C3.step, dst, Img8UC3.step, nppSizeNet, NPP_RND_NEAR);
    // cv::Mat out_img;
    // npp8u_ptr_to_cvmat(dst, nppSizeNet.height*nppSizeNet.width * 3, out_img, nppSizeNet.height, nppSizeNet.width);
    // cv::imwrite("npp8u_c3.jpg", out_img);
    // nppiFree(dst);
  }
  return CHWImg_32f_ptr;
}
float* DsImage::preprocessing(const cv::Mat& srcImg, const int& inputH, const int& inputW, int input_resize)
{
  if (!srcImg.data)
  {
    std::cout << "Unable to read image : " << std::endl;
    assert(0);
  }
  else if (srcImg.cols <= 0 || srcImg.rows <= 0)
  {
    std::cout << "image size - cols: " << srcImg.cols << ", rows: " << srcImg.rows << std::endl;
    assert(0);
  }

  if (srcImg.channels() != 3)
  {
    std::cout << "Non RGB images are not supported" << std::endl;
    assert(0);
  }

  m_Height = srcImg.rows;
  m_Width = srcImg.cols;

  assert(srcImg_npp8u_ptr);
  cudaMalloc((float**)&CHWImg_32f_ptr, nppSizeNet.width * nppSizeNet.height * 3 * sizeof(float));
  assert(srcImg_32f_ptr);
  assert(LetterBoxImg_npp8u_ptr);
  assert(ResizeImg_npp8u_ptr);

  if (m_Height != camera::image_height || m_Width != camera::image_width)
  {
    std::cout << "Input size is not equal to " << std::to_string(camera::image_width) << "x"
              << std::to_string(camera::image_height) << std::endl;
    assert(0);
  }
  else
  {
    nppSizeResize.width = m_Width;
    nppSizeResize.height = m_Height;
    m_XOffset = (nppSizeNet.width - m_Width) / 2;
    m_YOffset = (nppSizeNet.height - m_Height) / 2;

    // cv::Mat to Npp8u
    cvmat_to_npp8u_ptr(srcImg, ResizeImg_npp8u_ptr);
    // letterboxing
    nppiCopyConstBorder_8u_C3R(ResizeImg_npp8u_ptr, nppSizeResize.width * 3, nppSizeResize, LetterBoxImg_npp8u_ptr,
                               Img8UC3.step, nppSizeNet, m_YOffset, 0, pixelArr);

    // unsigned int to float
    nppiConvert_8u32f_C3R(LetterBoxImg_npp8u_ptr, Img8UC3.step, srcImg_32f_ptr, ImgFloat32C3.step, nppSizeNet);
    // // BRG to RGB
    nppiSwapChannels_32f_C3R(srcImg_32f_ptr, ImgFloat32C3.step, RGBImg_32f_ptr, ImgFloat32C3.step, nppSizeNet,
                             BGROrder);
    // // // HWC to CHW
    cudaReshape(CHWImg_32f_ptr, RGBImg_32f_ptr, nppSizeNet.width * nppSizeNet.height);

    // Cuda mem to cv:Mat
    // dst = nppiMalloc_8u_C3(nppSizeNet.width, nppSizeNet.height, &dummy);
    // nppiConvert_32f8u_C3R(CHWImg_32f_ptr, ImgFloat32C3.step, dst, Img8UC3.step, nppSizeNet, NPP_RND_NEAR);
    // cv::Mat out_img;
    // npp8u_ptr_to_cvmat(dst, nppSizeNet.height*nppSizeNet.width * 3, out_img, nppSizeNet.height, nppSizeNet.width);
    // if (!out_img.data)
    // {
    //     std::cout << "Unable to read image : " << std::endl;
    //     assert(0);
    // }
    // cv::imwrite("npp8u_c3.jpg", out_img);
    // cv::imshow("npp8u_c3.jpg", out_img);
    // cv::waitKey(1);
    // nppiFree(dst);
  }
  return CHWImg_32f_ptr;
}
float* DsImage::preprocessing(const cv::Mat& srcImg, const int& inputH, const int& inputW, int input_resize,
                              int crop_size, int crop_offset)
{
  if (!srcImg.data)
  {
    std::cout << "Unable to read image : " << std::endl;
    assert(0);
  }
  else if (srcImg.cols <= 0 || srcImg.rows <= 0)
  {
    std::cout << "image size - cols: " << srcImg.cols << ", rows: " << srcImg.rows << std::endl;
    assert(0);
  }

  if (srcImg.channels() != 3)
  {
    std::cout << "Non RGB images are not supported" << std::endl;
    assert(0);
  }

  m_Height = srcImg.rows;
  m_Width = srcImg.cols;

  assert(srcImg_npp8u_ptr);
  cudaMalloc((float**)&CHWImg_32f_ptr, nppSizeNet.width * nppSizeNet.height * 3 * sizeof(float));
  assert(srcImg_32f_ptr);
  assert(LetterBoxImg_npp8u_ptr);
  assert(ResizeImg_npp8u_ptr);

  if (m_Height != camera::image_height || m_Width != camera::image_width)
  {
    std::cout << "Input size is not equal to " << std::to_string(camera::image_width) << "x"
              << std::to_string(camera::image_height) << std::endl;
    assert(0);
  }
  else
  {
    nppSizeResize.width = m_Width;
    nppSizeResize.height = m_Height;
    m_XOffset = (nppSizeNet.width - m_Width) / 2;
    m_YOffset = (nppSizeNet.height - m_Height) / 2;

    nppSizeCrop.width = m_Width - crop_size;
    nppSizeCrop.height = m_Height;
    CropImg_npp8u_ptr = nppiMalloc_8u_C3(nppSizeCrop.width, nppSizeCrop.height, &dummy);

    // cv::Mat to Npp8u
    cvmat_to_npp8u_ptr(srcImg, ResizeImg_npp8u_ptr);
    // crop image
    nppiCopyConstBorder_8u_C3R(ResizeImg_npp8u_ptr, nppSizeResize.width * 3, nppSizeResize, CropImg_npp8u_ptr,
                               nppSizeCrop.width * 3, nppSizeCrop, 0, (-1) * crop_offset, pixelArr);
    // letterboxing
    nppiCopyConstBorder_8u_C3R(CropImg_npp8u_ptr, nppSizeCrop.width * 3, nppSizeCrop, LetterBoxImg_npp8u_ptr,
                               Img8UC3.step, nppSizeNet, m_YOffset, 0 + crop_offset, pixelArr);

    // unsigned int to float
    nppiConvert_8u32f_C3R(LetterBoxImg_npp8u_ptr, Img8UC3.step, srcImg_32f_ptr, ImgFloat32C3.step, nppSizeNet);
    // // BRG to RGB
    nppiSwapChannels_32f_C3R(srcImg_32f_ptr, ImgFloat32C3.step, RGBImg_32f_ptr, ImgFloat32C3.step, nppSizeNet,
                             BGROrder);
    // // // HWC to CHW
    cudaReshape(CHWImg_32f_ptr, RGBImg_32f_ptr, nppSizeNet.width * nppSizeNet.height);

    // Cuda mem to cv:Mat
    // dst = nppiMalloc_8u_C3(nppSizeNet.width, nppSizeNet.height, &dummy);
    // nppiConvert_32f8u_C3R(CHWImg_32f_ptr, ImgFloat32C3.step, dst, Img8UC3.step, nppSizeNet, NPP_RND_NEAR);
    // cv::Mat out_img;
    // npp8u_ptr_to_cvmat(dst, nppSizeNet.height*nppSizeNet.width * 3, out_img, nppSizeNet.height, nppSizeNet.width);
    // if (!out_img.data)
    // {
    //     std::cout << "Unable to read image : " << std::endl;
    //     assert(0);
    // }
    // cv::imwrite("npp8u_c3" +  std::to_string(crop_offset) +".jpg", out_img);
    // cv::imshow("npp8u_c3.jpg", out_img);
    // cv::waitKey(1);
    // nppiFree(dst);
  }
  return CHWImg_32f_ptr;
}
}  // namespace DriveNet
