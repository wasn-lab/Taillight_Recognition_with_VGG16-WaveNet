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
#include <assert.h>
#include <chrono>

using namespace DriveNet_npp;

namespace DriveNet{
DsImage::DsImage() :
    m_Height(0),
    m_Width(0),
    m_XOffset(0),
    m_YOffset(0)
{
}
void DsImage::init(int inputW, int inputH)
{
    m_InputWidth1 = 1920; m_InputHeight1 = 1208;  
    m_InputWidth2 = 1208; m_InputHeight2 = 1920; 
    float col_scale_factor_1 = 0.3167; float row_scale_factor_1 = 0.318; 
    float col_scale_factor_2 = 0.318; float row_scale_factor_2 = 0.3167; 
    nppSizeNet.width = inputW; nppSizeNet.height = inputH;

    pixelArr[0] = 128; pixelArr[1] = 128;  pixelArr[2] = 128; // letterBox color pixel
    BGROrder[0] = 2; BGROrder[1] = 1; BGROrder[2] = 0;   // [original channel] = swapped channel

    Img8UC3 = cv::Mat::zeros(cvSize(nppSizeNet.width, nppSizeNet.height), CV_8UC3);
    ImgFloat32C3 = cv::Mat::zeros(cvSize(nppSizeNet.width, nppSizeNet.height), CV_32FC3);

    cudaMalloc((float**)&srcImg_32f_ptr, nppSizeNet.width* nppSizeNet.height* 3* sizeof(float));
    cudaMalloc((float**)&RGBImg_32f_ptr, nppSizeNet.width* nppSizeNet.height* 3* sizeof(float)); 

    srcImg_npp8u_ptr = nppiMalloc_8u_C3(nppSizeNet.width, nppSizeNet.width, &dummy);
    srcImg_npp8u_ptr1 = nppiMalloc_8u_C3(m_InputWidth1, m_InputHeight1, &dummy);
    srcImg_npp8u_ptr2 = nppiMalloc_8u_C3(m_InputWidth2, m_InputHeight2, &dummy);
    ResizeImg_npp8u_ptr1 = nppiMalloc_8u_C3(int(m_InputWidth1*col_scale_factor_1), int(m_InputHeight1*row_scale_factor_1), &dummy);
    ResizeImg_npp8u_ptr2 = nppiMalloc_8u_C3(int(m_InputWidth2*col_scale_factor_2), int(m_InputHeight2*row_scale_factor_2), &dummy);   
    LetterBoxImg_npp8u_ptr = nppiMalloc_8u_C3(nppSizeNet.width, nppSizeNet.height, &dummy);
    
    resizer = new DriveNet_npp::NPPResizer(m_InputHeight1, m_InputWidth1, row_scale_factor_1, col_scale_factor_1);
    resizer2 = new DriveNet_npp::NPPResizer(m_InputHeight2, m_InputWidth2, row_scale_factor_2, col_scale_factor_2);
}
DsImage::~DsImage()
{
    nppiFree(srcImg_npp8u_ptr);
    nppiFree(srcImg_npp8u_ptr1);
    nppiFree(srcImg_npp8u_ptr2);
    nppiFree(ResizeImg_npp8u_ptr1);
    nppiFree(ResizeImg_npp8u_ptr2);
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
    m_Size = m_Height*m_Width*m_Channel;

    // resize the DsImage with scale
    float dim = std::max(m_Height, m_Width);
    int resizeH = ((m_Height / dim) * nppSizeNet.height);
    int resizeW = ((m_Width / dim) * nppSizeNet.width);

    // Additional checks for images with non even dims
    if ((nppSizeNet.width - resizeW) % 2) resizeW--;
    if ((nppSizeNet.height - resizeH) % 2) resizeH--;
    assert((nppSizeNet.width - resizeW) % 2 == 0);
    assert((nppSizeNet.height - resizeH) % 2 == 0);

    nppSizeResize.width = resizeW; nppSizeResize.height = resizeH;

    m_XOffset = (nppSizeNet.width - resizeW) / 2;
    m_YOffset = (nppSizeNet.height - resizeH) / 2;
    assert(2 * m_YOffset + resizeH == nppSizeNet.height);
    assert(2 * m_XOffset + resizeW == inputW);

    // std::cout << "m_Width: " << m_Width << ", m_Height: " << m_Height <<std::endl;
    // std::cout << "resizeW: " << resizeW << ", resizeH: " << resizeH <<std::endl;
    // std::cout << "nppSizeNet.width: " << nppSizeNet.width << ", nppSizeNet.height: " << nppSizeNet.height <<std::endl;

    assert(srcImg_npp8u_ptr1);
    assert(srcImg_npp8u_ptr2);
    assert(LetterBoxImg_npp8u_ptr);   
    assert(ResizeImg_npp8u_ptr1);   
    assert(ResizeImg_npp8u_ptr2);
    cudaMalloc((float**)&CHWImg_32f_ptr, nppSizeNet.width* nppSizeNet.height* 3* sizeof(float));
    assert(srcImg_32f_ptr);

    if(m_Height == 1208 && m_Width == 1920)
    {
        // cv::Mat to Npp8u
        cvmat_to_npp8u_ptr(srcImg, srcImg_npp8u_ptr1);
        // resizing 
        resizer->resize(srcImg_npp8u_ptr1, ResizeImg_npp8u_ptr1);
        // letterboxing
        nppiCopyConstBorder_8u_C3R(ResizeImg_npp8u_ptr1, resizeW*3, nppSizeResize, LetterBoxImg_npp8u_ptr, Img8UC3.step, nppSizeNet, m_YOffset, 0, pixelArr);
    }
    else if(m_Height == 1920 && m_Width == 1208)
    {
        // cv::Mat to Npp8u
        cvmat_to_npp8u_ptr(srcImg, srcImg_npp8u_ptr2);
        // resizing 
        resizer2->resize(srcImg_npp8u_ptr2, ResizeImg_npp8u_ptr2);
        // letterboxing
        nppiCopyConstBorder_8u_C3R(ResizeImg_npp8u_ptr2, resizeW*3, nppSizeResize, LetterBoxImg_npp8u_ptr, Img8UC3.step, nppSizeNet, 0, m_XOffset, pixelArr);    
    }
    // unsigned int to float
    assert(srcImg_32f_ptr);
    nppiConvert_8u32f_C3R(LetterBoxImg_npp8u_ptr, Img8UC3.step, srcImg_32f_ptr, ImgFloat32C3.step, nppSizeNet);  
    // BRG to RGB
    nppiSwapChannels_32f_C3R(srcImg_32f_ptr, ImgFloat32C3.step, RGBImg_32f_ptr, ImgFloat32C3.step, nppSizeNet, BGROrder);
    // // HWC to CHW
    cudaReshape(CHWImg_32f_ptr, RGBImg_32f_ptr, nppSizeNet.width * nppSizeNet.height);  

    // Cuda mem to cv:Mat 
    // dst = nppiMalloc_8u_C3(nppSizeNet.width, nppSizeNet.height, &dummy);
    // nppiConvert_32f8u_C3R(CHWImg_32f_ptr, ImgFloat32C3.step, dst, Img8UC3.step, nppSizeNet, NPP_RND_NEAR);
    // cv::Mat out_img;
    // npp8u_ptr_to_cvmat(dst, nppSizeNet.height*nppSizeNet.width * 3, out_img, nppSizeNet.height, nppSizeNet.width);
    // cv::imwrite("npp8u_c3.jpg", out_img);   
    // nppiFree(dst);

    return CHWImg_32f_ptr;
}
float* DsImage::preprocessing(const cv::Mat& srcImg, const int& inputH, const int& inputW, int input_resize)
{
    if (!srcImg.data)
    {
        std::cout << "Unable to read image : " << std::endl;
        assert(0);
    }
    else if (srcImg.cols <= 0 || srcImg.rows <= 0){
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
    cudaMalloc((float**)&CHWImg_32f_ptr, nppSizeNet.width* nppSizeNet.height* 3* sizeof(float));
    assert(srcImg_32f_ptr);
    assert(LetterBoxImg_npp8u_ptr);   
    assert(ResizeImg_npp8u_ptr1);   
    assert(ResizeImg_npp8u_ptr2);

    if((m_Height*m_Width != 608*384))
    {
        std::cout << "Input size is not equal to 608x384" << std::endl;
        assert(0);      
    }
    nppSizeResize.width = m_Width; nppSizeResize.height = m_Height;
    m_XOffset = (nppSizeNet.width - m_Width) / 2;
    m_YOffset = (nppSizeNet.height - m_Height) / 2;

    if(m_Height == 384 && m_Width == 608)
    {
        // cv::Mat to Npp8u
        cvmat_to_npp8u_ptr(srcImg, ResizeImg_npp8u_ptr1);
        // letterboxing
        nppiCopyConstBorder_8u_C3R(ResizeImg_npp8u_ptr1, m_Width*3, nppSizeResize, LetterBoxImg_npp8u_ptr, Img8UC3.step, nppSizeNet, m_YOffset, 0, pixelArr);
    }
    else if(m_Height == 608 && m_Width == 384)
    {
        // cv::Mat to Npp8u
        cvmat_to_npp8u_ptr(srcImg, ResizeImg_npp8u_ptr2);
        // letterboxing
        nppiCopyConstBorder_8u_C3R(ResizeImg_npp8u_ptr2, m_Width*3, nppSizeResize, LetterBoxImg_npp8u_ptr, Img8UC3.step, nppSizeNet, 0, m_XOffset, pixelArr);    
    }
    // unsigned int to float
    nppiConvert_8u32f_C3R(LetterBoxImg_npp8u_ptr, Img8UC3.step, srcImg_32f_ptr, ImgFloat32C3.step, nppSizeNet);  
    // // BRG to RGB
    nppiSwapChannels_32f_C3R(srcImg_32f_ptr, ImgFloat32C3.step, RGBImg_32f_ptr, ImgFloat32C3.step, nppSizeNet, BGROrder);
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

    return CHWImg_32f_ptr;
}
}
