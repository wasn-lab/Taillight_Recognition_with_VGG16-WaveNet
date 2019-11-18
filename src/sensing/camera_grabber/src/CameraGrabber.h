#ifndef __CAMERAGRABBER__
#define __CAMERAGRABBER__

#include <cstdio>
#include <thread>
#include <queue>
#include <boost/thread/shared_mutex.hpp>
#include <opencv2/opencv.hpp>

#include "ros/ros.h"
#include "ros/package.h"

#include <std_msgs/String.h>

#include "Util/ProgramArguments.hpp"
#include "Util/Checks.hpp"
#include "Util/HelperFunc.h"
#include "CameraGrabber/MultiGMSLCameraGrabber.h"
#include "RosImagePubSub.hpp"

#include "npp_utils.h"
#include "npp.h"
#include "npp_resizer.h"
#include "npp_remapper.h"
#include "nppdefs.h"
#include "camera_utils.h"
#include "camera_params.h"

#define CAMERA_NUM 12

namespace SensingSubSystem
{
class GPUFrame
{
public:
  // for camera grabber CUDA memory
  std::vector<void*> frames_GPU;

  // Number of camera frames
  size_t size;

  // cuda stream
  cudaStream_t st;

  GPUFrame() : frames_GPU(CAMERA_NUM), size(CAMERA_NUM)
  {
    init(CAMERA_NUM);
  }

  GPUFrame(size_t camera_num) : frames_GPU(camera_num), size(camera_num)
  {
    init(camera_num);
  }

  ~GPUFrame()
  {
    for (size_t i = 0; i < size; ++i)
      if (frames_GPU[i] != nullptr)
      {
        cudaFree(frames_GPU[i]);
      }
    cudaStreamDestroy(st);
  }

  bool initialize()
  {
    for (size_t i = 0; i < size; ++i)
      CHECK_CUDA_ERROR(cudaMalloc(&frames_GPU[i], MultiGMSLCameraGrabber::ImageSize));
    return true;
  }

private:
  void init(size_t s)
  {
    for (size_t i = 0; i < s; ++i)
      frames_GPU[i] = nullptr;
    cudaStreamCreate(&st);
  }
};

class BufferConfig
{
public:
  GPUFrame* cams_ptr;

  BufferConfig() : cams_ptr(nullptr), grabber_(nullptr)
  {
  }

  void setGrabber(MultiGMSLCameraGrabber* grabber)
  {
    grabber_ = grabber;
  }

  void initBuffer()
  {
    cams[0].initialize();
    cams[1].initialize();
    cams_ptr = &cams[0];
  }

private:
  std::mutex mut_;

  MultiGMSLCameraGrabber* grabber_;
  GPUFrame cams[2];
};

// for display detection results
class DisplayConfig
{
public:
  std::vector<cv::Mat> canvas;
  std::vector<uint8_t*> camera_buffer_cpu;
  BufferConfig* camera_buffer_ptr;

  DisplayConfig(BufferConfig* camera_buffer_ptr_in = nullptr)
    : canvas(CAMERA_NUM), camera_buffer_cpu(CAMERA_NUM), camera_buffer_ptr(camera_buffer_ptr_in)
  {
    for (size_t i = 0; i < CAMERA_NUM; ++i)
    {
      camera_buffer_cpu.at(i) = new uint8_t[RAWSIZE_];
    }
    cudaStreamCreate(&st);
  }

  ~DisplayConfig()
  {
    for (size_t i = 0; i < CAMERA_NUM; ++i)
    {
      delete[] camera_buffer_cpu.at(i);
    }
    cudaStreamDestroy(st);
  }

  void prepareMatCanvas()
  {
    if (camera_buffer_ptr != nullptr)
    {
      for (size_t i = 0; i < CAMERA_NUM; ++i)
      {
        cudaMemcpyAsync(camera_buffer_cpu[i], camera_buffer_ptr->cams_ptr->frames_GPU[i], RAWSIZE_ * sizeof(uint8_t),
                        cudaMemcpyDeviceToHost, st);
      }
    }
    cudaStreamSynchronize(st);
    for (size_t i = 0; i < CAMERA_NUM; ++i)
    {
      canvas[i] = cv::Mat(cam_H, cam_W, CV_8UC4, camera_buffer_cpu[i]);
      cv::cvtColor(canvas[i], canvas[i], CV_RGBA2BGR, cam_C);
    }
  }

private:
  const size_t RAWSIZE_ = MultiGMSLCameraGrabber::ImageSize / sizeof(uint8_t);
  const size_t cam_H = MultiGMSLCameraGrabber::H;
  const size_t cam_W = MultiGMSLCameraGrabber::W;
  const size_t cam_C = 3;  // BGR 3 channels
  cudaStream_t st;
};
}  // end namespace SensingSubSystem

#endif
