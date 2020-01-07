#ifndef __PLUGIN_LAYER_H__
#define __PLUGIN_LAYER_H__

#include <cassert>
#include <cstring>
#include <cudnn.h>
#include <iostream>
#include <memory>

#include "NvInferPlugin.h"

namespace DriveNet
{
#define NV_CUDA_CHECK(status)                                                                                          \
  {                                                                                                                    \
    if (status != 0)                                                                                                   \
    {                                                                                                                  \
      if (cudaGetErrorString(status) != (char*)("no error"))                                                           \
      {                                                                                                                \
        std::cout << "Cuda failure: status: " << status << std::endl;                                                  \
        std::cout << "Cuda failure: " << cudaGetErrorString(status) << " in file " << __FILE__ << " at line "          \
                  << __LINE__ << std::endl;                                                                            \
        abort();                                                                                                       \
      }                                                                                                                \
    }                                                                                                                  \
  }

// Forward declaration of cuda kernels
cudaError_t cudaYoloLayerV3(const void* input, void* output, const uint& batchSize, const uint& gridSize,
                            const uint& numOutputClasses, const uint& numBBoxes, uint64_t outputSize,
                            cudaStream_t stream);

class PluginFactory : public nvinfer1::IPluginFactory
{
public:
  PluginFactory();
  nvinfer1::IPlugin* createPlugin(const char* layerName, const void* serialData, size_t serialLength) override;
  bool isPlugin(const char* name);
  void destroy();

private:
  static const int m_MaxLeakyLayers = 72;
  static const int m_ReorgStride = 2;
  static constexpr float m_LeakyNegSlope = 0.1;
  static const int m_NumBoxes = 5;
  static const int m_NumCoords = 4;
  static const int m_NumClasses = 80;
  static const int m_MaxYoloLayers = 3;
  int m_LeakyReLUCount = 0;
  int m_YoloLayerCount = 0;
  nvinfer1::plugin::RegionParameters m_RegionParameters{ m_NumBoxes, m_NumCoords, m_NumClasses, nullptr };

  struct INvPluginDeleter
  {
    void operator()(nvinfer1::plugin::INvPlugin* ptr)
    {
      if (ptr)
      {
        ptr->destroy();
      }
    }
  };
  struct IPluginDeleter
  {
    void operator()(nvinfer1::IPlugin* ptr)
    {
      if (ptr)
      {
        ptr->terminate();
      }
    }
  };
  typedef std::unique_ptr<nvinfer1::plugin::INvPlugin, INvPluginDeleter> unique_ptr_INvPlugin;
  typedef std::unique_ptr<nvinfer1::IPlugin, IPluginDeleter> unique_ptr_IPlugin;

  unique_ptr_INvPlugin m_ReorgLayer;
  unique_ptr_INvPlugin m_RegionLayer;
  unique_ptr_INvPlugin m_LeakyReLULayers[m_MaxLeakyLayers];
  unique_ptr_IPlugin m_YoloLayers[m_MaxYoloLayers];
};

class YoloLayerV3 : public nvinfer1::IPlugin
{
public:
  YoloLayerV3(const void* data, size_t length);
  YoloLayerV3(const uint& numBoxes, const uint& numClasses, const uint& gridSize);
  int getNbOutputs() const override;
  nvinfer1::Dims getOutputDimensions(int index, const nvinfer1::Dims* inputs, int nbInputDims) override;
  void configure(const nvinfer1::Dims* inputDims, int nbInputs, const nvinfer1::Dims* outputDims, int nbOutputs,
                 int maxBatchSize) override;
  int initialize() override;
  void terminate() override;
  size_t getWorkspaceSize(int maxBatchSize) const override;
  int enqueue(int batchSize, const void* const* intputs, void** outputs, void* workspace, cudaStream_t stream) override;
  size_t getSerializationSize() override;
  void serialize(void* buffer) override;

private:
  template <typename T>
  void write(char*& buffer, const T& val)
  {
    *reinterpret_cast<T*>(buffer) = val;
    buffer += sizeof(T);
  }

  template <typename T>
  void read(const char*& buffer, T& val)
  {
    val = *reinterpret_cast<const T*>(buffer);
    buffer += sizeof(T);
  }
  uint m_NumBoxes;
  uint m_NumClasses;
  uint m_GridSize;
  uint64_t m_OutputSize;
};
}
#endif  // __PLUGIN_LAYER_H__