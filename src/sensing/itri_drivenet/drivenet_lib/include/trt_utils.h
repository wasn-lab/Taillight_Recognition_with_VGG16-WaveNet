#ifndef __TRT_UTILS_H__
#define __TRT_UTILS_H__

/* OpenCV headers */
#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <set>

#include "NvInfer.h"

#include "ds_image.h"
#include "plugin_factory.h"

namespace DriveNet
{
class DsImage;
struct BBox
{
  float x1, y1, x2, y2;
};

struct BBoxInfo
{
  BBox box;
  int label;
  int classId;  // For coco benchmarking
  float prob;
};

enum class yolo_class_id
{
  begin = 0,
  person = begin,  // 0
  bicycle,         // 1
  car,             // 2
  motorbike,       // 3
  others1,         // 4
  bus,             // 5
  others2,         // 6
  truck            // 7
};

class Logger : public nvinfer1::ILogger
{
public:
  std::string _msgs;

  void log(nvinfer1::ILogger::Severity severity, const char* msg) override
  {
    // suppress info-level messages
    if (severity == Severity::kINFO)
    {
      return;
    }

    switch (severity)
    {
      case Severity::kINTERNAL_ERROR:
        std::cerr << "INTERNAL_ERROR: ";
        break;
      case Severity::kERROR:
        std::cerr << "ERROR: ";
        break;
      case Severity::kWARNING:
        std::cerr << "WARNING: ";
        break;
      case Severity::kINFO:
        std::cerr << "INFO: ";
        break;
      default:
        std::cerr << "UNKNOWN: ";
        break;
    }
    std::cerr << msg << std::endl;
    _msgs = msg;
  }
};

class YoloTinyMaxpoolPaddingFormula : public nvinfer1::IOutputDimensionsFormula
{
private:
  std::set<std::string> m_SamePaddingLayers;

  nvinfer1::DimsHW compute(nvinfer1::DimsHW inputDims, nvinfer1::DimsHW kernelSize, nvinfer1::DimsHW stride,
                           nvinfer1::DimsHW padding, nvinfer1::DimsHW dilation, const char* layerName) const override
  {
    assert(inputDims.d[0] == inputDims.d[1]);
    assert(kernelSize.d[0] == kernelSize.d[1]);
    assert(stride.d[0] == stride.d[1]);
    assert(padding.d[0] == padding.d[1]);

    int outputDim;
    // Only layer maxpool_12 makes use of same padding
    if (m_SamePaddingLayers.find(layerName) != m_SamePaddingLayers.end())
    {
      outputDim = (inputDims.d[0] + 2 * padding.d[0]) / stride.d[0];
    }
    // Valid Padding
    else
    {
      outputDim = (inputDims.d[0] - kernelSize.d[0]) / stride.d[0] + 1;
    }
    return nvinfer1::DimsHW{ outputDim, outputDim };
  }

public:
  void addSamePaddingLayer(std::string input)
  {
    m_SamePaddingLayers.insert(input);
  }
};

// Common helper functions
cv::Mat blobFromDsImages(const std::vector<cv::Mat>& inputImages, const int& inputH, const int& inputW);
std::string trim(std::string s);
float clamp(const float val, const float minVal, const float maxVal);
bool fileExists(const std::string fileName, bool verbose = true);
bool removeFile(const std::string fileName, bool verbose = true);
BBox convertBBoxNetRes(const float& bx, const float& by, const float& bw, const float& bh, const uint& stride,
                       const uint& netW, const uint& netH);
void convertBBoxImgRes(const float scalingFactor, const float& xOffset, const float& yOffset, BBox& bbox);
void printPredictions(const BBoxInfo& info, const std::string& className);
std::vector<std::string> loadListFromTextFile(const std::string filename);
std::vector<std::string> loadImageList(const std::string filename, const std::string prefix);
std::vector<BBoxInfo> nmsAllClasses(const float nmsThresh, std::vector<BBoxInfo>& binfo, const uint numClasses);
std::vector<BBoxInfo> nonMaximumSuppression(const float nmsThresh, std::vector<BBoxInfo> binfo);
nvinfer1::ICudaEngine* loadTRTEngine(const std::string planFilePath, PluginFactory* pluginFactory, Logger& logger);
std::vector<float> loadWeights(const std::string weightsFilePath, const std::string& networkType);
std::string dimsToString(const nvinfer1::Dims d);
void displayDimType(const nvinfer1::Dims d);
int getNumChannels(nvinfer1::ITensor* t);
uint64_t get3DTensorVolume(nvinfer1::Dims inputDims);

// Helper functions to create yolo engine
nvinfer1::ILayer* netAddMaxpool(int layerIdx, std::map<std::string, std::string>& block, nvinfer1::ITensor* input,
                                nvinfer1::INetworkDefinition* network);
nvinfer1::ILayer* netAddConvLinear(int layerIdx, std::map<std::string, std::string>& block, std::vector<float>& weights,
                                   std::vector<nvinfer1::Weights>& trtWeights, int& weightPtr, int& inputChannels,
                                   nvinfer1::ITensor* input, nvinfer1::INetworkDefinition* network);
nvinfer1::ILayer* netAddConvBNLeaky(int layerIdx, std::map<std::string, std::string>& block,
                                    std::vector<float>& weights, std::vector<nvinfer1::Weights>& trtWeights,
                                    int& weightPtr, int& inputChannels, nvinfer1::ITensor* input,
                                    nvinfer1::INetworkDefinition* network);
nvinfer1::ILayer* netAddUpsample(int layerIdx, std::map<std::string, std::string>& block, std::vector<float>& weights,
                                 std::vector<nvinfer1::Weights>& trtWeights, int& inputChannels,
                                 nvinfer1::ITensor* input, nvinfer1::INetworkDefinition* network);
void printLayerInfo(std::string layerIndex, std::string layerName, std::string layerInput, std::string layerOutput,
                    std::string weightPtr);
}  // namespace DriveNet
#endif
