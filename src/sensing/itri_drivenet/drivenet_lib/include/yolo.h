#ifndef _YOLO_H_
#define _YOLO_H_

#include "calibrator.h"
#include "plugin_factory.h"
#include "trt_utils.h"

#include "NvInfer.h"

#include <cstdint>
#include <string>
#include <vector>

namespace DriveNet
{
/**
 * Holds all the file paths required to build a network.
 */
struct NetworkInfo
{
  std::string networkType;
  std::string configFilePath;
  std::string wtsFilePath;
  std::string labelsFilePath;
  std::string precision;
  std::string calibrationTablePath;
  std::string enginePath;
  std::string inputBlobName;
};

/**
 * Holds information about runtime inference params.
 */
struct InferParams
{
  bool printPerfInfo;
  bool printPredictionInfo;
  std::string calibImages;
  std::string calibImagesPath;
  float probThresh;
  float nmsThresh;
};

/**
 * Holds information about an output tensor of the yolo network.
 */
struct TensorInfo
{
  std::string blobName;
  uint stride{ 0 };
  uint gridSize{ 0 };
  uint numClasses{ 0 };
  uint numBBoxes{ 0 };
  uint64_t volume{ 0 };
  std::vector<uint> masks;
  std::vector<float> anchors;
  int bindingIndex{ -1 };
  float* hostBuffer{ nullptr };
};

class Yolo
{
public:
  cudaStream_t m_CudaStream;
  std::string getNetworkType() const
  {
    return m_NetworkType;
  }
  float getNMSThresh() const
  {
    return m_NMSThresh;
  }
  std::string getClassName(const int& label) const
  {
    return m_ClassNames.at(label);
  }
  int getClassId(const int& label) const
  {
    return m_ClassIds.at(label);
  }
  uint getInputH() const
  {
    return m_InputH;
  }
  uint getInputW() const
  {
    return m_InputW;
  }
  uint getNumClasses() const
  {
    return m_ClassNames.size();
  }
  bool isPrintPredictions() const
  {
    return m_PrintPredictions;
  }
  bool isPrintPerfInfo() const
  {
    return m_PrintPerfInfo;
  }
  void doInference(const unsigned char* input, const uint batchSize);
  void doInference(void* input, const uint batchSize);
  std::vector<BBoxInfo> decodeDetections(const int& imageIdx, const int& imageH, const int& imageW);

  virtual ~Yolo();

protected:
  Yolo(const uint batchSize, const NetworkInfo& networkInfo, const InferParams& inferParams);
  std::string m_EnginePath;
  const std::string m_NetworkType;
  const std::string m_ConfigFilePath;
  const std::string m_WtsFilePath;
  const std::string m_LabelsFilePath;
  const std::string m_Precision;
  const std::string m_CalibImages;
  const std::string m_CalibImagesFilePath;
  std::string m_CalibTableFilePath;
  const std::string m_InputBlobName;
  std::vector<TensorInfo> m_OutputTensors;
  std::vector<std::map<std::string, std::string>> m_configBlocks;
  uint m_InputH;
  uint m_InputW;
  uint m_InputC;
  uint64_t m_InputSize;
  const float m_ProbThresh;
  const float m_NMSThresh;
  std::vector<std::string> m_ClassNames;
  // Class ids for coco benchmarking
  const std::vector<int> m_ClassIds{ 1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                                     22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
                                     46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65,
                                     67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90 };
  const bool m_PrintPerfInfo;
  const bool m_PrintPredictions;
  Logger m_Logger;

  // TRT specific members
  const uint m_BatchSize;
  nvinfer1::INetworkDefinition* m_Network;
  nvinfer1::IBuilder* m_Builder;
  nvinfer1::IHostMemory* m_ModelStream;
  nvinfer1::ICudaEngine* m_Engine;
  nvinfer1::IExecutionContext* m_Context;
  std::vector<void*> m_DeviceBuffers;
  int m_InputBindingIndex;

  PluginFactory* m_PluginFactory;
  std::unique_ptr<YoloTinyMaxpoolPaddingFormula> m_TinyMaxpoolPaddingFormula;

  virtual std::vector<BBoxInfo> decodeTensor(const int imageIdx, const int imageH, const int imageW,
                                             const TensorInfo& tensor) = 0;

  inline void addBBoxProposal(const float bx, const float by, const float bw, const float bh, const uint stride,
                              const float scalingFactor, const float xOffset, const float yOffset, const int maxIndex,
                              const float maxProb, std::vector<BBoxInfo>& binfo)
  {
    BBoxInfo bbi;
    bbi.box = convertBBoxNetRes(bx, by, bw, bh, stride, m_InputW, m_InputH);
    if ((bbi.box.x1 > bbi.box.x2) || (bbi.box.y1 > bbi.box.y2))
    {
      return;
    }
    convertBBoxImgRes(scalingFactor, xOffset, yOffset, bbi.box);
    bbi.label = maxIndex;
    bbi.prob = maxProb;
    bbi.classId = getClassId(maxIndex);
    binfo.push_back(bbi);
  };

private:
  /// error message
  std::string compatibilityWarning = "Using an engine plan file across different models of devices is not recommended "
                                     "and is likely to affect performance or even cause errors.";
  std::string compatibilityError = "The engine plan file is generated on an incompatible device";
  void createYOLOEngine(const nvinfer1::DataType dataType = nvinfer1::DataType::kFLOAT,
                        Int8EntropyCalibrator* calibrator = nullptr);
  std::vector<std::map<std::string, std::string>> parseConfigFile(const std::string cfgFilePath);
  void parseConfigBlocks();
  void allocateBuffers();
  bool verifyYoloEngine();
  void destroyNetworkUtils(std::vector<nvinfer1::Weights>& trtWeights);
  void writePlanFileToDisk();
};
} // namespace DriveNet
#endif  // _YOLO_H_
