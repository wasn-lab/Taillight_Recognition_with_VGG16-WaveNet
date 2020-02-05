#ifndef _CALIBRATOR_H_
#define _CALIBRATOR_H_

#include "NvInfer.h"
#include "ds_image.h"
#include "trt_utils.h"

namespace DriveNet
{
class Int8EntropyCalibrator : public nvinfer1::IInt8EntropyCalibrator
{
public:
  Int8EntropyCalibrator(const uint& batchSize, const std::string& calibImages, const std::string& calibImagesPath,
                        const std::string& calibTableFilePath, const uint64_t& inputSize, const uint& inputH,
                        const uint& inputW, const std::string& inputBlobName);
  virtual ~Int8EntropyCalibrator();

  int getBatchSize() const override
  {
    return m_BatchSize;
  }
  bool getBatch(void* bindings[], const char* names[], int nbBindings) override;
  const void* readCalibrationCache(size_t& length) override;
  void writeCalibrationCache(const void* cache, size_t length) override;

private:
  const uint m_BatchSize;
  const uint m_InputH;
  const uint m_InputW;
  const uint64_t m_InputCount;
  const std::string m_InputBlobName;
  const std::string m_CalibTableFilePath{ nullptr };
  uint m_ImageIndex;
  bool m_ReadCache{ true };
  void* m_DeviceInput{ nullptr };
  std::vector<std::string> m_ImageList;
  std::vector<char> m_CalibrationCache;
};
}
#endif