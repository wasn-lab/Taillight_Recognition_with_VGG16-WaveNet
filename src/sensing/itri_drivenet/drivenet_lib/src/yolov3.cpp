#include "yolov3.h"

namespace DriveNet
{
YoloV3::YoloV3(const uint batchSize, const NetworkInfo& networkInfo, const InferParams& inferParams)
  : Yolo(batchSize, networkInfo, inferParams){};

std::vector<BBoxInfo> YoloV3::decodeTensor(const int imageIdx, const int imageH, const int imageW,
                                           const TensorInfo& tensor)
{
  float scalingFactor = std::min(static_cast<float>(m_InputW) / imageW, static_cast<float>(m_InputH) / imageH);
  float xOffset = (m_InputW - scalingFactor * imageW) / 2;
  float yOffset = (m_InputH - scalingFactor * imageH) / 2;

  const float* detections = &tensor.hostBuffer[imageIdx * tensor.volume];

  std::vector<BBoxInfo> binfo;
  for (uint y = 0; y < tensor.gridSize; ++y)
  {
    for (uint x = 0; x < tensor.gridSize; ++x)
    {
      for (uint b = 0; b < tensor.numBBoxes; ++b)
      {
        const float pw = tensor.anchors[tensor.masks[b] * 2];
        const float ph = tensor.anchors[tensor.masks[b] * 2 + 1];

        const int numGridCells = tensor.gridSize * tensor.gridSize;
        const int bbindex = y * tensor.gridSize + x;
        const float bx = x + detections[bbindex + numGridCells * (b * (5 + tensor.numClasses) + 0)];

        const float by = y + detections[bbindex + numGridCells * (b * (5 + tensor.numClasses) + 1)];
        const float bw = pw * detections[bbindex + numGridCells * (b * (5 + tensor.numClasses) + 2)];
        const float bh = ph * detections[bbindex + numGridCells * (b * (5 + tensor.numClasses) + 3)];

        const float objectness = detections[bbindex + numGridCells * (b * (5 + tensor.numClasses) + 4)];

        float maxProb = 0.0f;
        int maxIndex = -1;

        for (uint i = 0; i < tensor.numClasses; ++i)
        {
          float prob = (detections[bbindex + numGridCells * (b * (5 + tensor.numClasses) + (5 + i))]);

          if (prob > maxProb)
          {
            maxProb = prob;
            maxIndex = i;
          }
        }
        maxProb = objectness * maxProb;

        float probThresh = m_ProbThresh;
        if (maxIndex == 3 || maxIndex == 1)
        {
          if (maxProb > m_ProbThreshBike)
          {
            probThresh = m_ProbThreshBike;
          }
        }
        if (maxProb > probThresh)
        {
          addBBoxProposal(bx, by, bw, bh, tensor.stride, scalingFactor, xOffset, yOffset, maxIndex, maxProb, binfo);
        }
      }
    }
  }
  return binfo;
}
}