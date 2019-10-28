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
#include "yolov3.h"

YoloV3::YoloV3(const uint batchSize, const NetworkInfo& networkInfo,
               const InferParams& inferParams) :
    Yolo(batchSize, networkInfo, inferParams){};

std::vector<BBoxInfo> YoloV3::decodeTensor(const int imageIdx, const int imageH, const int imageW,
                                           const TensorInfo& tensor)
{
    float scalingFactor
        = std::min(static_cast<float>(m_InputW) / imageW, static_cast<float>(m_InputH) / imageH);
    float xOffset = (m_InputW - scalingFactor * imageW) / 2;
    float yOffset = (m_InputH - scalingFactor * imageH) / 2;

    const float* detections = &tensor.hostBuffer[imageIdx * tensor.volume];

    std::vector<BBoxInfo> binfo;
    const int numGridCells = tensor.gridSize * tensor.gridSize;
    int bbindex_offset_on_y = 0;
    for (uint y = 0; y < tensor.gridSize; ++y)
    {
        for (uint x = 0; x < tensor.gridSize; ++x)
        {
            const int bbindex = bbindex_offset_on_y + x;
            for (uint b = 0; b < tensor.numBBoxes; ++b)
            {
                const auto pw_index = (tensor.masks[b] << 1);
                const float pw = tensor.anchors[pw_index];
                const float ph = tensor.anchors[pw_index + 1];

                const int bx_index = bbindex + numGridCells * (b * (5 + tensor.numClasses) + 0);
                const int by_index = bx_index + numGridCells;
                const int bw_index = by_index + numGridCells;
                const int bh_index = bw_index + numGridCells;
                const int objectness_index = bh_index + numGridCells;
                const float bx = x + detections[bx_index];
                const float by = y + detections[by_index];
                const float bw = pw * detections[bw_index];
                const float bh = ph * detections[bh_index];

                const float objectness = detections[objectness_index];

                float maxProb = 0.0f;
                int maxIndex = -1;

                int prob_index = bx_index + numGridCells;
                for (uint i = 0; i < tensor.numClasses; ++i)
                {
                    // float prob = detections[bx_index + numGridCells * (1 + i)];
                    float prob = detections[prob_index];
                    prob_index += numGridCells;
                    if (prob > maxProb)
                    {
                        maxProb = prob;
                        maxIndex = i;
                    }
                }
                maxProb = objectness * maxProb;

                if (maxProb > m_ProbThresh)
                {
                    addBBoxProposal(bx, by, bw, bh, tensor.stride, scalingFactor, xOffset, yOffset,
                                    maxIndex, maxProb, binfo);
                }
            }
        }
        bbindex_offset_on_y += tensor.gridSize;
    }
    return binfo;
}
