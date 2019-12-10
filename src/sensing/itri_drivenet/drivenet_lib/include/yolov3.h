#ifndef _YOLO_V3_
#define _YOLO_V3_

#include "yolo.h"

#include <stdint.h>
#include <string>
#include <vector>

namespace DriveNet
{
class YoloV3 : public Yolo
{
public:
  YoloV3(const uint batchSize, const NetworkInfo& networkInfo, const InferParams& inferParams);

private:
  std::vector<BBoxInfo> decodeTensor(const int imageIdx, const int imageH, const int imageW,
                                     const TensorInfo& tensor) override;
};
}
#endif  // _YOLO_V3_