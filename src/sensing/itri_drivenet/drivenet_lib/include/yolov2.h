#ifndef _YOLO_V2_
#define _YOLO_V2_

#include "yolo.h"

#include <stdint.h>
#include <string>
#include <vector>

namespace DriveNet{
class YoloV2 : public Yolo
{
public:
    YoloV2(const uint batchSize, const NetworkInfo& networkInfo, const InferParams& inferParams);

private:
    std::vector<BBoxInfo> decodeTensor(const int imageIdx, const int imageH, const int imageW,
                                       const TensorInfo& tensor) override;
};
}
#endif // _YOLO_V2_