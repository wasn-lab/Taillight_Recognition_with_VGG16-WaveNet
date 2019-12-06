#ifndef TRT_YOLO_INTERFACE_H
#define TRT_YOLO_INTERFACE_H

/* OpenCV headers */
#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "ds_image.h"
#include "trt_utils.h"
#include "yolo.h"
#include "yolo_config_parser.h"
#include "yolov3.h"

using namespace DriveNet;
struct ITRI_Bbox
{
  int32_t label;
  int32_t classId;  // For coco benchmarking
  float prob;
  float x1;
  float y1;
  float x2;
  float y2;
};

class Yolo_app
{
public:
  Yolo* inferYolo;
  bool decode;
  bool doBenchmark;
  bool viewDetections;
  bool saveDetections;
  uint batchSize;
  bool shuffleTestSet;

  DsImage dsImags;

  std::vector<int> dsImgs_rows;
  std::vector<int> dsImgs_cols;

  float* yoloInput;

  void init_yolo(std::string pkg_path, std::string cfg_file);
  void input_preprocess(std::vector<cv::Mat*>& matSrcs);
  void input_preprocess(std::vector<cv::Mat*>& matSrcs, int input_size, std::vector<int> dist_w,
                        std::vector<int> dist_h);
  void inference_yolo();
  void get_yolo_result(std::vector<uint32_t>* order, std::vector<std::vector<ITRI_Bbox>*>& vbbx_output);
  void delete_yolo_infer();
};
#endif /*TRT_YOLO_INTERFACE_H*/
