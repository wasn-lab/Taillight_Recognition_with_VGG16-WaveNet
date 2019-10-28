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
#include "parknet.h"
#include "parknet_args_parser.h"
#include "glog/logging.h"
#include <stdlib.h>
#include <chrono>
#include <sys/stat.h>

#if USE(TENSORRT)
#include "npp.h"
#include "trt_yolo3_detector.h"
#include "yolov2.h"
#include "yolov3.h"
#include "rect_class_score.h"
#include "parknet_camera.h"
#include "parknet_logging.h"
#include "parknet_fs_utils.h"
#include "camera_params.h"
#include "camera_utils.h"
#include "npp_utils.h"
#include "cuda_runtime_api.h"

namespace parknet
{
static std::string get_yolo_cfg_file()
{
  if (use_tiny_yolov3())
  {
    return TINY_YOLOV3_NETWORK_CFG_FILE;
  }
  else
  {
    return YOLOV3_NETWORK_CFG_FILE;
  }
}

static std::string get_yolo_weights_file()
{
  if (use_tiny_yolov3())
  {
    return TINY_YOLOV3_NETWORK_WEIGHTS_FILE;
  }
  else
  {
    return YOLOV3_NETWORK_WEIGHTS_FILE;
  }
}

static std::string get_network_type()
{
  if (use_tiny_yolov3())
  {
    return "yolov3-tiny";
  }
  else
  {
    return "yolov3";
  }
}

static std::string get_calibration_table()
{
  if (use_tiny_yolov3())
  {
    return PARKNET_CFG_DIR "/yolov3-tiny-calibration.table";
  }
  else
  {
    return PARKNET_CFG_DIR "/yolov3-calibration.table";
  }
}

TRTYolo3Detector::TRTYolo3Detector() : inferNet_(nullptr)
{
  // parse config params
  const int batchSize = 1;

  InferParams yolo_infer_params;
  NetworkInfo yolo_network_info;

  yolo_infer_params.printPerfInfo = true;
  yolo_infer_params.printPredictionInfo = false;
  yolo_infer_params.calibImages = ".";
  yolo_infer_params.calibImagesPath = ".";
  yolo_infer_params.probThresh = parknet::get_yolo_score_threshold();
  yolo_infer_params.nmsThresh = parknet::get_yolo_nms_threshold();

  yolo_network_info.configFilePath = parknet::get_yolo_cfg_file();
  LOG(INFO) << "Use cfg file: " << yolo_network_info.configFilePath;
  yolo_network_info.wtsFilePath = parknet::get_yolo_weights_file();
  LOG(INFO) << "Use weight file: " << yolo_network_info.wtsFilePath;
  yolo_network_info.labelsFilePath = YOLOV3_OBJECT_NAMES_FILE;
  LOG(INFO) << "Use labels file: " << yolo_network_info.labelsFilePath;
  yolo_network_info.precision = "kINT8";  // kFLOAT, kHALF, or kINT8
  yolo_network_info.deviceType = "kGPU";
  yolo_network_info.inputBlobName = "data";

  if (!parknet::is_file(yolo_network_info.wtsFilePath))
  {
    std::cout << "File not exists: " << yolo_network_info.wtsFilePath << "\n";
    std::cout << "Run \"make parknet_yolov3_weight\" or \"make parknet_tiny_yolov3_weight\" to download it\n";
  }

  srand(2222222);
  yolo_network_info.networkType = get_network_type();
  LOG(INFO) << "Use network type: " << yolo_network_info.networkType;
  yolo_network_info.calibrationTablePath = get_calibration_table();
  LOG(INFO) << "Use calibration table: " << yolo_network_info.calibrationTablePath;
  yolo_network_info.enginePath = parknet::get_trt_engine_fullpath(yolo_network_info.wtsFilePath);
  inferNet_ = std::unique_ptr<Yolo>{ new YoloV3(batchSize, yolo_network_info, yolo_infer_params) };
}

TRTYolo3Detector::~TRTYolo3Detector()
{
}

/**
 * Detect images represented by |in_cv_mat_image|, and save the result in |out_detections|.
 *
 * @param[out] out_detections
 * @param[in] in_cv_mat_image Image stored in cv::Mat.
 * @param[in] cam_id Camera id.
 */

int TRTYolo3Detector::detect(std::vector<RectClassScore<float> >& out_detections, cv::Mat& in_cv_mat_image,
                             const int cam_id)
{
  assert(::camera::yolov3_image_width == in_cv_mat_image.cols);
  assert(::camera::yolov3_image_height == in_cv_mat_image.rows);
  dsImages[cam_id].clear();
  dsImages[cam_id].emplace_back(DsImage(in_cv_mat_image, ::camera::yolov3_image_height, ::camera::yolov3_image_width));

  cv::Mat trtInput = blobFromDsImages(dsImages[cam_id], ::camera::yolov3_image_height, ::camera::yolov3_image_width);
  {
    std::lock_guard<std::mutex> lk(mu_);
    inferNet_->doInference(trtInput.data, dsImages[cam_id].size());
  }
  return get_detections(out_detections);
}

/**
 * Detect images represented by blob |npp32f_ptr_369664_rgb|, and save the result in |out_detections|.
 * The number 369664 is 608x608.
 *
 * @param[out] out_detections
 * @param[in] npp32f_ptr_369664_rgb Flattened image blob in the form R0, R1,...R369663, G0, G1,...G369663, B0,..B369663.
 * @param[in] cam_id Camera id.
 */
int TRTYolo3Detector::detect(std::vector<RectClassScore<float> >& out_detections, const Npp32f* npp32f_ptr_369664_rgb,
                             const int cam_id)
{
  {
    std::lock_guard<std::mutex> lk(mu_);
    inferNet_->doInference(npp32f_ptr_369664_rgb, /*batch size*/ 1);
  }
  return get_detections(out_detections);
}

int TRTYolo3Detector::get_detections(std::vector<RectClassScore<float> >& out_detections)
{
  const int nth_image = 0;
  auto binfo = inferNet_->decodeDetections(nth_image, ::camera::yolov3_image_height, ::camera::yolov3_image_width);
  auto remaining = nmsAllClasses(inferNet_->getNMSThresh(), binfo, inferNet_->getNumClasses());
  out_detections.clear();
  for (auto& b : remaining)
  {
    if (inferNet_->isPrintPredictions())
    {
      printPredictions(b, inferNet_->getClassName(b.label));
    }
    RectClassScore<float> detection;
    detection.x = b.box.x1;
    detection.y = b.box.y1;
    detection.w = b.box.x2 - b.box.x1;
    detection.h = b.box.y2 - b.box.y1;
    detection.score = b.prob;
    detection.class_type = b.classId;

    assert((detection.x >= 0) && (detection.x <= ::camera::raw_image_width));
    assert((detection.y >= 0) && (detection.y <= ::camera::raw_image_height));
    assert((detection.w >= 0) && (detection.w <= ::camera::raw_image_width));
    assert((detection.h >= 0) && (detection.h <= ::camera::raw_image_height));
    assert((detection.score >= 0) && (detection.score <= 1.0));

    out_detections.emplace_back(detection);
  }
  return 0;
}

}  // namespace parknet
#endif  // USE(TENSORRT)
