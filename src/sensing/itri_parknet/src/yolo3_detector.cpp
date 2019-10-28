/*
   CREATER: ICL U300
   DATE: Feb, 2019
 */

#include "parknet.h"
#include "yolo3_detector.h"
#include <sensor_msgs/image_encodings.h>

// 0:left_120 1:mid_120 2:right_120

std::stringstream ss;
cv::Mat currentFrame;

namespace darknet
{
void Yolo3Detector::load(std::string& in_model_file, std::string& in_trained_file, double in_score_thresh,
                         double in_nms_thresh)
{
  score_thresh_ = in_score_thresh;
  nms_thresh_ = in_nms_thresh;
  darknet_network_ = parse_network_cfg(&in_model_file[0]);
  load_weights(darknet_network_, &in_trained_file[0]);
  set_batch_network(darknet_network_, 1);

  layer output_layer = darknet_network_->layers[darknet_network_->n - 1];
  darknet_boxes_.resize(output_layer.w * output_layer.h * output_layer.n);
}

Yolo3Detector::~Yolo3Detector()
{
  if (darknet_network_)
  {
    free_network(darknet_network_);
  }
}

int Yolo3Detector::detect(std::vector<RectClassScore<float> >& out_detections, image& in_darknet_image)
{
  return forward(out_detections, in_darknet_image);
}

int Yolo3Detector::forward(std::vector<RectClassScore<float> >& out_detections, image& in_darknet_image)
{
  std::lock_guard<std::mutex> lock(darknet_network_mutex_);
  image sized = letterbox_image(in_darknet_image, darknet_network_->w, darknet_network_->h);

  float* in_data = sized.data;
  float* prediction = network_predict(darknet_network_, in_data);
  layer output_layer = darknet_network_->layers[darknet_network_->n - 1];

  output_layer.output = prediction;
  int nboxes = 0;
  int num_classes = output_layer.classes;
  detection* darknet_detections = get_network_boxes(darknet_network_, darknet_network_->w, darknet_network_->h,
                                                    score_thresh_, .5, NULL, 0, &nboxes);

  do_nms_sort(darknet_detections, nboxes, num_classes, nms_thresh_);

  out_detections.clear();
  for (int i = 0; i < nboxes; i++)
  {
    int class_id = -1;
    float score = 0.f;
    // find the class
    for (int j = 0; j < num_classes; ++j)
    {
      if (darknet_detections[i].prob[j] >= score_thresh_)
      {
        if (class_id < 0)
        {
          class_id = j;
          score = darknet_detections[i].prob[j];
        }
      }
    }
    // if class found
    if (class_id >= 0)
    {
      RectClassScore<float> detection;

      detection.x = darknet_detections[i].bbox.x - darknet_detections[i].bbox.w / 2;
      detection.y = darknet_detections[i].bbox.y - darknet_detections[i].bbox.h / 2;
      detection.w = darknet_detections[i].bbox.w;
      detection.h = darknet_detections[i].bbox.h;
      detection.score = score;
      detection.class_type = class_id;

      out_detections.emplace_back(detection);
    }
  }
  free_detections(darknet_detections, nboxes);
  free_image(sized);
  return 0;
}

uint32_t Yolo3Detector::get_network_height()
{
  return darknet_network_->h;
}

uint32_t Yolo3Detector::get_network_width()
{
  return darknet_network_->w;
}

}  // namespace darknet
