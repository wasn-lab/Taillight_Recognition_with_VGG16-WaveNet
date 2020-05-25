#include <iostream>
#include <unordered_map>
#include <vector>
#include <string>
#include <cassert>

#include <glog/logging.h>
#include <opencv2/opencv.hpp>
#include <tensorflow/c/c_api.h>

#include "deeplab_segmenter_impl.h"
#include "deeplab_args_parser.h"
#include "deeplab_const.h"
#include "tf_utils.hpp"
#include "cv_color.h"

/*
classes:
0 background
1 aeroplane
2 bicycle
3 bird
4 boat
5 bottle
6 bus
7 car
8 cat
9 chair
10 cow
11 diningtable
12 dog
13 horse
14 motorbike
15 person
16 pottedplant
17 sheep
18 sofa
19 train
20 tv
*/

namespace deeplab
{
const std::vector<std::string> g_label_names{
  "background",  "aeroplane", "bicycle", "bird",      "boat",   "bottle",      "bus",   "car",  "cat",   "chair", "cow",
  "diningtable", "dog",       "horse",   "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tv"
};

static int32_t resize_to_deeplab_input(const cv::Mat& img_in, cv::Mat& img_out)
{
  cv::Mat tmp;

  cv::copyMakeBorder(img_in, tmp, PADDING_TOP_IN_PIXELS, PADDING_BOTTOM_IN_PIXELS, /*left*/ 0, /*right*/ 0,
                     cv::BORDER_CONSTANT, camera_utils::get_cv_color(camera_utils::color::black));
  cv::resize(tmp, img_out, cv::Size(DEEPLAB_IMAGE_WIDTH, DEEPLAB_IMAGE_HEIGHT));
  assert(img_out.rows == DEEPLAB_IMAGE_HEIGHT);
  assert(img_out.cols == DEEPLAB_IMAGE_WIDTH);
  return 0;
}

static int32_t resize_to_608x384(cv::Mat& img_out)
{
  cv::resize(img_out, img_out, cv::Size(ROS_IMAGE_WIDTH, ROS_IMAGE_WIDTH));
  cv::Rect roi;
  roi.x = 0;
  roi.y = PADDING_TOP_IN_PIXELS;
  roi.width = ROS_IMAGE_WIDTH;
  roi.height = ROS_IMAGE_HEIGHT;

  img_out = img_out(roi);

  assert(img_out.rows == ROS_IMAGE_HEIGHT);
  assert(img_out.cols == ROS_IMAGE_WIDTH);
  return 0;
}

DeeplabSegmenterImpl::DeeplabSegmenterImpl()
  : tf_graph_(tf_utils::LoadGraph(get_pb_file().c_str()), tf_utils::DeleteGraph)
  , tf_status_(TF_NewStatus(), TF_DeleteStatus)
  , tf_sess_options_(nullptr, TF_DeleteSessionOptions)
  , tf_sess_(tf_utils::CreateSession(tf_graph_.get()), tf_utils::DeleteSession)
  , input_tensor_(nullptr)
  , output_tensor_(nullptr)
{
  LOG(INFO) << "Load " << get_pb_file();
  if (tf_graph_.get() == nullptr)
  {
    LOG(FATAL) << "Can't load graph " << get_pb_file();
  }

  const int64_t input_dims[4] = { 1, DEEPLAB_IMAGE_HEIGHT, DEEPLAB_IMAGE_WIDTH, 3 };
  input_tensor_ = tf_utils::CreateTensor(TF_UINT8, input_dims, 4, nullptr, INPUT_TENSOR_SIZE_IN_BYTES);
}

DeeplabSegmenterImpl::~DeeplabSegmenterImpl()
{
  tf_utils::DeleteTensor(input_tensor_);
  input_tensor_ = nullptr;
  tf_utils::DeleteTensor(output_tensor_);
  output_tensor_ = nullptr;
}

/*
 * Returns:
 *   - 0: No resize
 *   - 1: resize image_in to DEEPLAB_IMAGE_WIDTH * image_height
 **/
int32_t DeeplabSegmenterImpl::preprocess_for_input_tensor(const cv::Mat& img_in, cv::Mat& img_rgb)
{
  cv::Mat img_bgr;
  int32_t resized = 0;

  assert(img_in.rows > 0);
  assert(img_in.cols > 0);
  if (img_in.rows != DEEPLAB_IMAGE_HEIGHT || img_in.cols != DEEPLAB_IMAGE_WIDTH)
  {
    LOG(WARNING) << "Expect image size " << DEEPLAB_IMAGE_WIDTH << "x" << DEEPLAB_IMAGE_HEIGHT << ", Got "
                 << img_in.cols << "x" << img_in.rows << ". Resize to fit deeplab requirement.";
    resize_to_deeplab_input(img_in, img_bgr);
    resized = 1;
  }
  else
  {
    img_bgr = img_in.clone();
  }
  cv::cvtColor(img_bgr, img_rgb, cv::COLOR_BGR2RGB);
  return resized;
}

int32_t DeeplabSegmenterImpl::inference(const cv::Mat& img_rgb)
{
  // Deeplab requires RGB layout, which is different from cv::Mat.
  tf_utils::SetTensorsData(input_tensor_, img_rgb.ptr(), INPUT_TENSOR_SIZE_IN_BYTES);

  TF_Output input_op = { TF_GraphOperationByName(tf_graph_.get(), "ImageTensor"), 0 };
  TF_Output out_op = { TF_GraphOperationByName(tf_graph_.get(), "SemanticPredictions"), 0 };

  TF_SessionRun(tf_sess_.get(),
                nullptr,                       // Run options.
                &input_op, &input_tensor_, 1,  // Input tensors, input tensor values, number of inputs.
                // input_op.data(), input_tensor.data(), input_tensor.size(), // Input tensors, input tensor values,
                // number of inputs.
                &out_op, &output_tensor_, 1,  // Output tensors, output tensor values, number of outputs.
                nullptr, 0,                   // Target operations, number of targets.
                nullptr,                      // Run metadata.
                tf_status_.get()              // Output status.
  );

  if (TF_GetCode(tf_status_.get()) != TF_OK)
  {
    LOG(WARNING) << "Error run session";
    return 1;
  }
  return 0;
}

int32_t DeeplabSegmenterImpl::postprocess_with_labels(const int64_t* labels, const cv::Mat& img_rgb, cv::Mat& img_out)
{
  // Mark pixels with specific colors.
  cv::Mat overlay(img_rgb.size(), img_rgb.type());
  uint32_t kth_pixel = 0;
  for (int32_t a = 0; a < DEEPLAB_IMAGE_HEIGHT; a++)
  {
    for (int32_t b = 0; b < DEEPLAB_IMAGE_WIDTH; b++)
    {
      auto label = static_cast<camera_utils::color>(labels[kth_pixel]);
      auto& ocolor = overlay.at<cv::Vec3b>(a, b);
      if (label > 0)
      {
        auto label_color = camera_utils::get_cv_color(label);
        for (int32_t i = 0; i < 3; i++)
        {
          ocolor[i] = label_color[i];
        }
      }
      else
      {
        ocolor = img_rgb.at<cv::Vec3b>(a, b);
      }
      kth_pixel++;
    }
  }

  // overlay * alpha + img_in * beta + gamma = img_out
  const double alpha = 0.7;
  const auto beta = 1 - alpha;
  cv::addWeighted(overlay, alpha, img_rgb, beta, /*gamma*/ 0, img_out);
  cv::cvtColor(img_out, img_out, cv::COLOR_RGB2BGR);
  return 0;
}

int32_t DeeplabSegmenterImpl::segment_into_labels(const cv::Mat& img_in, uint8_t* labels)
{
  cv::Mat img_rgb;
  preprocess_for_input_tensor(img_in, img_rgb);
  inference(img_rgb);

  auto* labels64 = static_cast<int64_t*>(TF_TensorData(output_tensor_));
  for (auto idx = 0; idx < NUM_PIXELS; idx++)
  {
    labels[idx] = static_cast<uint8_t>(labels64[idx]);
  }
  tf_utils::DeleteTensor(output_tensor_);
  output_tensor_ = nullptr;
  return 0;
}

int32_t DeeplabSegmenterImpl::segment(const cv::Mat& img_in, cv::Mat& img_out)
{
  cv::Mat img_rgb;
  auto resized = preprocess_for_input_tensor(img_in, img_rgb);
  inference(img_rgb);
  postprocess_with_labels(static_cast<int64_t*>(TF_TensorData(output_tensor_)), img_rgb, img_out);

  if (resized)
  {
    resize_to_608x384(img_out);
  }

  // Delete output_tensor_ in every TF_SessionRun to avoid memory leak.
  // See https://github.com/tensorflow/tensorflow/issues/29733
  tf_utils::DeleteTensor(output_tensor_);
  output_tensor_ = nullptr;
  return 0;
}

};  // namespace deeplab
