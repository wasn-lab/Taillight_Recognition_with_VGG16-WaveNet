#include <iostream>
#include <unordered_map>
#include <vector>
#include <string>

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

DeeplabSegmenterImpl::DeeplabSegmenterImpl()
  : tf_graph_(tf_utils::LoadGraph(get_pb_file().c_str()), tf_utils::DeleteGraph)
  , tf_status_(nullptr, TF_DeleteStatus)
  , tf_sess_options_(nullptr, TF_DeleteSessionOptions)
  , tf_sess_(nullptr, tf_utils::DeleteSession)
{
  LOG(INFO) << "Load " << get_pb_file();
  if (tf_graph_.get() == nullptr)
  {
    LOG(ERROR) << "Can't load graph " << get_pb_file();
  }

#if 1
  tf_status_.reset(TF_NewStatus());
  tf_sess_.reset(tf_utils::CreateSession(tf_graph_.get()));
#else
  tf_sess_options_.reset(TF_NewSessionOptions());

  uint8_t config[16] = { 0x32, 0xe, 0x9, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0xe0, 0x3f, 0x20, 0x1, 0x2a, 0x1, 0x30 };
  TF_SetConfig(tf_sess_options_.get(), (void*)config, 16, tf_status_.get());

  tf_sess_.reset(TF_NewSession(tf_graph_.get(), tf_sess_options_.get(), tf_status_.get()));
#endif
}

DeeplabSegmenterImpl::~DeeplabSegmenterImpl() = default;

int DeeplabSegmenterImpl::segment(const cv::Mat& img_in, cv::Mat& img_out)
{
  const int64_t input_dims[4] = { 1, image_height, image_width, 3 };
  //  auto rgb_blob = std::move(blob_from_image(img_in));
  cv::Mat img_rgb = img_in.clone();
  cv::cvtColor(img_rgb, img_rgb, cv::COLOR_BGR2RGB);
  TF_Tensor* input_tensor =
      tf_utils::CreateTensor(TF_UINT8, input_dims, 4, img_rgb.ptr(), num_pixels * img_in.channels() * sizeof(uint8_t));

  TF_Tensor* output_tensor = nullptr;
  TF_Output input_op = { TF_GraphOperationByName(tf_graph_.get(), "ImageTensor"), 0 };
  TF_Output out_op = { TF_GraphOperationByName(tf_graph_.get(), "SemanticPredictions"), 0 };

  TF_SessionRun(tf_sess_.get(),
                nullptr,                      // Run options.
                &input_op, &input_tensor, 1,  // Input tensors, input tensor values, number of inputs.
                // input_op.data(), input_tensor.data(), input_tensor.size(), // Input tensors, input tensor values,
                // number of inputs.
                &out_op, &output_tensor, 1,  // Output tensors, output tensor values, number of outputs.
                nullptr, 0,                  // Target operations, number of targets.
                nullptr,                     // Run metadata.
                tf_status_.get()             // Output status.
  );

  if (TF_GetCode(tf_status_.get()) != TF_OK)
  {
    LOG(WARNING) << "Error run session";
  }
#if 0 
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/c/tf_datatype.h

  LOG(INFO) << "output tensor byte size: " << TF_TensorByteSize(output_tensor);
  LOG(INFO) << "output tensor type: " << TF_TensorType(output_tensor);
  LOG(INFO) << "output tensor dim " << TF_NumDims(output_tensor);
  for(int i=0; i<TF_NumDims(output_tensor); i++)
  {
    LOG(INFO) << "output tensor dim " << i << " : " << TF_Dim(output_tensor, i);
  }
#endif
  const auto labels = static_cast<int64_t*>(TF_TensorData(output_tensor));  // Return a pointer to the underlying data
                                                                            // buffer of TF_Tensor*

  std::unordered_map<int64_t, int> counter;
  cv::Mat overlay(img_in.size(), img_in.type());
  for (int a = 0; a < image_height; a++)
  {
    for (int b = 0; b < image_width; b++)
    {
      auto label = static_cast<camera_utils::color>(labels[image_width * a + b]);
      auto& ocolor = overlay.at<cv::Vec3b>(a, b);
      if (label > 0)
      {
        auto label_color = camera_utils::get_cv_color(label);
        for (int i = 0; i < 3; i++)
        {
          ocolor[i] = label_color[i];
        }
      }
      else
      {
        ocolor = img_in.at<cv::Vec3b>(a, b);
      }
      if (counter.find(label) == counter.end())
      {
        counter[label] = 1;
      }
      else
      {
        counter[label]++;
      }
    }
  }

  // overlay * alpha + img_in * beta + gamma = img_out
  const double alpha = 0.4;
  const auto beta = 1 - alpha;
  cv::addWeighted(overlay, alpha, img_in, beta, /*gamma*/ 0, img_out);
  for (const auto& kv : counter)
  {
    LOG(INFO) << g_label_names[kv.first] << " : " << kv.second << " pixels";
  }
  return 0;
}

};  // namespace deeplab
