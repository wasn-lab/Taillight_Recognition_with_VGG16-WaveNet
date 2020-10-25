// ----------------------- OpenPose C++ API Tutorial - Example 3 - Body from image -----------------------
// It reads an image, process it, and displays it with the pose (and optionally hand and face) keypoints. In addition,
// it includes all the OpenPose configuration flags (enable/disable hand, face, output saving, etc.).

// Third-party dependencies
#include "ped_tf_def.h"
#include "ros/ros.h"
#include <tensorflow/c/c_api.h>  // TensorFlow C API header.
#include "tf_utils.hpp"
#include <scope_guard.hpp>
#include "msgs/PredictCrossing.h"
// Command-line user interface
// OpenPose dependencies
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>  // opencv general include file

#define FEATURE_NUM 1174
#define FRAME_NUM 10

const std::vector<std::int64_t> INPUT_DIMS = { 1, FRAME_NUM, FEATURE_NUM };

TF_Graph* g_graph = nullptr;
TF_Status* g_status = nullptr;
TF_SessionOptions* g_options = nullptr;
TF_Session* g_sess = nullptr;

// return Euclidian distance between two points
float get_distance2(float x1, float y1, float x2, float y2)
{
  return sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2));
}

// return degree with line formed by two points and vertical line
float get_angle2(float x1, float y1, float x2, float y2)
{
  return M_PI / 2 - std::atan2(std::fabs(y1 - y2), std::fabs(x1 - x2));
}

// return 3 inner angles of the triangle formed by three points
float* get_triangle_angle(float x1, float y1, float x2, float y2, float x3, float y3)
{
  float a = get_distance2(x1, y1, x2, y2);
  float b = get_distance2(x2, y2, x3, y3);
  float c = get_distance2(x1, y1, x3, y3);
  float test = (a * a + c * c - b * b) / (2 * a * c);
  static float angle[3] = { 0.0f, 0.0f, 0.0f };
  if (test <= 1 && test >= -1)
  {
    angle[0] = std::acos((a * a + c * c - b * b) / (2 * a * c));
    angle[1] = std::acos((a * a + b * b - c * c) / (2 * a * b));
    angle[2] = M_PI - angle[0] - angle[1];
  }
  else
  {
    if (std::max(a, std::max(b, c)) == a)
    {
      angle[2] = M_PI;
    }
    else if (std::max(a, std::max(b, c)) == b)
    {
      angle[0] = M_PI;
    }
    else
    {
      angle[1] = M_PI;
    }
  }
  return angle;
}

bool callback(msgs::PredictCrossing::Request& req, msgs::PredictCrossing::Response& res)
{
  ros::Time start = ros::Time::now();

  std::vector<std::vector<cv::Point2f>> keypoint_array;
  // get processed_keypoints return from skip_frame service
  for (unsigned int i = 0; i < req.keypoints.size(); i++)
  {
    std::vector<cv::Point2f> back_predict_keypoints;
    for (unsigned int j = 0; j < req.keypoints.at(i).keypoint.size(); j++)
    {
      cv::Point2f back_predict_keypoint;
      back_predict_keypoint.x = req.keypoints.at(i).keypoint.at(j).x;
      back_predict_keypoint.y = req.keypoints.at(i).keypoint.at(j).y;
      back_predict_keypoints.emplace_back(back_predict_keypoint);
    }
    keypoint_array.emplace_back(back_predict_keypoints);
    back_predict_keypoints.clear();
    std::vector<cv::Point2f>().swap(back_predict_keypoints);
  }

  std::vector<std::vector<float>> bbox_array;
  // get processed_keypoints return from skip_frame service
  for (unsigned int i = 0; i < req.bboxes.size(); i++)
  {
    std::vector<float> bbox;
    bbox.emplace_back(req.bboxes.at(i).u / 608);
    bbox.emplace_back(req.bboxes.at(i).v / 384);
    bbox.emplace_back((req.bboxes.at(i).u + req.bboxes.at(i).width) / 608);
    bbox.emplace_back((req.bboxes.at(i).v + req.bboxes.at(i).height) / 384);
    bbox_array.emplace_back(bbox);
    bbox.clear();
    std::vector<float>().swap(bbox);
  }

  // initialize feature
  std::vector<float> feature;

  for (unsigned int index = 0; index < FRAME_NUM; index++)
  {
    // Add bbox to feature vector
    std::vector<float> bbox = bbox_array.at(index);
    feature.insert(feature.end(), bbox.begin(), bbox.end());

    std::vector<cv::Point2f> keypoint = keypoint_array.at(index);
    if (!keypoint.empty())
    {
      std::vector<float> keypoints_x;
      std::vector<float> keypoints_y;

      // Get body keypoints we need
      int body_part[13] = { 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14 };
      int body_part_size = sizeof(body_part) / sizeof(*body_part);
      for (int i = 0; i < body_part_size; i++)
      {
        keypoints_x.insert(keypoints_x.end(), keypoint[body_part[i]].x);
        keypoints_y.insert(keypoints_y.end(), keypoint[body_part[i]].y);
      }

      // Calculate x_distance, y_distance, distance, angle
      for (int m = 0; m < body_part_size; m++)
      {
        for (int n = m + 1; n < body_part_size; n++)
        {
          float dist_x, dist_y, dist, angle;
          if (keypoints_x[m] != 0.0f && keypoints_y[m] != 0.0f && keypoints_x[n] != 0.0f && keypoints_y[n] != 0.0f)
          {
            dist_x = std::fabs(keypoints_x[m] - keypoints_x[n]);
            dist_y = std::fabs(keypoints_y[m] - keypoints_y[n]);
            dist = get_distance2(keypoints_x[m], keypoints_y[m], keypoints_x[n], keypoints_y[n]);
            angle = get_angle2(keypoints_x[m], keypoints_y[m], keypoints_x[n], keypoints_y[n]);
          }
          else
          {
            dist_x = 0.0f;
            dist_y = 0.0f;
            dist = 0.0f;
            angle = 0.0f;
          }
          float input[] = { dist_x, dist_y, dist, angle };
          feature.insert(feature.end(), input, input + sizeof(input) / sizeof(input[0]));
        }
      }

      // Calculate 3 inner angles of each 3 keypoints
      for (int m = 0; m < body_part_size; m++)
      {
        for (int n = m + 1; n < body_part_size; n++)
        {
          for (int k = n + 1; k < body_part_size; k++)
          {
            float angle[3] = { 0.0f, 0.0f, 0.0f };
            float* angle_ptr;
            if ((keypoints_x[m] != 0.0f || keypoints_y[m] != 0.0f) &&
                (keypoints_x[n] != 0.0f || keypoints_y[n] != 0.0f) &&
                (keypoints_x[k] != 0.0f || keypoints_y[k] != 0.0f))
            {
              angle_ptr = get_triangle_angle(keypoints_x[m], keypoints_y[m], keypoints_x[n], keypoints_y[n],
                                             keypoints_x[k], keypoints_y[k]);
              angle[0] = *angle_ptr;
              angle[1] = *(angle_ptr + 1);
              angle[2] = *(angle_ptr + 2);
            }
            feature.insert(feature.end(), angle, angle + sizeof(angle) / sizeof(angle[0]));
          }
        }
      }
      keypoints_x.clear();
      std::vector<float>().swap(keypoints_x);
      keypoints_y.clear();
      std::vector<float>().swap(keypoints_y);
    }
    else  // if keypoint is empty
    {
      float* zero_arr;
      // The first four feature are bb_x1, bb_y1, bb_x2, bb_y2
      int other_feature = FEATURE_NUM - 4;
      zero_arr = new float[other_feature]();
      feature.insert(feature.end(), zero_arr, zero_arr + other_feature);
      delete[] zero_arr;
    }
  }

  std::cout << "Init input_op testing..." << std::endl;
  auto input_op = TF_Output{ TF_GraphOperationByName(g_graph, "modelInput"), 0 };
  if (input_op.oper == nullptr)
  {
    std::cout << "Can't init input_op" << std::endl;
  }

  std::cout << "Init Output_op testing..." << std::endl;
  auto out_op = TF_Output{ TF_GraphOperationByName(g_graph, "dense_3/Sigmoid"), 0 };
  if (out_op.oper == nullptr)
  {
    std::cout << "Can't init out_op" << std::endl;
  }

  std::cout << "Create Output_tensor ..." << std::endl;
  TF_Tensor* output_tensor = nullptr;
  SCOPE_EXIT
  {
    tf_utils::DeleteTensor(output_tensor);
  };  // Auto-delete on scope exit.

  std::cout << "Create Input_tensor testing..." << std::endl;
  auto input_tensor = tf_utils::CreateTensor(TF_FLOAT, INPUT_DIMS, feature);
  SCOPE_EXIT
  {
    tf_utils::DeleteTensor(input_tensor);
  };  // Auto-delete on scope exit.
  std::cout << "Running session ..." << std::endl;
  TF_SessionRun(g_sess,
                nullptr,                      // Run options.
                &input_op, &input_tensor, 1,  // Input tensors, input tensor values, number of inputs.
                &out_op, &output_tensor, 1,   // Output tensors, output tensor values, number of outputs.
                nullptr, 0,                   // Target operations, number of targets.
                nullptr,                      // Run metadata.
                g_status                      // Output status.
  );

  std::cout << "Getting Output ..." << std::endl;
  auto data = static_cast<float*>(TF_TensorData(output_tensor));
  ros::Time stop = ros::Time::now();
  res.result_0 = data[0];
  res.result_1 = data[1];
  std::cout << "Output vals: " << data[0] << ", " << data[1] << ", " << stop - start << std::endl;

  if (TF_GetCode(g_status) != TF_OK)
  {
    std::cout << "Error run session";
  }

  return true;
}

int main(int argc, char** argv)
{
  ros::Time::init();
  ros::Time start = ros::Time::now();

  std::cout << "Load graph testing..." << std::endl;

  g_graph = tf_utils::LoadGraph((PED_TF_MODEL_DIR + std::string("/keras_2.2.4_model.pb")).c_str(), nullptr, nullptr);
  SCOPE_EXIT
  {
    tf_utils::DeleteGraph(g_graph);
  };  // Auto-delete on scope exit.
  if (g_graph == nullptr)
  {
    std::cout << "Can't load graph" << std::endl;
  }

  g_status = TF_NewStatus();
  SCOPE_EXIT
  {
    TF_DeleteStatus(g_status);
  };  // Auto-delete on scope exit.
  g_options = TF_NewSessionOptions();

  std::cout << "Creating New session ..." << std::endl;
  g_sess = TF_NewSession(g_graph, g_options, g_status);
  if (TF_GetCode(g_status) != TF_OK)
  {
    std::cout << "Can't create New session" << std::endl;
  }

  // run a blank session to load libcublas.so
  std::vector<float> feature;
  float* zero_arr;
  // The first four feature are bb_x1, bb_y1, bb_x2, bb_y2
  int other_feature = FEATURE_NUM * FRAME_NUM;
  zero_arr = new float[other_feature]();
  feature.insert(feature.end(), zero_arr, zero_arr + other_feature);
  delete[] zero_arr;

  std::cout << "Init input_op testing..." << std::endl;
  auto input_op = TF_Output{ TF_GraphOperationByName(g_graph, "modelInput"), 0 };
  if (input_op.oper == nullptr)
  {
    std::cout << "Can't init input_op" << std::endl;
  }

  std::cout << "Init Output_op testing..." << std::endl;
  auto out_op = TF_Output{ TF_GraphOperationByName(g_graph, "dense_3/Sigmoid"), 0 };
  if (out_op.oper == nullptr)
  {
    std::cout << "Can't init out_op" << std::endl;
  }

  std::cout << "Create Output_tensor ..." << std::endl;
  TF_Tensor* output_tensor = nullptr;
  SCOPE_EXIT
  {
    tf_utils::DeleteTensor(output_tensor);
  };  // Auto-delete on scope exit.

  std::cout << "Create Input_tensor testing..." << std::endl;
  auto input_tensor = tf_utils::CreateTensor(TF_FLOAT, INPUT_DIMS, feature);
  SCOPE_EXIT
  {
    tf_utils::DeleteTensor(input_tensor);
  };  // Auto-delete on scope exit.
  std::cout << "Running session ..." << std::endl;
  TF_SessionRun(g_sess,
                nullptr,                      // Run options.
                &input_op, &input_tensor, 1,  // Input tensors, input tensor values, number of inputs.
                &out_op, &output_tensor, 1,   // Output tensors, output tensor values, number of outputs.
                nullptr, 0,                   // Target operations, number of targets.
                nullptr,                      // Run metadata.
                g_status                      // Output status.
  );

  ros::init(argc, argv, "pedcross_tf_server");
  ros::NodeHandle n;
  ros::ServiceServer service = n.advertiseService("pedcross_tf", callback);

  ros::Time stop = ros::Time::now();
  std::cout << "PedCross TF Server started. Init time: " << stop - start << " sec" << std::endl;
  ros::spin();

  TF_DeleteSessionOptions(g_options);

  TF_CloseSession(g_sess, g_status);
  if (TF_GetCode(g_status) != TF_OK)
  {
    std::cout << "Error close session";
  }

  TF_DeleteSession(g_sess, g_status);
  if (TF_GetCode(g_status) != TF_OK)
  {
    std::cout << "Error delete session";
  }

  return 0;
}
