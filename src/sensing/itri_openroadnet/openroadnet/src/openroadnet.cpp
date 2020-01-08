/*
   CREATER: ICL U300
   DATE: Aug, 2019
*/
#include "openroadnet.h"
#include "DistanceEstimation.h"

using namespace npp_wrapper;

npp npp_;

openroadnet::openroadnet()
{
}

openroadnet::~openroadnet()
{
}

std::vector<msgs::PointXYZ> openroadnet::calBoundary(std::vector<uint8_t> res, std::vector<std::vector<int>>& dis_free,
                                                     int type)
{
  DistanceEstimation de;
  std::vector<msgs::PointXYZ> output_bound;

  msgs::PointXYZ Dis_Est;

  for (int a = 0; a < input_w; a = a + 10)
  {
    for (int b = 0; b < input_h; b = b + 3)
    {
      if (res[b * input_w + a] != 0)
      {
        // msgs::PointXYZ DistanceEstimation::GetPointDist(int x, int y, int cam_id)
        if (type == 1)
        {
          Dis_Est = de.GetPointDist(a, b, cam_);
        }
        else if (type == 2)
        {
          Dis_Est.x = (float)arr_x_[b * input_w + a] / 100;
          Dis_Est.y = (float)arr_y_[b * input_w + a] / 100;
          Dis_Est.z = (float)arr_z_[b * input_w + a] / 100;
        }

        output_bound.push_back(Dis_Est);
        std::vector<int> tmp_dis = { a, b };
        // Dis_Est.x = a;
        // Dis_Est.y = b;
        // output_bound.push_back(Dis_Est);
        dis_free.push_back(tmp_dis);
        break;
      }

      if (b >= input_h - 2)
      {
        if (type == 1)
        {
          Dis_Est = de.GetPointDist(a, b, cam_);
        }
        else if (type == 2)
        {
          Dis_Est.x = (float)arr_x_[b * input_w + a] / 100;
          Dis_Est.y = (float)arr_y_[b * input_w + a] / 100;
          Dis_Est.z = (float)arr_z_[b * input_w + a] / 100;
        }

        // // std::cout << Dis_Est;
        output_bound.push_back(Dis_Est);
        std::vector<int> tmp_dis = { a, b };
        // Dis_Est.x = a;
        // Dis_Est.y = b;
        // output_bound.push_back(Dis_Est);
        dis_free.push_back(tmp_dis);
        break;
      }
    }
  }

  return output_bound;
}

// int openroadnet::read_distance_from_json(const std::string& filename, int arr_x[], int arr_y[], int arr_z[], const
// int rows, const int cols)
// {
//   // dist_in_cm should be malloc by caller.
//   // assert(dist_in_cm_);
//   // for (int i = 0; i < rows; i++)
//   // {
//   //   assert(dist_in_cm_[i]);
//   // }

//   std::cout << "Loading json file for distance estimation" << std::endl;

//   std::ifstream ifs(filename);
//   Json::Reader jreader;
//   Json::Value jdata;
//   jreader.parse(ifs, jdata);

//   for (Json::ArrayIndex i = 0; i < jdata.size(); i++)
//   {
//     if(i % (jdata.size()/10) == 0)std::cout << "[LOADING]: " << std::to_string((int)floor(i/(jdata.size()/10))) << "0
//     / 100" << std::endl;
//     auto image_x = jdata[i]["im_x"].asInt();
//     auto image_y = jdata[i]["im_y"].asInt();
//     assert(image_x >= 0);
//     assert(image_y >= 0);
//     // image_x // 1920
//     // image_y // 1208
//     if ((image_y < rows) && (image_x < cols))
//     {
//       // dist_in_cm_[image_y][image_x].x = jdata[i]["dist_in_cm"][0].asInt();
//       // dist_in_cm_[image_y][image_x].y = jdata[i]["dist_in_cm"][1].asInt();
//       // dist_in_cm_[image_y][image_x].z = jdata[i]["dist_in_cm"][0].asInt();
//       arr_x[image_y*cols + image_x] = jdata[i]["dist_in_cm"][0].asInt();
//       arr_y[image_y*cols + image_x] = jdata[i]["dist_in_cm"][1].asInt();
//       arr_z[image_y*cols + image_x] = jdata[i]["dist_in_cm"][2].asInt();
//     }
//   }
//   std::cout << "[LOADING]: 100 / 100" << std::endl;
//   return 0;
// }

msgs::FreeSpaceResult openroadnet::run(cv::Mat mat60)
{
  // t1 = clock();

  cv::Size dsize = cv::Size(infer_w, infer_h);
  cv::Size Ori_size = mat60.size();
  cv::Mat resized = cv::Mat(dsize, CV_8UC3);

  npp_.resize3(mat60, resized, infer_w, infer_h);

  const int64_t input_dims[4] = { 1, resized.size[0], resized.size[1], resized.channels() };

  TF_Tensor* input_tensor = tf_utils::CreateTensor(
      TF_UINT8, input_dims, 4, resized.ptr(), resized.size[0] * resized.size[1] * resized.channels() * sizeof(uint8_t));

  TF_Tensor* output_tensor = nullptr;

  TF_SessionRun(sess,
                nullptr,                      // Run options.
                &input_op, &input_tensor, 1,  // Input tensors, input tensor values, number of inputs.
                // input_op.data(), input_tensor.data(), input_tensor.size(), // Input tensors, input tensor values,
                // number of inputs.
                &out_op, &output_tensor, 1,  // Output tensors, output tensor values, number of outputs.
                nullptr, 0,                  // Target operations, number of targets.
                nullptr,                     // Run metadata.
                status                       // Output status.
                );

  if (TF_GetCode(status) != TF_OK)
  {
    std::cout << "Error run session" << std::endl;
    // TF_DeleteStatus(status);
  }

  const auto label =
      static_cast<int*>(TF_TensorData(output_tensor));  // Return a pointer to the underlying data buffer of TF_Tensor*
  const int label_size = TF_TensorByteSize(output_tensor) / TF_DataTypeSize(TF_TensorType(output_tensor));

  uint8_t Result_mat[infer_w][infer_h];
  for (size_t a = 0; a < infer_w; a++)
  {
    for (size_t b = 0; b < infer_h; b++)
    {
      Result_mat[a][b] = label[infer_w * a + b];
    }
  }

  cv::Mat Result(infer_w, infer_h, CV_8UC1, &Result_mat);
  cv::Mat Result_resized = cv::Mat(Ori_size, CV_8UC1);

  npp_.resize1(Result, Result_resized, input_h, input_w);

  std::vector<uint8_t> array;
  if (Result_resized.isContinuous())
  {
    array.assign((uint8_t*)Result_resized.data, (uint8_t*)Result_resized.data + Result_resized.total());
  }
  else
  {
    for (int i = 0; i < Result_resized.rows; ++i)
    {
      array.insert(array.end(), Result_resized.ptr<uint8_t>(i), Result_resized.ptr<uint8_t>(i) + Result_resized.cols);
    }
  }

  dis_free_.clear();
  OpenRoadNet_output = calBoundary(array, dis_free_, 1);

  // Copy data to publish
  OpenRoadNet_Free.current.clear();
  OpenRoadNet_output_pub.freespace.clear();

  for (int len = 0; len < OpenRoadNet_output.size(); len++)
  {
    OpenRoadNet_Bound.x = OpenRoadNet_output[len].x;
    OpenRoadNet_Bound.y = OpenRoadNet_output[len].y;
    OpenRoadNet_Free.current.push_back(OpenRoadNet_Bound);
    // OpenRoadNet_output_pub.freespace.
  }

  OpenRoadNet_output_pub.freespace.push_back(OpenRoadNet_Free);

  tf_utils::DeleteTensor(input_tensor);
  tf_utils::DeleteTensor(output_tensor);

  // t2 = clock();
  // std::cout << "[INFO] [FPS]:    " << (1/((t2-t1)/(double)(CLOCKS_PER_SEC))) << std::endl;
  std::cout << "[OpenRoadNet] Running... " << std::endl;
  // std::cout << std::endl;
  // std::cout << std::endl;
  return OpenRoadNet_output_pub;
}

msgs::FreeSpaceResult openroadnet::run(const Npp8u* rawCUDA)
{
  t1 = clock();

  cv::Size dsize = cv::Size(infer_w, infer_h);
  cv::Size Ori_size = cv::Size(input_w, input_h);
  cv::Mat resized = cv::Mat(dsize, CV_8UC3);

  npp_.resize3(rawCUDA, resized, infer_w, infer_h);

  const int64_t input_dims[4] = { 1, resized.size[0], resized.size[1], resized.channels() };

  TF_Tensor* input_tensor = tf_utils::CreateTensor(
      TF_UINT8, input_dims, 4, resized.ptr(), resized.size[0] * resized.size[1] * resized.channels() * sizeof(uint8_t));

  TF_Tensor* output_tensor = nullptr;

  TF_SessionRun(sess,
                nullptr,                      // Run options.
                &input_op, &input_tensor, 1,  // Input tensors, input tensor values, number of inputs.
                // input_op.data(), input_tensor.data(), input_tensor.size(), // Input tensors, input tensor values,
                // number of inputs.
                &out_op, &output_tensor, 1,  // Output tensors, output tensor values, number of outputs.
                nullptr, 0,                  // Target operations, number of targets.
                nullptr,                     // Run metadata.
                status                       // Output status.
                );

  if (TF_GetCode(status) != TF_OK)
  {
    std::cout << "Error run session" << std::endl;
    // TF_DeleteStatus(status);
  }

  const auto label =
      static_cast<int*>(TF_TensorData(output_tensor));  // Return a pointer to the underlying data buffer of TF_Tensor*
  const int label_size = TF_TensorByteSize(output_tensor) / TF_DataTypeSize(TF_TensorType(output_tensor));

  uint8_t Result_mat[infer_w][infer_h];
  for (size_t a = 0; a < infer_w; a++)
  {
    for (size_t b = 0; b < infer_h; b++)
    {
      Result_mat[a][b] = label[infer_w * a + b];
    }
  }

  cv::Mat Result(infer_w, infer_h, CV_8UC1, &Result_mat);
  cv::Mat Result_resized = cv::Mat(Ori_size, CV_8UC1);

  npp_.resize1(Result, Result_resized, input_h, input_w);

  std::vector<uint8_t> array;
  if (Result_resized.isContinuous())
  {
    array.assign((uint8_t*)Result_resized.data, (uint8_t*)Result_resized.data + Result_resized.total());
  }
  else
  {
    for (int i = 0; i < Result_resized.rows; ++i)
    {
      array.insert(array.end(), Result_resized.ptr<uint8_t>(i), Result_resized.ptr<uint8_t>(i) + Result_resized.cols);
    }
  }

  dis_free_.clear();
  OpenRoadNet_output = calBoundary(array, dis_free_, 1);

  // Copy data to publish
  OpenRoadNet_Free.current.clear();
  OpenRoadNet_output_pub.freespace.clear();

  for (int len = 0; len < OpenRoadNet_output.size(); len++)
  {
    OpenRoadNet_Bound.x = OpenRoadNet_output[len].x;
    OpenRoadNet_Bound.y = OpenRoadNet_output[len].y;
    OpenRoadNet_Free.current.push_back(OpenRoadNet_Bound);
    // OpenRoadNet_output_pub.freespace.
  }

  OpenRoadNet_output_pub.freespace.push_back(OpenRoadNet_Free);

  tf_utils::DeleteTensor(input_tensor);
  tf_utils::DeleteTensor(output_tensor);

  t2 = clock();
  std::cout << "[INFO] [FPS]:    " << (1 / ((t2 - t1) / (double)(CLOCKS_PER_SEC))) << std::endl;
  std::cout << std::endl;
  std::cout << std::endl;
  return OpenRoadNet_output_pub;
}

void openroadnet::display_result(cv::Mat& video_ptr, std::vector<std::vector<int>> dis_free_)
{
  for (int i = 0; i < dis_free_.size(); i++)
  {
    cv::Point start = cv::Point(dis_free_[i - 1][0], dis_free_[i - 1][1]);
    cv::Point end = cv::Point(dis_free_[i][0], dis_free_[i][1]);
    if (i == 0)
      continue;
    cv::line(video_ptr, start, end, cv::Scalar(0, 0, 255), 3);
  }
}

void openroadnet::init(std::string pb_path)
{
  // Alignment
  // std::string filename = pkg_path + "/right120.json";

  // auto ret = read_distance_from_json(json_path, arr_x_, arr_y_, arr_z_, 1208, 1920);

  // =============== Tensorflow ==================
  // ======== load graph ========
  // const char* str = "./src/openroadnet/frozen_inference_graph_4k.pb";
  // std::string s = str;
  const char* pb_path_c = pb_path.c_str();
  TF_Graph* graph = tf_utils::LoadGraph(pb_path_c);
  if (graph == nullptr)
  {
    std::cout << "Can't load graph" << std::endl;
    // return 1;
  }

  // ======== get input ops ========
  input_op = { TF_GraphOperationByName(graph, "ImageTensor"), 0 };

  if (input_op.oper == nullptr)
  {
    std::cout << "Can't init input_op" << std::endl;
    // return 2;
  }

  // ======== get output ops ========
  out_op = { TF_GraphOperationByName(graph, "SemanticPredictions"), 0 };
  if (out_op.oper == nullptr)
  {
    std::cout << "Can't init out_op" << std::endl;
    // return 3;
  }

  status = TF_NewStatus();
  TF_SessionOptions* options = TF_NewSessionOptions();

  uint8_t config[16] = { 0x32, 0xe, 0x9, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0xe0, 0x3f, 0x20, 0x1, 0x2a, 0x1, 0x30 };
  TF_SetConfig(options, (void*)config, 16, status);

  sess = TF_NewSession(graph, options, status);
  TF_DeleteSessionOptions(options);

  if (TF_GetCode(status) != TF_OK)
  {
    TF_DeleteStatus(status);
    // return 4;
  }

  npp_.init_1(infer_w, infer_h, input_w, input_h);
  npp_.init_3(input_w, input_h, infer_w, infer_h);

  // return 0;
}

void openroadnet::release()
{
}
