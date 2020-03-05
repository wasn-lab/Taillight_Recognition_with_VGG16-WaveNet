#include "ssn_v2_config.h"

void
norm_mean (float* mean_ptr,
           const string& data_set,
           char ViewType,
           float phi_center)
{

  if (data_set.compare (0, data_set.size()-1, "hino")==0)
  {
    switch (ViewType)
    {
      case 'X':
      {
        float INPUT_MEAN[4][5] = { { 0.10, -2.95, 0.22, 3.15, 3.32 }, { 3.58, -0.07, 0.23, 1.51, 4.04 }, { 0.14, 3.29, 0.22, 2.22, 3.69 }, { -3.51, 0.07, 0.26,
            1.49, 3.92 } };

        float phi_range = 90.0;
        int phi_center_ind = int (phi_center / phi_range) + 1;

        for (size_t i = 0; i < 5; i++) {
          mean_ptr[i] = INPUT_MEAN[phi_center_ind][i];
        }

        break;
      }
      case 'T':
      {
        float INPUT_MEAN[3][5] = { { -1.63, -1.57, 0.16, 1.54, 2.52 }, { 1.91, 0.00, 0.18, 2.04, 2.97 }, { -1.67, 1.57, 0.17, 1.54, 2.55 } };

        int phi_center_ind = 0;

        switch (int (phi_center))
        {
          case -135:
          {
            phi_center_ind = 0;
            break;
          }
          case 0:
          {
            phi_center_ind = 1;
            break;
          }
          case 135:
          {
            phi_center_ind = 2;
            break;
          }
          default:
            cout << "No matched phi_center found !!!!!!!!!!" << endl;
        }

        for (size_t i = 0; i < 5; i++) {
          mean_ptr[i] = INPUT_MEAN[phi_center_ind][i];
        }

        break;
      }
      default:
        cout << "No matched ViewType found !!!!!!!!!!" << endl;
    }
  }
  else if (data_set.compare(0, data_set.size()-1, "b")==0)
  {
    float INPUT_MEAN[4][5] = {{0.04, -0.99, 0.08, 2.53, 1.12}, {1.73, 0.08, 0.08, 2.79, 1.95}, {-0.02, 1.84, 0.12, 4.64, 2.08}, {-1.79, 0.19, 0.11, 2.68, 2.00}};

    float phi_range = 90.0;
    int phi_center_ind = int(phi_center / phi_range) + 1;

    for (size_t i = 0; i < 5; i++) {
      mean_ptr[i] = INPUT_MEAN[phi_center_ind][i];
    }
  }
  else if (data_set.compare (0, data_set.size()-1, "kitti")==0)
  {
    float INPUT_MEAN[5] = { 10.88, 0.23, -1.04, 0.21, 12.12 };

    for (size_t i = 0; i < 5; i++) {
      mean_ptr[i] = INPUT_MEAN[i];
    }
  }
  else
  {
    float INPUT_MEAN[4][5] = { { 0.10, -2.95, 0.22, 3.15, 3.32 }, { 3.58, -0.07, 0.23, 1.51, 4.04 }, { 0.14, 3.29, 0.22, 2.22, 3.69 }, { -3.51, 0.07, 0.26,
        1.49, 3.92 } };

    float phi_range = 90.0;
    int phi_center_ind = int (phi_center / phi_range) + 1;

    for (size_t i = 0; i < 5; i++) {
      mean_ptr[i] = INPUT_MEAN[phi_center_ind][i];
    }
  }

}

void
norm_std (float* std_ptr,
          const string& data_set,
          char ViewType,
          float phi_center)
{

  if (data_set.compare (0, data_set.size()-1, "hino")==0)
  {
    switch (ViewType)
    {
      case 'X':
      {
        float INPUT_STD[4][5] = { { 3.35, 6.26, 0.86, 8.39, 7.01 }, { 8.04, 4.00, 1.02, 5.82, 8.86 }, { 3.60, 6.75, 0.89, 7.09, 7.53 }, { 8.78, 3.93, 0.95,
            6.15, 9.52 } };

        float phi_range = 90.0;
        int phi_center_ind = int (phi_center / phi_range) + 1;

        for (size_t i = 0; i < 5; i++) {
          std_ptr[i] = INPUT_STD[phi_center_ind][i];
        }

        break;
      }
      case 'T':
      {
        float INPUT_STD[3][5] = { { 5.46, 4.49, 0.69, 5.99, 7.12 }, { 5.74, 5.50, 0.87, 7.77, 7.71 }, { 5.43, 4.24, 0.70, 5.95, 6.90 } };

        int phi_center_ind = 0;

        switch (int (phi_center))
        {
          case -135:
          {
            phi_center_ind = 0;
            break;
          }
          case 0:
          {
            phi_center_ind = 1;
            break;
          }
          case 135:
          {
            phi_center_ind = 2;
            break;
          }
          default:
            cout << "No matched phi_center found !!!!!!!!!!" << endl;
        }

        for (size_t i = 0; i < 5; i++) {
          std_ptr[i] = INPUT_STD[phi_center_ind][i];
        }

        break;
      }
      default:
        cout << "No matched ViewType found !!!!!!!!!!" << endl;
    }
  }
  else if (data_set.compare(0, data_set.size()-1, "b")==0)
  {
    float INPUT_STD[4][5] = {{2.03, 3.86, 0.56, 10.09, 4.38}, {5.24, 2.56, 0.77, 9.56, 5.81}, {2.67, 5.05, 0.73, 13.54, 5.69}, {6.18, 2.81, 0.77, 10.18, 6.79}};

    float phi_range = 90.0;
    int phi_center_ind = int(phi_center / phi_range) + 1;

    for (size_t i = 0; i < 5; i++) {
      std_ptr[i] = INPUT_STD[phi_center_ind][i];
    }
  }
  else if (data_set.compare (0, data_set.size()-1, "kitti")==0)
  {
    float INPUT_STD[5] = { 11.47, 6.91, 0.86, 0.16, 12.32 };

    for (size_t i = 0; i < 5; i++) {
      std_ptr[i] = INPUT_STD[i];
    }
  }
  else
  {
    float INPUT_STD[4][5] = { { 3.35, 6.26, 0.86, 8.39, 7.01 }, { 8.04, 4.00, 1.02, 5.82, 8.86 }, { 3.60, 6.75, 0.89, 7.09, 7.53 }, { 8.78, 3.93, 0.95, 6.15,
        9.52 } };

    float phi_range = 90.0;
    int phi_center_ind = int (phi_center / phi_range) + 1;

    for (size_t i = 0; i < 5; i++) {
      std_ptr[i] = INPUT_STD[phi_center_ind][i];
    }
  }

}

vector<float>
phi_center_grid (char ViewType)
{
  vector<float> phi_center_all;

  switch (ViewType)
  {
    case 'X':
    {
      phi_center_all =
      { 0, 90, 180, -90};
      break;
    }
    case 'T':
    {
      phi_center_all =
      { 0, 135, -135};
      break;
    }
    default:
    {
      phi_center_all =
      { 0, 90, 180, -90};
    }
  }

  return phi_center_all;
}

TF_inference::TF_inference ()
{
}

TF_inference::TF_inference (string input_data_set,
                            char input_ViewType,
                            float input_phi_center,
                            int input_pub_type)
{
  data_set = input_data_set;
  ViewType = input_ViewType;
  phi_center = input_phi_center;
  pub_type = input_pub_type;

  if (phi_center >= 0)
  {
    stringstream ss;
    ss << phi_center;
    phi_center_name = "P" + ss.str () + "deg";
  }
  else
  {
    stringstream ss;
    ss << phi_center * -1;
    phi_center_name = "N" + ss.str () + "deg";
  }

  // =============== init parameters of SSN simultaneously ==============
  norm_mean (INPUT_MEAN, data_set, ViewType, phi_center);
  norm_std (INPUT_STD, data_set, ViewType, phi_center);
  SSNspan_config (SPAN_PARA, ViewType, phi_center);
  x_projCenter = proj_center (data_set, 0);
  z_projCenter = proj_center (data_set, 1);
  theta_UPbound = SSNtheta_config (data_set, 0);
  theta_range = SSNtheta_config (data_set, 1);
}

TF_inference::~TF_inference ()
{
}

int
TF_inference::TF_init ()
{
  // =============== Tensorflow ==================
  // ======== load graph ========
  string pb_dir = (ros::package::getPath ("lidar_squseg_inference") + "/model/SqueezeSegNet/" + data_set + "/" + phi_center_name + ".pb");
  
  if (! (BFS::exists (pb_dir)))
  {
    pb_dir = (ros::package::getPath ("lidar_squseg_inference") + "/model/SqueezeSegNet/" + "hino1" + "/" + phi_center_name + ".pb");
  }
  std::cout << "pb_dir: " <<  pb_dir << std::endl;
  //TF_status status;
  TF_Graph *graph = tf_utils::LoadGraph (pb_dir.c_str ());
  //std::cout << "Status: " << status << std::endl;
  if (graph == nullptr)
  {
    std::cout << "Can't load graph" << std::endl;
    return 1;
  }

  // ======== get input ops ========
  input_ops =
  {
    { TF_GraphOperationByName(graph, "keep_prob"), 0},
    { TF_GraphOperationByName(graph, "lidar_input"), 0},
    { TF_GraphOperationByName(graph, "lidar_mask"), 0}};

  for (size_t i = 0; i < 3; i++)
  {
    if (input_ops[i].oper == nullptr)
    {
      std::cout << "Can't init input_ops[" + to_string (i) + "]" << std::endl;
      return 2;
    }
  }

  // ======== get output ops ========
  out_op =
  { TF_GraphOperationByName(graph, "interpret_output/pred_cls"), 0};
  if (out_op.oper == nullptr)
  {
    std::cout << "Can't init out_op" << std::endl;
    return 3;
  }

  status = TF_NewStatus ();
  TF_SessionOptions *options = TF_NewSessionOptions ();

  int protobuf_size = 13;
  if (ViewType == 'T' && phi_center == 0)
  {
    auto config = std::unique_ptr<uint8_t[]>(
        new uint8_t[protobuf_size]{ 0x32, 0x09, 0x09, 0x7b, 0x14, 0xae, 0x47, 0xe1, 0x7a, 0x94, 0x3f, 0x38, 0x01 });
    TF_SetConfig (options, (void *) config.get(), protobuf_size, status);
  }
  else
  {
    auto config = std::unique_ptr<uint8_t[]>(
        new uint8_t[protobuf_size]{ 0x32, 0x09, 0x09, 0x7b, 0x14, 0xae, 0x47, 0xe1, 0x7a, 0xa4, 0x3f, 0x38, 0x01 });
    TF_SetConfig (options, (void *) config.get(), protobuf_size, status);
  }

  sess = TF_NewSession (graph, options, status);
  TF_DeleteSessionOptions (options);
  TF_DeleteGraph (graph);

  if (TF_GetCode (status) != TF_OK)
  {
    return 4;
  }
  return 0;
}

void
TF_inference::TF_run (VPointCloud::Ptr release_Cloud,
                      VPointCloudXYZIL::Ptr result_cloud)
{
  stopWatch.reset ();

  VPointCloudXYZID filtered_cloud = sph_proj (release_Cloud, phi_center, SPAN_PARA[0], SPAN_PARA[1], theta_UPbound, theta_range);

  //-----------------------------------------------------------------------
  int layer_TFnum = 6;
  //int layer_TFnum = 5;
  int layer_tfmask = 1;

  std::vector<float> spR_TFnum (filtered_cloud.points.size () * layer_TFnum);
  std::vector<float> spR_TFmask (filtered_cloud.points.size () * layer_tfmask);
  std::vector<float> TFprob = { 1.0 };

  for (size_t i = 0; i < filtered_cloud.points.size (); i++)
  {
    spR_TFnum[i * layer_TFnum] = (filtered_cloud.points[i].x - INPUT_MEAN[0]) / INPUT_STD[0];
    spR_TFnum[i * layer_TFnum + 1] = (filtered_cloud.points[i].y - INPUT_MEAN[1]) / INPUT_STD[1];
    spR_TFnum[i * layer_TFnum + 2] = (filtered_cloud.points[i].z - INPUT_MEAN[2]) / INPUT_STD[2];
    spR_TFnum[i * layer_TFnum + 3] = (filtered_cloud.points[i].intensity - INPUT_MEAN[3]) / INPUT_STD[3];
    spR_TFnum[i * layer_TFnum + 4] = (filtered_cloud.points[i].d - INPUT_MEAN[4]) / INPUT_STD[4];

    if (filtered_cloud.points[i].d > 0)
    {
      spR_TFnum[i * layer_TFnum + 5] = 1.0;
      spR_TFmask[i] = 1.0;
    }
  }

  //-----------------------------------------------------------------------

  const std::vector<std::int64_t> num_dims = { 1, 64, static_cast<std::int64_t> (SPAN_PARA[1]), 6 };
  //const std::vector<std::int64_t> num_dims = { 1, 64, static_cast<std::int64_t> (SPAN_PARA[1]), 5 };
  const std::vector<std::int64_t> mask_dims = { 1, 64, static_cast<std::int64_t> (SPAN_PARA[1]), 1 };
  const std::vector<std::int64_t> prob_dims = { 1 };

  std::vector<TF_Tensor *> input_tensors = { tf_utils::CreateTensor (TF_FLOAT, prob_dims.data (), prob_dims.size (), TFprob.data (),
                                                                     TFprob.size () * sizeof(float)), tf_utils::CreateTensor (
      TF_FLOAT, num_dims.data (), num_dims.size (), spR_TFnum.data (), spR_TFnum.size () * sizeof(float)), tf_utils::CreateTensor (
      TF_FLOAT, mask_dims.data (), mask_dims.size (), spR_TFmask.data (), spR_TFmask.size () * sizeof(float)) };

  TF_Tensor *output_tensor = nullptr;

  TF_SessionRun (sess, nullptr,                                                // Run options.
                 input_ops.data (), input_tensors.data (), input_tensors.size (),  // Input tensors, input tensor values, number of inputs.
                 &out_op, &output_tensor, 1,                                   // Output tensors, output tensor values, number of outputs.
                 nullptr, 0,                                                   // Target operations, number of targets.
                 nullptr,                                                      // Run metadata.
                 status                                                        // Output status.
                 );

  if (TF_GetCode (status) != TF_OK)
  {
    std::cout << "Error run session";
    // TF_DeleteStatus(status);
  }

  const auto label = static_cast<long int *> (TF_TensorData (output_tensor));  // Return a pointer to the underlying data buffer of TF_Tensor*
  const int label_size = TF_TensorByteSize (output_tensor) / TF_DataTypeSize (TF_TensorType (output_tensor));

  // cout << "check length of output tensor = " << label_size << endl;
  // cout << "check data type size of output tensor = " <<  TF_DataTypeSize(TF_TensorType(output_tensor)) << endl;

  // try using pointer to init vector
  // {data, data + (TF_TensorByteSize(tensor) / TF_DataTypeSize(TF_TensorType(tensor)))}
  // ======== Remapping ========
  // VPointCloudXYZIL::Ptr result_cloud(new VPointCloudXYZIL);

  result_cloud->points.reserve (imageHeight * SPAN_PARA[1]);  // 64 * 512

  if (pub_type == 0)
  {
    for (size_t i = 0; i < (size_t) (label_size); i++)
    {
      if (label[i] > 0)
      {
        VPointXYZIL pointinfo;
        pointinfo.x = filtered_cloud.points.at (i).x + x_projCenter;  // recover the origin shift in callback
        pointinfo.y = filtered_cloud.points.at (i).y;
        pointinfo.z = filtered_cloud.points.at (i).z + z_projCenter;  // recover the origin shift in callback
        pointinfo.intensity = filtered_cloud.points.at (i).intensity;
        pointinfo.label = (int) label[i];

        result_cloud->points.push_back (pointinfo);
      }
    }
  }
  else
  {
    for (size_t i = 0; i < (size_t) (label_size); i++)
    {
      VPointXYZIL pointinfo;
      pointinfo.x = filtered_cloud.points.at (i).x + x_projCenter;  // recover the origin shift in callback
      pointinfo.y = filtered_cloud.points.at (i).y;
      pointinfo.z = filtered_cloud.points.at (i).z + z_projCenter;  // recover the origin shift in callback
      pointinfo.intensity = filtered_cloud.points.at (i).intensity;
      pointinfo.label = (int) label[i];

      result_cloud->points.push_back (pointinfo);
    }
  }

  for (size_t i = 0; i < input_tensors.size (); i++)
  {
    tf_utils::DeleteTensor (input_tensors[i]);
  }

  tf_utils::DeleteTensor (output_tensor);

  if (stopWatch.getTimeSeconds () > 0.05)
  {
    cout << "[SSN] thread: " << phi_center_name << " " << stopWatch.getTimeSeconds () << endl;
  }
}

void
TF_inference::TF_quit ()
{
  TF_CloseSession (sess, status);
  TF_DeleteSession (sess, status);
  TF_DeleteStatus (status);
}
