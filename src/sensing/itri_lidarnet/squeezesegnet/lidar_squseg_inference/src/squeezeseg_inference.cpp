#include <iostream>
#include <cmath>
#include <mutex>
#include <omp.h>

#include <ros/ros.h>
#include <ros/package.h>
#include <sensor_msgs/PointCloud2.h>
#include <std_msgs/Int8.h>
#include <std_msgs/Float32MultiArray.h>

#include <pcl/point_types.h>
#include <pcl/common/time.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>

#include <eigen3/Eigen/Dense>

#include "preprolib_squseg.h"

#include "ssn_config.h"

#include "tf_utils.hpp"

#include <boost/filesystem.hpp> 

#include "CuboidFilter.h"

namespace BFS = boost::filesystem;

ros::Publisher nn_pub;        // spherical projection
// ros::Publisher all_pub;        // publish release_cloud subscribed from lidar_preprocessing (/Lidarall/Nonground)

string data_set;
bool hybrid_detect = 0;
int pub_type = 0;
float phi_center = 0;
char ViewType = 'X';
float INPUT_MEAN[5], INPUT_STD[5];
float SPAN_PARA[2];         // {span, imagewidth}

// position of projection center
// #define x_projCenter -2
// #define z_projCenter -1.4
const float x_projCenter = proj_center(data_set,0);
const float z_projCenter = proj_center(data_set,1);

//  ======= global variabes of tensorflow =======
std::vector<TF_Output> input_ops;
TF_Output out_op;
TF_Status *status;
TF_Session *sess;

void
callback_LidarAll (const pcl::PointCloud<pcl::PointXYZI>::ConstPtr& msg)
{
  // cout << "TensorFlow Version: " << TF_Version() << endl;

  pcl::StopWatch stopWatch;

  VPointCloud::Ptr release_Cloud (new VPointCloud);

  *release_Cloud = *msg;

  if (release_Cloud->size () > 100)
  {
    //   ===============  temporally enable part of rule-based code for ensuring front-view detection ==========================
    VPointCloud::Ptr select_Cloud(new VPointCloud);
    if (hybrid_detect == 1)
    {
      *select_Cloud = CuboidFilter().pass_through_soild<PointXYZI>(release_Cloud, -10, 25, -1.2, 1.2, -3.1, 0);
    }
    //   ========================================================================================================================

    for (size_t i = 0; i < release_Cloud->points.size (); i++)
    {
      release_Cloud->points[i].x = release_Cloud->points[i].x - x_projCenter;
      // float release_Cloud->points[i].y = release_Cloud->points[i].y;
      release_Cloud->points[i].z = release_Cloud->points[i].z - z_projCenter;
    }

    VPointCloudXYZID filtered_cloud = sph_proj (release_Cloud, phi_center, SPAN_PARA[0], SPAN_PARA[1]);

    //-----------------------------------------------------------------------
    std::vector<float> spR_TFnum;
    std::vector<float> spR_TFmask;
    std::vector<float> TFprob = { 1.0 };

    int layer_TFnum = 5;
    int layer_tfmask = 1;

    spR_TFnum.resize (filtered_cloud.points.size () * layer_TFnum);
    spR_TFmask.resize (filtered_cloud.points.size () * layer_tfmask);

    for (size_t i = 0; i < filtered_cloud.points.size (); i++)
    {
      spR_TFnum[i * layer_TFnum] = (filtered_cloud.points[i].x - INPUT_MEAN[0]) / INPUT_STD[0];
      spR_TFnum[i * layer_TFnum + 1] = (filtered_cloud.points[i].y - INPUT_MEAN[1]) / INPUT_STD[1];
      spR_TFnum[i * layer_TFnum + 2] = (filtered_cloud.points[i].z - INPUT_MEAN[2]) / INPUT_STD[2];
      spR_TFnum[i * layer_TFnum + 3] = (filtered_cloud.points[i].intensity - INPUT_MEAN[3]) / INPUT_STD[3];
      spR_TFnum[i * layer_TFnum + 4] = (filtered_cloud.points[i].d - INPUT_MEAN[4]) / INPUT_STD[4];

      if (filtered_cloud.points[i].d > 0)
      {
        spR_TFmask[i] = 1.0;
      }
    }

    //-----------------------------------------------------------------------

    const std::vector<std::int64_t> num_dims = { 1, 64, static_cast<std::int64_t>(SPAN_PARA[1]), 5 };
    const std::vector<std::int64_t> mask_dims = { 1, 64, static_cast<std::int64_t>(SPAN_PARA[1]), 1 };
    const std::vector<std::int64_t> prob_dims = { 1 };

    std::vector<TF_Tensor*> input_tensors = { tf_utils::CreateTensor (TF_FLOAT, prob_dims.data (), prob_dims.size (), TFprob.data (),
                                                                      TFprob.size () * sizeof(float)), tf_utils::CreateTensor (
        TF_FLOAT, num_dims.data (), num_dims.size (), spR_TFnum.data (), spR_TFnum.size () * sizeof(float)), tf_utils::CreateTensor (
        TF_FLOAT, mask_dims.data (), mask_dims.size (), spR_TFmask.data (), spR_TFmask.size () * sizeof(float)) };

    TF_Tensor *output_tensor = nullptr;

    // cout << input_tensors.size() << endl;

    TF_SessionRun (sess, nullptr,                     // Run options.
                   input_ops.data (), input_tensors.data (), input_tensors.size (),  // Input tensors, input tensor values, number of inputs.
                   &out_op, &output_tensor, 1,  // Output tensors, output tensor values, number of outputs.
                   nullptr, 0,                  // Target operations, number of targets.
                   nullptr,                     // Run metadata.
                   status                       // Output status.
                   );

    if (TF_GetCode (status) != TF_OK)
    {
      std::cout << "Error run session";
      // TF_DeleteStatus(status);
    }

    const auto label = static_cast<long int*> (TF_TensorData (output_tensor));    // Return a pointer to the underlying data buffer of TF_Tensor*
    const int label_size = TF_TensorByteSize (output_tensor) / TF_DataTypeSize (TF_TensorType (output_tensor));

    // cout << "check length of output tensor = " << label_size << endl;
    // cout << "check data type size of output tensor = " <<  TF_DataTypeSize(TF_TensorType(output_tensor)) << endl;

    // try using pointer to init vector 
    // {data, data + (TF_TensorByteSize(tensor) / TF_DataTypeSize(TF_TensorType(tensor)))}
    // ======== Remapping ========
    VPointCloudXYZIL::Ptr result_cloud (new VPointCloudXYZIL);

    result_cloud->points.reserve (imageHeight * SPAN_PARA[1]);  // 64 * 512

    if (pub_type == 0)
    {
      for (size_t i = 0; i < (size_t)(label_size); i++)
      {
        if (label[i] > 0)
        {
          VPointXYZIL pointinfo;
          pointinfo.x = filtered_cloud.points.at (i).x;
          pointinfo.y = filtered_cloud.points.at (i).y;
          pointinfo.z = filtered_cloud.points.at (i).z;
          pointinfo.intensity = filtered_cloud.points.at (i).intensity;
          pointinfo.label = (int) label[i];

          result_cloud->points.push_back (pointinfo);
        }
      }
    }
    else
    {
      for (size_t i = 0; i < (size_t)(label_size); i++)
      {
        VPointXYZIL pointinfo;
        pointinfo.x = filtered_cloud.points.at (i).x;
        pointinfo.y = filtered_cloud.points.at (i).y;
        pointinfo.z = filtered_cloud.points.at (i).z;
        pointinfo.intensity = filtered_cloud.points.at (i).intensity;
        pointinfo.label = (int) label[i];

        result_cloud->points.push_back (pointinfo);
      }
    }


    for (size_t i = 0; i < input_tensors.size (); i++)
      tf_utils::DeleteTensor (input_tensors[i]);

    tf_utils::DeleteTensor (output_tensor);

    result_cloud->header.frame_id = msg->header.frame_id;
    result_cloud->header.stamp = msg->header.stamp;
    // pcl_conversions::toPCL(ros::Time::now(), result_cloud.header.stamp);
    // result_cloud.header.seq = msg->header.seq;

    //the origin of coordinate is shifted -2 m in x-axis and -1.4 m in z-axis in SSN
    for (size_t i = 0; i < result_cloud->points.size (); i++)
    {
      result_cloud->points[i].x = result_cloud->points[i].x + x_projCenter;
      result_cloud->points[i].z = result_cloud->points[i].z + z_projCenter;
    }

    if (hybrid_detect)
    {
      for (size_t i = 0; i < (size_t) select_Cloud->points.size(); i++)
      {
        VPointXYZIL pointinfo;
        pointinfo.x = select_Cloud->points.at(i).x;
        pointinfo.y = select_Cloud->points.at(i).y;
        pointinfo.z = select_Cloud->points.at(i).z;
        pointinfo.intensity = select_Cloud->points.at(i).intensity;
        pointinfo.label = 4;

        result_cloud->points.push_back(pointinfo);
      }
    }

    nn_pub.publish (*result_cloud);

    // ======== following comment used for debugging of subscription ========
    // sensor_msgs::PointCloud2 all_msg;
    // pcl::toROSMsg (*release_Cloud, all_msg);
    // all_msg.header.frame_id = "lidar";
    // all_msg.header.stamp = ros::Time::now ();
    // all_msg.header.seq = msg->header.seq;
    // all_pub.publish (all_msg);  // publish to /release_cloud
  }

  if (stopWatch.getTimeSeconds () > 0.05)
  {
    cout << "[SSN slow]:" << stopWatch.getTimeSeconds () << "s" << endl << endl;
  }
}

int
main (int argc,
      char **argv)
{
  string phi_center_name;

  if (argc >= 2)
  {
    data_set = string (argv[1]);
    ViewType = *argv[2];
    stringstream ss1, ss2, ss3;
    ss1 << string (argv[3]);
    ss1 >> phi_center;

    ss2 << string (argv[4]);
    ss2 >> pub_type;

    ss3 << string (argv[5]);
    ss3 >> hybrid_detect;
  }

  ros::init (argc, argv, "cpp_preprocessing");
  ros::NodeHandle nh;

  ros::Subscriber LidarAllSub = nh.subscribe ("/LidarAll/NonGround", 1, callback_LidarAll);

  norm_mean(INPUT_MEAN, data_set, ViewType, phi_center);
  norm_std(INPUT_STD, data_set, ViewType, phi_center);
  SSNspan_config(SPAN_PARA, ViewType, phi_center);

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

  nn_pub = nh.advertise<pcl::PointCloud<pcl::PointXYZIL>> ("/squ_seg/result_cloud_" + phi_center_name, 1);

  // all_pub = nh.advertise<sensor_msgs::PointCloud2> ("/release_cloud", 1);

  // =============== Tensorflow ==================
  // ======== load graph ========
  string pb_dir = (ros::package::getPath("lidar_squseg_inference") + "/model/SqueezeSegNet/" + data_set + "/" + phi_center_name + ".pb");
  
  if ( !(BFS::exists(pb_dir)) )
    pb_dir = (ros::package::getPath("lidar_squseg_inference") + "/model/SqueezeSegNet/" + "hino1" + "/" + phi_center_name + ".pb");

  TF_Graph* graph = tf_utils::LoadGraph (pb_dir.c_str());
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
    { TF_GraphOperationByName(graph, "lidar_mask"), 0}
  };

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
  if (ViewType=='T' && phi_center==0)
  {
    uint8_t config[protobuf_size] = { 0x32, 0x09, 0x09, 0x7b, 0x14, 0xae, 0x47, 0xe1, 0x7a, 0x94, 0x3f, 0x38, 0x01 };
    TF_SetConfig(options, (void *)config, protobuf_size, status);
  }
  else
  {
    uint8_t config[protobuf_size] = { 0x32, 0x09, 0x09, 0x7b, 0x14, 0xae, 0x47, 0xe1, 0x7a, 0x84, 0x3f, 0x38, 0x01 };
    TF_SetConfig(options, (void *)config, protobuf_size, status);
  }

  sess = TF_NewSession (graph, options, status);
  TF_DeleteSessionOptions (options);

  if (TF_GetCode (status) != TF_OK)
  {
    return 4;
  }

  ros::Rate loop_rate (10);

  while (ros::ok ())
  {
    ros::spinOnce ();
    loop_rate.sleep ();
  }

  TF_CloseSession (sess, status);
  TF_DeleteSession (sess, status);
  TF_DeleteStatus (status);
  TF_DeleteGraph (graph);

  return 0;
}
