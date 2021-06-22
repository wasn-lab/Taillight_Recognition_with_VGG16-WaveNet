/*
 * Copyright (c) 2021, Industrial Technology and Research Institute.
 * All rights reserved.
 */
#include <string>
#include <vector>
#include <glog/logging.h>
#include <std_msgs/String.h>
#include <sensor_msgs/PointCloud.h>
#include <std_msgs/Empty.h>
#include <pcl_conversions/pcl_conversions.h>
#include "pc_transform_args_parser.h"
#include "pc_transform_utils.h"
#include "pc_transform_node.h"

namespace pc_transform
{
PCTransformNode::PCTransformNode() : pc_transform_gpu_()
{
}

void PCTransformNode::callback(sensor_msgs::PointCloud2Ptr msg)
{
  publish(transform(msg));
}

sensor_msgs::PointCloud2ConstPtr PCTransformNode::transform(const sensor_msgs::PointCloud2ConstPtr& msg)
{
  auto cloud = pc2_msg_to_xyzi(msg);
  bool ret = pc_transform_gpu_.transform(*cloud);
  CHECK(ret == true) << "Fail to transform point cloud";
  sensor_msgs::PointCloud2Ptr output_msg{ new sensor_msgs::PointCloud2 };
  pcl::toROSMsg(*cloud, *output_msg);
  output_msg->header = msg->header;
  return output_msg;
}

void PCTransformNode::publish(sensor_msgs::PointCloud2ConstPtr msg)
{
  publisher_.publish(msg);

  std_msgs::Empty empty_msg;
  heartbeat_publisher_.publish(empty_msg);

  ros::Time now = ros::Time::now();
  int32_t latency = (now.sec - msg->header.stamp.sec) * 1000 + (now.nsec - msg->header.stamp.nsec) / 1000000;
  LOG_EVERY_N(INFO, 64) << publisher_.getTopic() << " latency: " << latency << " ms.";
}

int PCTransformNode::set_subscriber()
{
  std::string topic = get_input_topic();
  if (topic.empty())
  {
    LOG(ERROR) << "Empty input topic name is not allow. Please pass it with -input_topic in the command line";
    return EXIT_FAILURE;
  }
  LOG(INFO) << ros::this_node::getName() << ":"
            << " subscribe " << topic;
  subscriber_ = node_handle_.subscribe(topic, /*queue size*/ 2, &PCTransformNode::callback, this);
  return EXIT_SUCCESS;
}

int PCTransformNode::set_publisher()
{
  std::string topic = get_output_topic();
  if (topic.empty())
  {
    LOG(ERROR) << "Empty output topic name is not allow. Please pass it with -output_topic in the command line";
    return EXIT_FAILURE;
  }
  LOG(INFO) << ros::this_node::getName() << ":"
            << " publish compressed pointcloud at topic " << topic;
  publisher_ = node_handle_.advertise<sensor_msgs::PointCloud2>(topic, /*queue size=*/2);
  heartbeat_publisher_ = node_handle_.advertise<std_msgs::Empty>(topic + "/heartbeat", /*queue size=*/2);

  return EXIT_SUCCESS;
}

int PCTransformNode::set_transform_parameters()
{
  std::string param_name = get_transform_param_name();
  // params is {tx, ty, tz, rx, ry, rz}, where t* is translation and r* is rotation
  std::vector<double> transform_params{ 0, 0, 0, 0, 0.2, 0 };

  if (ros::param::has(param_name))
  {
    node_handle_.getParam(param_name, transform_params);
  }
  else
  {
    LOG(INFO) << "Cannot find transform parameters from " << param_name
              << ". Assume this is front-top lidar and use default values.";
  }

  LOG(INFO) << "transform parameters -- tx: " << transform_params[0] << ", ty: " << transform_params[1]
            << ", tz: " << transform_params[2] << ", rx: " << transform_params[3] << ", ry: " << transform_params[4]
            << ", rz: " << transform_params[5];
  pc_transform_gpu_.set_transform_matrix(transform_params[0], transform_params[1], transform_params[2],
                                         transform_params[3], transform_params[4], transform_params[5]);

  return 0;
}

void PCTransformNode::run()
{
  set_transform_parameters();
  if ((set_subscriber() != EXIT_SUCCESS) || (set_publisher() != EXIT_SUCCESS))
  {
    return;
  }
  ros::AsyncSpinner spinner(/*thread_count*/ 1);
  spinner.start();
  ros::Rate r(1);
  while (ros::ok())
  {
    r.sleep();
  }
  spinner.stop();
}
};  // namespace pc_transform
