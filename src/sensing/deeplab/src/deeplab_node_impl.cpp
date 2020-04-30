#include "deeplab_node_impl.h"
#include "deeplab_args_parser.h"
#include <glog/logging.h>

namespace deeplab {

static bool done_with_profiling()
{
  return false;
}

DeeplabNodeImpl::DeeplabNodeImpl()
//: pb_file_(ros::package::getPath("deeplab") + "/models/frozen_inference_graph.pb")
{
}

DeeplabNodeImpl::~DeeplabNodeImpl() = default;

void DeeplabNodeImpl::subscribe_topics()
{
  image_transport::ImageTransport it(node_handle_);
  image_subscriber_ = it.subscribe(get_image_topic(), 2, &DeeplabNodeImpl::image_callback, this);
}

void DeeplabNodeImpl::advertise_topics()
{
  image_transport::ImageTransport it(node_handle_);
  image_publisher_ = it.advertise(get_image_topic() + std::string("/segment"), 1);
}

void DeeplabNodeImpl::image_callback(const sensor_msgs::ImageConstPtr& msg)
{
  LOG(INFO) << "Got image";
}

void DeeplabNodeImpl::run(int argc, char* argv[])
{
  subscribe_topics();
  advertise_topics();
  // ros::AsyncSpinner spinner(parknet::camera::num_cams_e);
  ros::AsyncSpinner spinner(1);
  spinner.start();
  ros::Rate r(30);
  while (ros::ok() && !done_with_profiling())
  {
    r.sleep();
  }
  spinner.stop();
  LOG(INFO) << "END detection";
}
}; //namespace deeplab
