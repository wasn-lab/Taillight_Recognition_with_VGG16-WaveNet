/*
   CREATER: ICL U300
   DATE: Feb, 2019
 */

#include "parknet_node.h"
#include "parknet_node_impl.h"
ParknetNode::ParknetNode()
{
  parknet_node_impl_.reset(new ParknetNodeImpl());
}

ParknetNode::~ParknetNode() = default;

int ParknetNode::on_init()
{
  return parknet_node_impl_->on_init();
}

int ParknetNode::on_inference(std::vector<cv::Mat>& frames)
{
  return parknet_node_impl_->on_inference(frames);
}

int ParknetNode::on_inference(std::vector<Npp8u*>& npp8u_ptrs_cuda, const int num_bytes)
{
  return parknet_node_impl_->on_inference(npp8u_ptrs_cuda, num_bytes);
}

int ParknetNode::on_release()
{
  return parknet_node_impl_->on_release();
}

void ParknetNode::run(int argc, char* argv[])
{
  parknet_node_impl_->run(argc, argv);
}

void ParknetNode::subscribe_and_advertise_topics(ros::NodeHandle& node_handle)
{
  parknet_node_impl_->subscribe_and_advertise_topics(node_handle);
}
