/*
   CREATER: ICL U300
   DATE: Feb, 2019
 */

#ifndef __PARKNET_NODE_H__
#define __PARKNET_NODE_H__

#include <memory>
#include "ros/ros.h"
#include "opencv2/core/mat.hpp"
#include "nppdefs.h"

class ParknetNodeImpl;

class ParknetNode
{
private:
  std::unique_ptr<ParknetNodeImpl> parknet_node_impl_;

public:
  ParknetNode();
  ~ParknetNode();
  int on_init();
  int on_inference(std::vector<cv::Mat>& frames);
  int on_inference(std::vector<Npp8u *>& npp8u_ptrs_cuda, const int num_bytes);
  int on_release();
  void run(int argc, char* argv[]);
  void subscribe_and_advertise_topics(ros::NodeHandle& node_handle);
};

#endif  // __PARKNET_NODE_H__
