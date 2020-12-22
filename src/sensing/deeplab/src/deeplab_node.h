#pragma once
#include <memory>
#include <string>
#include "ros/ros.h"
#include "opencv2/core/mat.hpp"

namespace deeplab
{
class DeeplabNodeImpl;
class DeeplabNode
{
private:
  std::unique_ptr<DeeplabNodeImpl> deeplab_node_impl_;

public:
  DeeplabNode();
  ~DeeplabNode();
  void run(int argc, char* argv[]);
};
};  // namespace deeplab
