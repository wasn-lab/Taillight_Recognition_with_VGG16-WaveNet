#include "deeplab_node.h"
#include "deeplab_node_impl.h"

namespace deeplab {
DeeplabNode::DeeplabNode():
  deeplab_node_impl_(new DeeplabNodeImpl())
{
}

DeeplabNode::~DeeplabNode() = default;

void DeeplabNode::run(int argc, char* argv[])
{
  deeplab_node_impl_->run(argc, argv);
}
}; // namespace deeplab
