#include "pcd_saver_node.h"
#include "pcd_saver_node_impl.h"

PCDSaverNode::PCDSaverNode()
{
  pcd_saver_node_impl_.reset(new PCDSaverNodeImpl());
}

PCDSaverNode::~PCDSaverNode() = default;

void PCDSaverNode::run()
{
  pcd_saver_node_impl_->run();
}
