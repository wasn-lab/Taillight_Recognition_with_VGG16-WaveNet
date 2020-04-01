#include "video_saver_node.h"
#include "video_saver_node_impl.h"

VideoSaverNode::VideoSaverNode()
{
  video_saver_node_impl_.reset(new VideoSaverNodeImpl());
}

VideoSaverNode::~VideoSaverNode() = default;

void VideoSaverNode::run()
{
  video_saver_node_impl_->run();
}
