/*
   CREATER: ICL U300
   DATE: Feb, 2019
 */

#include "image_saver_node.h"
#include "image_saver_node_impl.h"

ImageSaverNode::ImageSaverNode()
{
  image_saver_node_impl_.reset(new ImageSaverNodeImpl());
}

ImageSaverNode::~ImageSaverNode() = default;

void ImageSaverNode::run()
{
  image_saver_node_impl_->run();
}
