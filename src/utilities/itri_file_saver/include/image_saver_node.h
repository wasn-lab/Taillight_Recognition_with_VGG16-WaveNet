#ifndef __IMAGE_SAVER_NODE_H__
#define __IMAGE_SAVER_NODE_H__

#include <memory>

class ImageSaverNodeImpl;

class ImageSaverNode
{
private:
  std::unique_ptr<ImageSaverNodeImpl> image_saver_node_impl_;

public:
  ImageSaverNode();
  ~ImageSaverNode();
  void run();
};

#endif  // __IMAGE_SAVER_NODE_H__
