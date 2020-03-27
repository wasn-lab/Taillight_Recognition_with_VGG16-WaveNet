#pragma once

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
