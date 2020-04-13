#pragma once
#include <memory>

class VideoSaverNodeImpl;

class VideoSaverNode
{
private:
  std::unique_ptr<VideoSaverNodeImpl> video_saver_node_impl_;

public:
  VideoSaverNode();
  ~VideoSaverNode();
  void run();
};
