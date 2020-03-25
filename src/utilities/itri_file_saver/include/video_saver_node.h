#ifndef __VIDEO_SAVER_NODE_H__
#define __VIDEO_SAVER_NODE_H__

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

#endif  // __VIDEO_SAVER_NODE_H__
