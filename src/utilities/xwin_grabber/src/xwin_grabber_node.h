#pragma once
#include <ros/ros.h>
#include <string>
#include "xwin_grabber.h"

namespace xwin_grabber
{
class XWinGrabberNode
{
private:
  ros::Publisher jpg_publisher_;
  ros::Publisher raw_publisher_;
  ros::Publisher heartbeat_publisher_;
  ros::NodeHandle node_handle_;
  XWinGrabber grabber_;

  void streaming_xwin();

public:
  XWinGrabberNode(const std::string&& xwin_title);
  XWinGrabberNode() = delete;
  ~XWinGrabberNode() = default;
  int run();
};
};  // namespace xwin_grabber
