#include <X11/X.h>     // for XID
#include <X11/Xlib.h>  // for Display
#include <X11/extensions/XShm.h>
#include <ros/ros.h>
#include <opencv2/core/mat.hpp>  // for Mat
#include <string>                // for string

namespace xwin_grabber
{
class XWinGrabber
{
private:
  Display* display_;
  XID xid_;
  bool composite_enabled_;
  std::string xwin_title_;
  ros::Publisher publisher_;
  ros::Publisher heartbeat_publisher_;
  ros::NodeHandle node_handle_;

  XShmSegmentInfo* shm_segment_info_ = nullptr;
  bool use_shm_ = false;
  XImage* x_shm_image_ = nullptr;

  void find_xid_if_necessary();
  void streaming_xwin();
  cv::Mat capture_window_by_xgetimage();
  cv::Mat capture_window_by_xshmgetimage();
  cv::Mat capture_window();
  int init_shm();
  void release_shm();

public:
  XWinGrabber(const std::string&& xwin_title);
  XWinGrabber() = delete;
  ~XWinGrabber();
  int run();
};
};  // namespace xwin_grabber
