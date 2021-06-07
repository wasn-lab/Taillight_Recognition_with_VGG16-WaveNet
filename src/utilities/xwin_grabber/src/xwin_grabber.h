#pragma once
#include <X11/X.h>     // for XID
#include <X11/Xlib.h>  // for Display
#include <X11/extensions/XShm.h>
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

  XShmSegmentInfo* shm_segment_info_ = nullptr;
  bool use_shm_ = false;
  XImage* x_shm_image_ = nullptr;

  void find_xid_if_necessary();
  cv::Mat capture_window_by_xgetimage();
  cv::Mat capture_window_by_xshmgetimage();
  int init_shm();
  void release_shm();

public:
  XWinGrabber(const std::string&& xwin_title);
  XWinGrabber() = delete;
  ~XWinGrabber();
  cv::Mat capture_window();
};
};  // namespace xwin_grabber
