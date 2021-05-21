#include <X11/X.h>               // for XID
#include <X11/Xlib.h>            // for Display
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
  cv::Mat capture_window();

public:
  XWinGrabber(const std::string&& xwin_title);
  XWinGrabber() = delete;
  ~XWinGrabber();
  int run();
};
};  // namespace xwin_grabber
