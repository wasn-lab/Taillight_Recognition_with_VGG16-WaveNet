#include <X11/X.h>                      // for Pixmap, ZPixmap
#include <X11/Xlib.h>                   // for XWindowAttributes, XCloseDisplay
#include <X11/Xutil.h>                  // for BitmapSuccess, XDestroyImage
#include <X11/extensions/Xcomposite.h>  // for XCompositeNameWindowPixmap
#include <X11/extensions/composite.h>   // for CompositeRedirectAutomatic
#include <glog/logging.h>               // for COMPACT_GOOGLE_LOG_INFO, LOG
#include <opencv2/core/mat.hpp>         // for Mat
#include <opencv2/core/mat.inl.hpp>     // for Mat::Mat, Mat::~Mat, Mat::empty
#include <opencv2/highgui.hpp>          // for destroyAllWindows, imshow
#include <string>                       // for string
#include <chrono>
#include <thread>
#include "xwin_grabber_utils.h"  // for search_xid_by_title, ximage_t...
#include "xwin_grabber.h"

namespace xwin_grabber
{
XWinGrabber::XWinGrabber(const std::string&& xwin_title) : xwin_title_(xwin_title)
{
  display_ = XOpenDisplay(nullptr);
  xid_ = search_xid_by_title(xwin_title_);

  int event_base_return;
  int error_base_return;
  composite_enabled_ = bool(XCompositeQueryExtension(display_, &event_base_return, &error_base_return));

  XSynchronize(display_, true);
}

XWinGrabber::~XWinGrabber()
{
  XCloseDisplay(display_);
}

cv::Mat XWinGrabber::capture_window()
{
  if (xid_ == 0)
  {
    return cv::Mat{};
  }

  XCompositeRedirectWindow(display_, xid_, CompositeRedirectAutomatic);
  XWindowAttributes attr;
  if (XGetWindowAttributes(display_, xid_, &attr) == 0)
  {
    LOG(INFO) << "Fail to get window attributes!";
    xid_ = 0;
    return cv::Mat{};
  }

  XImage* ximage = XGetImage(display_, xid_, 0, 0, attr.width, attr.height, AllPlanes, ZPixmap);
  if (!ximage)
  {
    LOG(INFO) << "XGetImage failed";
    return cv::Mat{};
  }

  auto img = ximage_to_cvmat(ximage);
  XDestroyImage(ximage);
  return img;
}

int XWinGrabber::run()
{
  if (!display_)
  {
    LOG(ERROR) << "Cannot open display";
    return 1;
  }

  if (!composite_enabled_)
  {
    LOG(INFO) << "Composite is NOT enabled!";
    return 1;
  }

  bool stop = false;
  constexpr int esc_key = 27;
  while (!stop)
  {
    while (xid_ == 0)
    {
      std::this_thread::sleep_for(std::chrono::seconds(1));
      xid_ = search_xid_by_title(xwin_title_);
      if (xid_ == 0)
      {
        LOG(INFO) << "xid is lost, reconnecting...";
      }
      else
      {
        LOG(INFO) << "Find xid " << xid_;
      }
    }

    cv::Mat img = capture_window();
    for (int q = 5; q <= 95; q += 5)
    {
      std::vector<int> jpg_params{
        cv::IMWRITE_JPEG_QUALITY,
        q,
        cv::IMWRITE_JPEG_OPTIMIZE,
        1,
      };
      std::vector<uint8_t> cmpr_data;
      cmpr_data.reserve(img.total());
      cv::imencode(".jpg", img, cmpr_data, jpg_params);
      const uint64_t org_len = img.total() * img.elemSize();
      const uint64_t cmpr_len = cmpr_data.size();
      LOG(INFO) << "jpg quality: " << q
                << ", compression rate : " << cmpr_len << "/" << org_len << " = " << double(cmpr_len) / org_len;
    }

    if (!img.empty())
    {
      cv::imshow("test", img);
    }
    // Set FPS = 15
    if (cv::waitKey(1000 / 15) == esc_key)
    {
      stop = true;
    }
  }
  cv::destroyAllWindows();
  return 0;
}
};  // namespace xwin_grabber
