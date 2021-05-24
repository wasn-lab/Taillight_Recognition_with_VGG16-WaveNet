#include <X11/X.h>                      // for Pixmap, ZPixmap
#include <X11/Xlib.h>                   // for XWindowAttributes, XCloseDisplay
#include <X11/Xutil.h>                  // for BitmapSuccess, XDestroyImage
#include <X11/extensions/Xcomposite.h>  // for XCompositeNameWindowPixmap
#include <X11/extensions/composite.h>   // for CompositeRedirectAutomatic
#include <glog/logging.h>               // for COMPACT_GOOGLE_LOG_INFO, LOG
#include <sensor_msgs/CompressedImage.h>
#include <std_msgs/Empty.h>
#include <opencv2/core/mat.hpp>      // for Mat
#include <opencv2/core/mat.inl.hpp>  // for Mat::Mat, Mat::~Mat, Mat::empty
#include <opencv2/highgui.hpp>       // for destroyAllWindows, imshow
#include <string>                    // for string
#include <chrono>
#include <thread>
#include "xwin_grabber_args_parser.h"
#include "xwin_grabber_utils.h"  // for search_xid_by_title, ximage_t...
#include "xwin_grabber.h"

namespace xwin_grabber
{

bool g_xerror = false;

int xerror_handler(Display*, XErrorEvent* e)
{
  // Deal with X Error of failed request:  BadMatch (invalid parameter attributes)
  LOG(INFO) << "Error code: " << e->error_code;
  g_xerror = true;
  return 0;
}


XWinGrabber::XWinGrabber(const std::string&& xwin_title) : xwin_title_(xwin_title)
{
  display_ = XOpenDisplay(nullptr);
  xid_ = search_xid_by_title(xwin_title_);

  int event_base_return;
  int error_base_return;
  composite_enabled_ = bool(XCompositeQueryExtension(display_, &event_base_return, &error_base_return));

  XSynchronize(display_, 1);
  XSetErrorHandler(xerror_handler);

  // Set up publishers
  std::string topic = get_output_topic();
  publisher_ = node_handle_.advertise<sensor_msgs::CompressedImage>(topic, /*queue size=*/1);
  heartbeat_publisher_ = node_handle_.advertise<std_msgs::Empty>(topic + "/heartbeat", /*queue size=*/1);
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

  if ((ximage == nullptr) || g_xerror)
  {
    LOG(INFO) << "XGetImage failed";
    if (ximage != nullptr)
    {
      XDestroyImage(ximage);
      ximage = nullptr;
    }
    g_xerror = false;
    return cv::Mat{};
  }
  g_xerror = false;

  auto img = ximage_to_cvmat(ximage);
  XDestroyImage(ximage);
  return img;
}

void XWinGrabber::find_xid_if_necessary()
{
  if (xid_ > 0)
  {
    return;
  }
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

void XWinGrabber::streaming_xwin()
{
  cv::Mat img = capture_window();
  if (img.empty())
  {
    return;
  }
  std::vector<int> jpg_params{
    cv::IMWRITE_JPEG_QUALITY,
    75,
    cv::IMWRITE_JPEG_OPTIMIZE,
    1,
  };

  sensor_msgs::CompressedImagePtr msg{ new sensor_msgs::CompressedImage };
  msg->data.reserve(img.total());
  msg->format = "jpeg";
  cv::imencode(".jpg", img, msg->data, jpg_params);
  publisher_.publish(msg);
  heartbeat_publisher_.publish(std_msgs::Empty{});

  const uint64_t org_len = img.total() * img.elemSize();
  const uint64_t cmpr_len = msg->data.size();
  VLOG(2) << "Image size: " << img.cols << "x" << img.rows << ", jpg quality: " << jpg_params[1]
          << ", compression rate : " << cmpr_len << "/" << org_len << " = " << double(cmpr_len) / org_len;
}

int XWinGrabber::run()
{
  if (display_ != nullptr)
  {
    LOG(ERROR) << "Cannot open display";
    return 1;
  }

  if (!composite_enabled_)
  {
    LOG(INFO) << "Composite is NOT enabled!";
    return 1;
  }

  ros::Rate r(15);

  while (ros::ok())
  {
    find_xid_if_necessary();
    if (xid_ > 0)
    {
      streaming_xwin();
    }
    r.sleep();
  }
  return 0;
}
};  // namespace xwin_grabber
