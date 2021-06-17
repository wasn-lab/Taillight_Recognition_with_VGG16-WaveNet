#include <X11/X.h>                      // for Pixmap, ZPixmap
#include <X11/Xlib.h>                   // for XWindowAttributes, XCloseDisplay
#include <X11/Xutil.h>                  // for BitmapSuccess, XDestroyImage
#include <X11/extensions/XShm.h>
#include <X11/extensions/Xcomposite.h>  // for XCompositeNameWindowPixmap
#include <X11/extensions/composite.h>   // for CompositeRedirectAutomatic
#include <glog/logging.h>               // for COMPACT_GOOGLE_LOG_INFO, LOG
#include <opencv2/core/mat.hpp>      // for Mat
#include <opencv2/core/mat.inl.hpp>  // for Mat::Mat, Mat::~Mat, Mat::empty
#include <opencv2/highgui.hpp>       // for destroyAllWindows, imshow
#include <opencv2/imgproc.hpp>
#include <string>                    // for string
#include <chrono>
#include <thread>
#include <sys/ipc.h>
#include <sys/shm.h>

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


XWinGrabber::XWinGrabber(const std::string&& xwin_title) : xwin_title_(xwin_title), use_shm_(false)
{
  display_ = XOpenDisplay(nullptr);
  xid_ = search_xid_by_title(xwin_title_);

  int event_base_return;
  int error_base_return;
  composite_enabled_ = bool(XCompositeQueryExtension(display_, &event_base_return, &error_base_return));

  if (xid_ > 0 && composite_enabled_)
  {
    XCompositeRedirectWindow(display_, xid_, CompositeRedirectAutomatic);
  }
  int major, minor;
  Bool have_pixmaps;
  if (!XShmQueryVersion(display_, &major, &minor, &have_pixmaps))
  {
    LOG(INFO) << "Shared memory not supported.";
  }
  else
  {
    bool ret = init_shm();
    LOG_IF(INFO, ret == 0) << "Using X shared memory extension v" << major << "." << minor;
  }

  // XSynchronize(display_, 1);
  XSetErrorHandler(xerror_handler);
}

void XWinGrabber::release_shm()
{
  if (x_shm_image_)
  {
    XDestroyImage(x_shm_image_);
    x_shm_image_ = nullptr;
  }
  if (shm_segment_info_)
  {
    if (shm_segment_info_->shmaddr != nullptr)
    {
      shmdt(shm_segment_info_->shmaddr);
    }
    if (shm_segment_info_->shmid != -1)
    {
      shmctl(shm_segment_info_->shmid, IPC_RMID, 0);
    }
    delete shm_segment_info_;
    shm_segment_info_ = nullptr;
  }
}

XWinGrabber::~XWinGrabber()
{
  xid_ = 0;
  release_shm();
  XCloseDisplay(display_);
}

int XWinGrabber::init_shm()
{
  XWindowAttributes attr;
  XGetWindowAttributes(display_, xid_, &attr);

  use_shm_ = false;
  shm_segment_info_ = new XShmSegmentInfo;
  shm_segment_info_->shmid = -1;
  shm_segment_info_->shmaddr = nullptr;
  shm_segment_info_->readOnly = False;
  x_shm_image_ =
      XShmCreateImage(display_, attr.visual, attr.depth, ZPixmap, 0, shm_segment_info_, attr.width, attr.height);
  LOG(INFO)  << "Create x_shm_image_ with width " << attr.width << ", height " << attr.height << ", depth " << attr.depth;
  if (x_shm_image_)
  {
    shm_segment_info_->shmid =
        shmget(IPC_PRIVATE, x_shm_image_->bytes_per_line * x_shm_image_->height, IPC_CREAT | 0600);
    if (shm_segment_info_->shmid != -1)
    {
      void* shmat_result = shmat(shm_segment_info_->shmid, 0, 0);
      if (shmat_result != reinterpret_cast<void*>(-1))
      {
        shm_segment_info_->shmaddr = reinterpret_cast<char*>(shmat_result);
        x_shm_image_->data = shm_segment_info_->shmaddr;

        use_shm_ = XShmAttach(display_, shm_segment_info_);
        XSync(display_, False);
        if (use_shm_)
        {
          LOG(INFO) << "Using X shared memory segment " << shm_segment_info_->shmid;
        }
      }
    }
    else
    {
      LOG(INFO) << "Failed to get shared memory segment. Performance may be degraded.";
      return 1;
    }
  }
  return 0;
}

cv::Mat XWinGrabber::capture_window_by_xshmgetimage()
{
  XWindowAttributes attr;
  if (XGetWindowAttributes(display_, xid_, &attr) == 0)
  {
    LOG(INFO) << "Fail to get window attributes!";
    xid_ = 0;
    return cv::Mat{};
  }

  if (attr.width != x_shm_image_->width || attr.height != x_shm_image_->height)
  {
    LOG(INFO) << "Reconfigure x_shm_image for resizing window.";
    release_shm();
    init_shm();
  }

  auto succeed = XShmGetImage(display_, xid_, x_shm_image_, 0, 0, AllPlanes);
  if (!succeed)
  {
    LOG(INFO) << "Fail to call XShmGetImage.";
    return cv::Mat{};
  }
  return ximage_to_cvmat(x_shm_image_);
}

cv::Mat XWinGrabber::capture_window_by_xgetimage()
{
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
  cv::Mat img = ximage_to_cvmat(ximage);
  XDestroyImage(ximage);
  return img;
}

cv::Mat XWinGrabber::capture_window()
{
  auto start_time = std::chrono::system_clock::now();
  if (display_ == nullptr)
  {
    LOG(ERROR) << "Cannot open display";
    return cv::Mat{};
  }

  if (!composite_enabled_)
  {
    LOG(INFO) << "Composite is NOT enabled!";
    return cv::Mat{};
  }

  find_xid_if_necessary();
  if (xid_ == 0)
  {
    return cv::Mat{};
  }

  cv::Mat img;
  if (use_shm_)
  {
    img = capture_window_by_xshmgetimage();
  }
  else
  {
    img = capture_window_by_xgetimage();
  }
  auto end_time = std::chrono::system_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
  std::string backend = (use_shm_) ? "XShmGetImage" : "XGetImage";
  LOG_EVERY_N(INFO, 64) << __FUNCTION__ << " takes " << duration.count() << " ms. Uisng " << backend;

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

};  // namespace xwin_grabber
