#include "xwin_grabber_utils.h"
#include <X11/X.h>                       // for Window, Atom, XID, Success
#include <X11/Xatom.h>                   // for XA_CARDINAL, XA_STRING, XA_W...
#include <X11/Xlib.h>                    // for XImage, Display, XFree, XInt...
#include <X11/Xutil.h>                   // for XGetPixel
#include <glog/logging.h>                // for COMPACT_GOOGLE_LOG_INFO, LOG
#include <opencv2/core/hal/interface.h>  // for CV_8UC3
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <opencv2/core/mat.hpp>      // for Mat
#include <opencv2/core/mat.inl.hpp>  // for Mat::Mat, Mat::~Mat, Mat::at
#include <opencv2/core/matx.hpp>     // for Vec, Vec3b
#include <opencv2/imgproc.hpp>
#include <ostream>                   // for operator<<, basic_ostream
#include <string>                    // for string, operator<<, char_traits

namespace xwin_grabber
{
constexpr int MAX_PROPERTY_VALUE_LEN = 4096;

static cv::Mat slow_ximage_to_cvmat(XImage* image);
static cv::Mat fast_ximage_to_cvmat(XImage* image);

static Window* get_client_list(Display* disp, unsigned long* size);
static char* get_window_title(Display* disp, Window win);
static char* get_property(Display* disp, Window win, Atom xa_prop_type, const std::string& prop_name,
                          unsigned long* size);

static cv::Mat fast_ximage_to_cvmat(XImage* image)
{
  // Ximage depth = 24
  cv::Mat res(image->height, image->width, CV_8UC4);
  memcpy(res.data, image->data, image->height * image->width * 4);
  return res;
}

cv::Mat slow_ximage_to_cvmat(XImage* image)
{
  cv::Mat res(image->height, image->width, CV_8UC3);
  unsigned long red_mask = image->red_mask;
  unsigned long green_mask = image->green_mask;
  unsigned long blue_mask = image->blue_mask;
  for (int y = 0; y < image->height; y++)
  {
    for (int x = 0; x < image->width; x++)
    {
      unsigned long pixel = XGetPixel(image, x, y);
      auto& dest = res.at<cv::Vec3b>(y, x);
      dest[0] = pixel & blue_mask;
      dest[1] = (pixel & green_mask) >> 8;
      dest[2] = (pixel & red_mask) >> 16;
    }
  }
  return res;
}

cv::Mat ximage_to_cvmat(XImage* image)
{
  if (image->depth == 24 && image->bitmap_unit == 32)
  {
    return fast_ximage_to_cvmat(image);
  }
  else
  {
    return slow_ximage_to_cvmat(image);
  }
}

static Window* get_client_list(Display* disp, unsigned long* size)
{
  Window* client_list;

  if ((client_list = (Window*)get_property(disp, DefaultRootWindow(disp), XA_WINDOW, "_NET_CLIENT_LIST", size)) ==
      nullptr)
  {
    if ((client_list = (Window*)get_property(disp, DefaultRootWindow(disp), XA_CARDINAL, "_WIN_CLIENT_LIST", size)) ==
        nullptr)
    {
      fputs("Cannot get client list properties. \n"
            "(_NET_CLIENT_LIST or _WIN_CLIENT_LIST)"
            "\n",
            stderr);
      return nullptr;
    }
  }

  return client_list;
}

static char* get_window_title(Display* disp, Window win)
{
  char* title = nullptr;

  char* wm_name = get_property(disp, win, XA_STRING, "WM_NAME", nullptr);
  char* net_wm_name = get_property(disp, win, XInternAtom(disp, "UTF8_STRING", 0), "_NET_WM_NAME", nullptr);
  if (net_wm_name != nullptr)
  {
    title = strdup(net_wm_name);
  }
  else if (wm_name != nullptr)
  {
    title = strdup(wm_name);
  }

  if (net_wm_name != nullptr)
  {
    free(net_wm_name);
    net_wm_name = nullptr;
  }

  if (wm_name != nullptr)
  {
    free(wm_name);
    wm_name = nullptr;
  }

  return title;
}

static char* get_property(Display* disp, Window win, Atom xa_prop_type, const std::string& prop_name,
                          unsigned long* size)
{
  Atom xa_prop_name;
  Atom xa_ret_type;
  int ret_format;
  unsigned long ret_nitems;
  unsigned long ret_bytes_after;
  unsigned long tmp_size;
  unsigned char* ret_prop;
  char* ret;

  xa_prop_name = XInternAtom(disp, prop_name.c_str(), 0);

  /* MAX_PROPERTY_VALUE_LEN / 4 explanation (XGetWindowProperty manpage):
   *
   * long_length = Specifies the length in 32-bit multiples of the
   *               data to be retrieved.
   */
  if (XGetWindowProperty(disp, win, xa_prop_name, 0, MAX_PROPERTY_VALUE_LEN / 4, 0, xa_prop_type, &xa_ret_type,
                         &ret_format, &ret_nitems, &ret_bytes_after, &ret_prop) != Success)
  {
    LOG(INFO) << "Cannot get " << prop_name << " property.";
    return nullptr;
  }

  if (xa_ret_type != xa_prop_type)
  {
    XFree(ret_prop);
    return nullptr;
  }

  /* null terminate the result to make string handling easier */
  tmp_size = (ret_format / 8) * ret_nitems;
  // Correct 64 Architecture implementation of 32 bit data
  if (ret_format == 32)
  {
    tmp_size *= sizeof(long) / 4;
  }
  ret = (char*)malloc(tmp_size + 1);
  memcpy(ret, ret_prop, tmp_size);
  ret[tmp_size] = '\0';

  if (size)
  {
    *size = tmp_size;
  }

  XFree(ret_prop);
  return ret;
}

XID search_xid_by_title(const std::string& title)
{
  Display* display = XOpenDisplay(nullptr);
  Window* client_list;
  unsigned long client_list_size;
  XID ret = 0;
  if ((client_list = get_client_list(display, &client_list_size)) == nullptr)
  {
    LOG(INFO) << "fail to get client list";
    return 0;
  }
  for (size_t i = 0; i < client_list_size / sizeof(Window); i++)
  {
    char* title_out = get_window_title(display, client_list[i]);
    std::string whole_title{ title_out };
    if (title_out != nullptr)
    {
      free(title_out);
      title_out = nullptr;
    }

    if (whole_title.find(title) != std::string::npos)
    {
      ret = client_list[i];
    }
  }
  free(client_list);
  client_list = nullptr;
  XCloseDisplay(display);
  return ret;
}
};  // namespace xwin_grabber
