#pragma once
#include <X11/X.h>               // for XID
#include <X11/Xlib.h>            // for XImage
#include <opencv2/core/mat.hpp>  // for Mat
#include <string>                // for string

namespace xwin_grabber
{
cv::Mat ximage_to_cvmat(XImage* image);
cv::Mat capture_window(XID xid);

XID search_xid_by_title(const std::string& title);
};  // namespace xwin_grabber
