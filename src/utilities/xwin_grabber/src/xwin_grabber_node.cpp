#include <X11/X.h>                      // for Pixmap, ZPixmap
#include <X11/Xlib.h>                   // for XWindowAttributes, XCloseDisplay
#include <X11/Xutil.h>                  // for BitmapSuccess, XDestroyImage
#include <X11/extensions/XShm.h>
#include <X11/extensions/Xcomposite.h>  // for XCompositeNameWindowPixmap
#include <X11/extensions/composite.h>   // for CompositeRedirectAutomatic
#include <glog/logging.h>               // for COMPACT_GOOGLE_LOG_INFO, LOG
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/CompressedImage.h>
#include <sensor_msgs/Image.h>
#include <std_msgs/Empty.h>
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
#include "xwin_grabber.h"
#include "xwin_grabber_node.h"

namespace xwin_grabber
{

XWinGrabberNode::XWinGrabberNode(const std::string&& xwin_title): grabber_(std::move(xwin_title))
{
  // Set up publishers
  std::string topic = get_output_topic();
  jpg_publisher_ = node_handle_.advertise<sensor_msgs::CompressedImage>(topic + "/jpg", /*queue size=*/1);
  heartbeat_publisher_ = node_handle_.advertise<std_msgs::Empty>(topic + "/heartbeat", /*queue size=*/1);

  if (should_publish_raw_image())
  {
    raw_publisher_ = node_handle_.advertise<sensor_msgs::Image>(topic, /*queue size=*/1);
  }
}

void XWinGrabberNode::streaming_xwin()
{
  cv::Mat img = grabber_.capture_window();
  if (img.empty())
  {
    return;
  }

  int32_t org_width = img.cols;
  int32_t org_height = img.rows;
  if (img.cols > 640)
  {
    // resize image
    const double scale = 640.0 / img.cols;
    cv::Mat temp;
    cv::resize(img, temp, cv::Size(), /*width*/scale, /*height*/ scale);
    img = temp;
  }

  std::vector<int> jpg_params{
    cv::IMWRITE_JPEG_QUALITY,
    85,
    cv::IMWRITE_JPEG_OPTIMIZE,
    1,
  };

  sensor_msgs::CompressedImagePtr msg{ new sensor_msgs::CompressedImage };
  msg->data.reserve(img.total());
  msg->format = "jpeg";
  cv::imencode(".jpg", img, msg->data, jpg_params);
  jpg_publisher_.publish(msg);

  if (should_publish_raw_image())
  {
    cv::Mat temp_img;
    if (img.type() == CV_8UC4)
    {
      cv::cvtColor(img, temp_img, cv::COLOR_RGBA2RGB);
    }
    else
    {
      temp_img = img;
    }
    raw_publisher_.publish(cv_bridge::CvImage(std_msgs::Header(), "bgr8", temp_img).toImageMsg());
  }
  heartbeat_publisher_.publish(std_msgs::Empty{});

  const uint64_t org_len = img.step[0] * img.rows;
  const uint64_t cmpr_len = msg->data.size();
  LOG_EVERY_N(INFO, 64) << "Image size: " << img.cols << "x" << img.rows << ", original image size:" << org_width << "x"
                        << org_height << ", jpg quality: " << jpg_params[1] << ", compression rate : " << cmpr_len
                        << "/" << org_len << " = " << double(cmpr_len) / org_len;
  
}

int XWinGrabberNode::run()
{
  ros::Rate r(15);
  while (ros::ok())
  {
    streaming_xwin();
    r.sleep();
  }
  return 0;
}
};  // namespace xwin_grabber
