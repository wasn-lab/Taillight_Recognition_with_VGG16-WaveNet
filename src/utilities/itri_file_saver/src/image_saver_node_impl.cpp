#include <opencv2/opencv.hpp>
#include <thread>
#include "glog/logging.h"

#include "image_saver_node.h"
#include "image_saver_node_impl.h"
#include "image_saver_args_parser.h"

namespace image_saver
{
ImageSaverNodeImpl::ImageSaverNodeImpl() = default;
ImageSaverNodeImpl::~ImageSaverNodeImpl() = default;

void ImageSaverNodeImpl::image_callback(const sensor_msgs::ImageConstPtr& in_image_message)
{
  auto stamp = in_image_message->header.stamp;
  cv_bridge::CvImageConstPtr cv_ptr;
  try
  {
    cv_ptr = cv_bridge::toCvShare(in_image_message, sensor_msgs::image_encodings::BGR8);
  }
  catch (cv_bridge::Exception& e)
  {
    LOG(ERROR) << "cv_bridge exception: " << e.what();
    return;
  }
  std::thread saver_thread(&ImageSaverNodeImpl::save, this, cv_ptr, stamp.sec, stamp.nsec);
  saver_thread.detach();
}

void ImageSaverNodeImpl::save(const cv_bridge::CvImageConstPtr& cv_ptr, int sec, int nsec)
{
  char buff[32] = { 0 };

  snprintf(buff, sizeof(buff), "%10d%09d.jpg", sec, nsec);  // NOLINT
  std::string fname = get_output_dir() + static_cast<const char*>(buff);
  LOG(INFO) << "write " << fname;
  cv::imwrite(fname, cv_ptr->image);
}

void ImageSaverNodeImpl::subscribe()
{
  image_transport::ImageTransport it(node_handle_);
  std::string topic = image_saver::get_image_topic();
  im_subscriber_ = it.subscribe(topic, 2, &ImageSaverNodeImpl::image_callback, this);
}

void ImageSaverNodeImpl::run()
{
  subscribe();
  ros::AsyncSpinner spinner(1);  // number of threads: 1
  spinner.start();
  ros::Rate r(30);  // expected FPS
  while (ros::ok())
  {
    r.sleep();
  }
  spinner.stop();
}
};  // namespace image_saver
