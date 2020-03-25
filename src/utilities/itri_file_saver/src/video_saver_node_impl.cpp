#include <opencv2/opencv.hpp>
#include "glog/logging.h"

#include "video_saver_node.h"
#include "video_saver_node_impl.h"
#include "video_saver_args_parser.h"

VideoSaverNodeImpl::VideoSaverNodeImpl()
#if CV_VERSION_MAJOR == 4
  : video_
{
  video_saver::get_output_filename(), cv::VideoWriter::fourcc('X', '2', '6', '4'), video_saver::get_frame_rate(),
      cv::Size(video_saver::get_frame_width(), video_saver::get_frame_height())
}
#else
  : video_
{
  video_saver::get_output_filename(), CV_FOURCC('X', '2', '6', '4'), video_saver::get_frame_rate(),
      cv::Size(video_saver::get_frame_width(), video_saver::get_frame_height())
}
#endif
, num_images_(0)
{
}

VideoSaverNodeImpl::~VideoSaverNodeImpl()
{
  std::string fname = video_saver::get_output_filename();
  LOG(INFO) << "write " << fname;
  video_.release();
}

void VideoSaverNodeImpl::image_callback(const sensor_msgs::ImageConstPtr& in_image_message)
{
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
  ++num_images_;

  if (cv_ptr->image.rows == video_saver::get_frame_height() && cv_ptr->image.cols == video_saver::get_frame_width())
  {
    video_.write(cv_ptr->image);
  }
  else
  {
    cv::Mat img;
    const int ewidth = video_saver::get_frame_width();
    const int eheight = video_saver::get_frame_height();
    LOG(INFO) << "Expect images of " << ewidth << "x" << eheight << ", Got " << cv_ptr->image.rows << "x"
              << cv_ptr->image.cols;
    cv::resize(cv_ptr->image, img, cv::Size{ ewidth, eheight });
    video_.write(img);
  }
}

void VideoSaverNodeImpl::subscribe()
{
  image_transport::ImageTransport it(node_handle_);
  std::string topic = video_saver::get_image_topic();
  im_subscriber_ = it.subscribe(topic, 2, &VideoSaverNodeImpl::image_callback, this);
  LOG(INFO) << "subscribe " << video_saver::get_image_topic();
}

void VideoSaverNodeImpl::run()
{
  subscribe();
  ros::AsyncSpinner spinner(1);  // number of threads: 1
  spinner.start();
  ros::Rate r(1000 / video_saver::get_frame_rate());
  while (ros::ok())
  {
    r.sleep();
    LOG_EVERY_N(INFO, 60) << "Got " << num_images_ << " images";
  }
  spinner.stop();
}
