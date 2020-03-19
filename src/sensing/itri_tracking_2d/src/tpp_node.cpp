#include "tpp_node.h"

namespace tpp
{
boost::shared_ptr<ros::AsyncSpinner> g_spinner;
static double output_fps = 10;  // expected publish rate

bool g_trigger = false;

void signal_handler(int sig)
{
  if (sig == SIGINT)
  {
    LOG_INFO << "END itri_tracking_2d" << std::endl;
    g_spinner->stop();
    ros::shutdown();
  }
}

static bool done_with_profiling()
{
#if ENABLE_PROFILING_MODE
  static int num_loop = 0;
  if (num_loop < 60 * output_fps)
  {
    num_loop++;
    return false;
  }
  else
  {
    return true;
  }
#else
  return false;
#endif
}

void TPPNode::callback_camera(const msgs::DetectedObjectArray::ConstPtr& input)
{
  objs_header_prev_ = objs_header_;
  objs_header_ = input->header;

  double objs_header_stamp_ = objs_header_.stamp.toSec();
  double objs_header_stamp_prev_ = objs_header_prev_.stamp.toSec();

  is_legal_dt_ =
      (objs_header_stamp_prev_ > 0 && vel_.init_time(objs_header_stamp_, objs_header_stamp_prev_) == 0) ? true : false;

  dt_ = vel_.get_dt();

  KTs_.header_ = objs_header_;

  if (is_legal_dt_)
  {
    std::vector<msgs::DetectedObject>().swap(KTs_.objs_);

#if INPUT_ALL_CLASS
    KTs_.objs_.assign(input->objects.begin(), input->objects.end());
#else
    KTs_.objs_.reserve(input->objects.size());
    for (unsigned i = 0; i < input->objects.size(); i++)
    {
      if (input->objects[i].classId >= 1 && input->objects[i].classId <= 3)
      {
        KTs_.objs_.push_back(input->objects[i]);
      }
    }
#endif
  }

  g_trigger = true;
}

void TPPNode::subscribe_and_advertise_topics()
{
  if (in_source_ == 1)
  {
    LOG_INFO << "Input Source: /CameraDetection/polygon" << std::endl;
    camera_sub_ = nh_.subscribe("/CameraDetection/polygon", 1, &TPPNode::callback_camera, this);
  }
  else
  {
    LOG_INFO << "Input Source: /CamObjFrontCenter" << std::endl;
    camera_sub_ = nh_.subscribe("/CamObjFrontCenter", 1, &TPPNode::callback_camera, this);
  }

  track2d_pub_ = nh_.advertise<msgs::DetectedObjectArray>("2DTracking", 2);
}

void TPPNode::publish()
{
  for (const auto& track : KTs_.tracks_)
  {
    msgs::DetectedObject box = track.box_;

    // init max_length, head, is_over_max_length
    box.track.max_length = 10;
    box.track.head = 255;
    box.track.is_over_max_length = false;

    box.track.id = track.id_;

    box.track.tracktime = track.tracktime_;

    // set max_length
    if (track.hist_.max_len_ > 0)
    {
      box.track.max_length = track.hist_.max_len_;
    }

    // set head
    if (track.hist_.head_ < 255)
    {
      box.track.head = track.hist_.head_;
    }

    // set is_over_max_length
    if (track.hist_.len_ >= (unsigned short)track.hist_.max_len_)
    {
      box.track.is_over_max_length = true;
    }

    // set states
    box.track.states.resize(box.track.max_length);

    for (unsigned k = 0; k < box.track.states.size(); k++)
    {
      box.track.states[k] = track.hist_.states_[k];
    }
  }
}

void TPPNode::set_ros_params()
{
  std::string domain = "/itri_tracking_2d/";
  nh_.param<int>(domain + "input_source", in_source_, 0);
  nh_.param<double>(domain + "output_fps", output_fps, 10.);
}

int TPPNode::run()
{
  set_ros_params();

  subscribe_and_advertise_topics();

  LOG_INFO << "itri_tracking_2d is running!" << std::endl;

  signal(SIGINT, signal_handler);

  // Create AsyncSpinner, run it on all available cores and make it process custom callback queue
  g_spinner.reset(new ros::AsyncSpinner(0, &queue_));
  g_spinner->start();

  g_trigger = true;

  ros::Rate loop_rate(output_fps);

  while (ros::ok() && !done_with_profiling())
  {
    if (g_trigger && is_legal_dt_)
    {
      KTs_.kalman_tracker_main(dt_); // MOT: SORT algorithm
      publish();

      g_trigger = false;
    }

    ros::spinOnce();  // Process callback_camera()
    loop_rate.sleep();
  }

  return 0;
}
}  // namespace tpp
