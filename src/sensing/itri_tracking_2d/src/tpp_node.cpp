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
#if DEBUG_CALLBACK
  LOG_INFO << "callback_camera() start" << std::endl;
#endif

#if DEBUG_COMPACT
  LOG_INFO << "-----------------------------------------" << std::endl;
#endif

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
#if DEBUG
    LOG_INFO << "=============================================" << std::endl;
    LOG_INFO << "[Callback Sequence ID] " << objs_header_.seq << std::endl;
    LOG_INFO << "=============================================" << std::endl;
#endif

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

    for (auto& obj : KTs_.objs_)
    {
      obj.absSpeed = 0.f;
      obj.relSpeed = 0.f;
    }

#if DEBUG_DATA_IN
    for (auto& obj : KTs_.objs_)
      LOG_INFO << "[Object " << i << "] p0 = (" << obj.bPoint.p0.x << ", " << obj.bPoint.p0.y << ", " << obj.bPoint.p0.z
               << ")" << std::endl;
#endif
  }
  else
  {
#if DEBUG_COMPACT
    LOG_INFO << "seq  t-1: " << objs_header_prev_.seq << std::endl;
    LOG_INFO << "seq  t  : " << objs_header_.seq << std::endl;
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

  nh2_.setCallbackQueue(&queue_);
}

void TPPNode::init_velocity(msgs::TrackInfo& track)
{
  track.absolute_velocity.x = 0;
  track.absolute_velocity.y = 0;
  track.absolute_velocity.z = 0;
  track.absolute_velocity.speed = 0;

  track.relative_velocity.x = 0;
  track.relative_velocity.y = 0;
  track.relative_velocity.z = 0;
  track.relative_velocity.speed = 0;
}

float TPPNode::compute_relative_speed_obj2ego(const Vector3_32 rel_v_rel, const MyPoint32 obj_rel)
{
  return compute_scalar_projection_A_onto_B(rel_v_rel.x, rel_v_rel.y, rel_v_rel.z, obj_rel.x, obj_rel.y, obj_rel.z);
}

void TPPNode::compute_velocity_kalman()
{
  for (auto& track : KTs_.tracks_)
  {
    init_velocity(track.box_.track);

    if (KTs_.get_dt() <= 0)
    {
      LOG_INFO << "Warning: dt = " << KTs_.get_dt() << " ! Illegal time input !" << std::endl;
    }
    else
    {
      // absolute velocity in absolute coordinate
      float abs_vx_abs = 3.6f * track.kalman_.statePost.at<float>(2);  // km/h
      float abs_vy_abs = 3.6f * track.kalman_.statePost.at<float>(3);  // km/h

      // compute absolute velocity in relative coordinate (km/h)
      transform_vector_abs2rel(abs_vx_abs, abs_vy_abs, track.box_.track.absolute_velocity.x,
                               track.box_.track.absolute_velocity.y, ego_heading_);
      track.box_.track.absolute_velocity.z = 0.f;  // km/h

      // absolute speed
      track.box_.track.absolute_velocity.speed =
          euclidean_distance(track.box_.track.absolute_velocity.x, track.box_.track.absolute_velocity.y);

      // relative velocity in absolute coordinate
      float rel_vx_abs = abs_vx_abs - ego_velx_abs_kmph_;  // km/h
      float rel_vy_abs = abs_vy_abs - ego_vely_abs_kmph_;  // km/h

      // compute relative velocity in relative coordinate (km/h)
      transform_vector_abs2rel(rel_vx_abs, rel_vy_abs, track.box_.track.relative_velocity.x,
                               track.box_.track.relative_velocity.y, ego_heading_);
      track.box_.track.relative_velocity.z = 0.f;  // km/h

      // relative speed
      track.box_.track.relative_velocity.speed =
          euclidean_distance(track.box_.track.relative_velocity.x, track.box_.track.relative_velocity.y);
    }

#if DEBUG_VELOCITY
    LOG_INFO << "[Track ID] " << track.box_.track.id << std::endl;

    LOG_INFO << "relative_velocity on relative coord = ("     //
             << track.box_.track.relative_velocity.x << ", "  //
             << track.box_.track.relative_velocity.y << ") "  //
             << track.box_.track.relative_velocity.speed << " km/h" << std::endl;

    LOG_INFO << "absolute_velocity  on relative coord = ("    //
             << track.box_.track.absolute_velocity.x << ", "  //
             << track.box_.track.absolute_velocity.y << ") "  //
             << track.box_.track.absolute_velocity.speed << " km/h" << std::endl;
#endif

    track.box_.absSpeed = track.box_.track.absolute_velocity.speed;  // km/h

    if (std::isnan(track.box_.absSpeed))
    {
      track.box_.absSpeed = 0.f;
    }

    MyPoint32 p_rel;
    track.box_center_.pos.get_point_rel(p_rel);  // m

    Vector3_32 rel_v_rel;
    rel_v_rel.x = track.box_.track.relative_velocity.x;  // km/h
    rel_v_rel.y = track.box_.track.relative_velocity.y;  // km/h
    rel_v_rel.z = track.box_.track.relative_velocity.z;  // km/h

    track.box_.relSpeed = compute_relative_speed_obj2ego(rel_v_rel, p_rel);  // km/h

    if (std::isnan(track.box_.relSpeed))
    {
      track.box_.relSpeed = 0.f;
    }
  }
}

void TPPNode::push_to_vector(BoxCenter a, std::vector<MyPoint32>& b)
{
  MyPoint32 c_rel;
  a.pos.get_point_rel(c_rel);
  b.push_back(c_rel);
}

void TPPNode::publish_tracking()
{
  for (const auto& track : KTs_.tracks_)
  {
#if NOT_OUTPUT_SHORT_TERM_TRACK_LOST_BBOX
      if (track.lost_time_ == 0)
      {
#endif  // NOT_OUTPUT_SHORT_TERM_TRACK_LOST_BBOX

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

#if NOT_OUTPUT_SHORT_TERM_TRACK_LOST_BBOX
      }
#endif  // NOT_OUTPUT_SHORT_TERM_TRACK_LOST_BBOX
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
#if DEBUG_CALLBACK
    LOG_INFO << "ROS loop start" << std::endl;
#endif

    if (g_trigger && is_legal_dt_)
    {
#if DEBUG
      LOG_INFO << "Tracking main process start" << std::endl;
#endif

      // MOT: SORT algorithm
      KTs_.kalman_tracker_main(dt_, ego_x_abs_, ego_y_abs_, ego_z_abs_, ego_heading_);
      compute_velocity_kalman();

      publish_tracking();

      g_trigger = false;
    }

    ros::spinOnce();  // Process callback_camera()
    loop_rate.sleep();
  }

  return 0;
}
}  // namespace tpp
