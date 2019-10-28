#include "tpp_node.h"

namespace tpp
{
std::mutex bbox_mutex;
std::condition_variable keep_go_cv;
bool keep_go = false;
static volatile bool keep_run = true;

void signal_handler(int sig)
{
  if (sig == SIGINT)
  {
    keep_go_cv.notify_all();
    keep_run = false;
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

void TPPNode::callback_ego_speed(const itri_msgs::VehicleState::ConstPtr& input)
{
  ego_speed_mps_.update(kmph_to_mps(input->speed));

#if DEBUG_DATA_IN
  LOG_INFO << "ego_speed = " << ego_speed_mps_.kf_.get_estimate() << " m/s" << std::endl;
#endif

  vel_.init_ego_speed(ego_speed_mps_.kf_.get_estimate());  // mps
}

void TPPNode::callback_ego_RPYrate(const sensor_msgs::Imu::ConstPtr& input)
{
  ego_rollrate_radps_.update(input->angular_velocity.x);
  ego_pitchrate_radps_.update(input->angular_velocity.y);
  ego_yawrate_radps_.update(input->angular_velocity.z);

#if DEBUG_DATA_IN
  LOG_INFO << "ego_rollrate = " << ego_rollrate_radps_.kf_.get_estimate() << " rad/s" << std::endl;
  LOG_INFO << "ego_pitchrate = " << ego_pitchrate_radps_.kf_.get_estimate() << " rad/s" << std::endl;
  LOG_INFO << "ego_yawrate = " << ego_yawrate_radps_.kf_.get_estimate() << " rad/s" << std::endl;
#endif

  vel_.init_ego_yawrate(ego_yawrate_radps_.kf_.get_estimate());  // radians / second
}

void TPPNode::callback_fusion(const msgs::DetectedObjectArray::ConstPtr& input)
{
#if DEBUG_COMPACT
  LOG_INFO << "-----------------------------------------" << std::endl;
#endif

  std::unique_lock<std::mutex> lock(bbox_mutex);

#if FPS_EXTRAPOLATION
  loop_begin = ros::Time::now().toSec();
#endif

#if DEBUG
  LOG_INFO << "callback_fusion() start" << std::endl;
#endif

#if FPS
  clock_t begin_time = clock();
#endif

  objs_header_prev_ = objs_header_;
  objs_header_ = input->header;

  double objs_header_stamp_ = objs_header_.stamp.toSec();
  double objs_header_stamp_prev_ = objs_header_prev_.stamp.toSec();

  if (objs_header_stamp_prev_ > 0 &&  //
      vel_.init_time(objs_header_stamp_, objs_header_stamp_prev_) == 0)
  {
    is_legal_dt_ = true;
  }
  else
  {
    is_legal_dt_ = false;
  }

  KTs_.header_ = objs_header_;

  if (is_legal_dt_)
  {
#if DEBUG
    LOG_INFO << "=============================================" << std::endl;
    LOG_INFO << "[Callback Sequence ID] " << objs_header_.seq << std::endl;
    LOG_INFO << "=============================================" << std::endl;
#endif

    std::vector<msgs::DetectedObject>().swap(KTs_.objs_);
    KTs_.objs_.assign(input->objects.begin(), input->objects.end());

    if (in_source_ == 1 || in_source_ == 3)
    {
      for (unsigned i = 0; i < KTs_.objs_.size(); i++)
      {
        KTs_.objs_[i].header.frame_id = "lidar";
      }
    }
    else if (in_source_ == 2)
    {
      for (unsigned i = 0; i < KTs_.objs_.size(); i++)
      {
        KTs_.objs_[i].header.frame_id = "radar";
      }
    }
    else
    {
      for (unsigned i = 0; i < KTs_.objs_.size(); i++)
      {
        KTs_.objs_[i].header.frame_id = "SensorFusion";
      }
    }

#if USE_RADAR_REL_SPEED
    for (unsigned i = 0; i < KTs_.objs_.size(); i++)
    {
      if (KTs_.objs_[i].header.frame_id == "RadarFront")
      {
        KTs_.objs_[i].relSpeed = mps_to_kmph(KTs_.objs_[i].relSpeed);
      }
    }
#endif

#if FILL_CONVEX_HULL
    for (unsigned i = 0; i < KTs_.objs_.size(); i++)
    {
      fill_convex_hull(KTs_.objs_[i].bPoint, KTs_.objs_[i].cPoint, KTs_.objs_[i].header.frame_id);
    }
#endif

#if DEBUG_DATA_IN
    for (unsigned i = 0; i < KTs_.objs_.size(); i++)
      LOG_INFO << "[Object " << i << "] p0 = (" << KTs_.objs_[i].bPoint.p0.x << ", " << KTs_.objs_[i].bPoint.p0.y
               << ", " << KTs_.objs_[i].bPoint.p0.z << ")" << std::endl;
#endif
  }
  else
  {
#if DEBUG_COMPACT
    LOG_INFO << "seq  t-1: " << objs_header_prev_.seq << std::endl;
    LOG_INFO << "seq  t  : " << objs_header_.seq << std::endl;
#endif
  }

  keep_go = true;

  keep_go_cv.notify_one();

#if FPS
  clock_t end_time = clock();
  LOG_INFO << "Running time of callback_fusion(): " << clock_to_milliseconds(end_time - begin_time) << "ms"
           << std::endl;
#endif
}

void TPPNode::subscribe_and_advertise_topics()
{
  std::string topic;

  if (in_source_ == 1)
  {
    LOG_INFO << "Input Source: Lidar" << std::endl;
    fusion_sub_ = nh_.subscribe("LidarDetection", 2, &TPPNode::callback_fusion, this);
    topic = "PathPredictionOutput/lidar";
    set_ColorRGBA(mc_.color, mc_.color_lidar_tpp);
  }
  else if (in_source_ == 2)
  {
    LOG_INFO << "Input Source: Radar" << std::endl;
    fusion_sub_ = nh_.subscribe("RadarDetection", 2, &TPPNode::callback_fusion, this);
    topic = "PathPredictionOutput/radar";
    set_ColorRGBA(mc_.color, mc_.color_radar_tpp);
  }
  else if (in_source_ == 3)
  {
    LOG_INFO << "Input Source: Camera" << std::endl;
    fusion_sub_ = nh_.subscribe("DetectedObjectArray/cam60_1", 2, &TPPNode::callback_fusion, this);
    topic = "PathPredictionOutput/camera";
    set_ColorRGBA(mc_.color, mc_.color_camera_tpp);
  }
  else
  {
    LOG_INFO << "Input Source: Fusion" << std::endl;
    fusion_sub_ = nh_.subscribe("SensorFusion", 2, &TPPNode::callback_fusion, this);
    topic = "PathPredictionOutput";
    set_ColorRGBA(mc_.color, mc_.color_fusion_tpp);
  }

  pp_pub_ = nh_.advertise<msgs::DetectedObjectArray>(topic, 2);

  if (use_ego_speed_)
  {
    ego_speed_sub_ = nh_.subscribe("vehicle_state", 2, &TPPNode::callback_ego_speed, this);
  }
  ego_RPYrate_sub_ = nh_.subscribe("imu/data", 2, &TPPNode::callback_ego_RPYrate, this);

  if (gen_markers_)
  {
    std::string topic1 = topic + "/markers";
    mc_.pub_bbox = nh_.advertise<visualization_msgs::MarkerArray>(topic1, 2);

    std::string topic2 = topic + "/polygons";
    mc_.pub_polygon = nh_.advertise<visualization_msgs::MarkerArray>(topic2, 2);

    std::string topic3 = topic + "/id";
    mc_.pub_id = nh_.advertise<visualization_msgs::MarkerArray>(topic3, 2);

    std::string topic4 = topic + "/speed";
    mc_.pub_speed = nh_.advertise<visualization_msgs::MarkerArray>(topic4, 2);

    std::string topic5 = topic + "/delay";
    mc_.pub_delay = nh_.advertise<visualization_msgs::MarkerArray>(topic5, 2);

    if (mc_.show_pp)
    {
      std::string topic6 = topic + "/pp";
      mc_.pub_pp = nh_.advertise<visualization_msgs::MarkerArray>(topic6, 2);
    }
  }
}

void TPPNode::fill_convex_hull(const msgs::BoxPoint& bPoint, msgs::ConvexPoint& cPoint, const std::string frame_id)
{
  if (cPoint.lowerAreaPoints.size() == 0 || frame_id == "RadarFront")
  {
    std::vector<Point32>().swap(cPoint.lowerAreaPoints);
    cPoint.lowerAreaPoints.reserve(4);
    cPoint.lowerAreaPoints.push_back(bPoint.p0);
    cPoint.lowerAreaPoints.push_back(bPoint.p3);
    cPoint.lowerAreaPoints.push_back(bPoint.p7);
    cPoint.lowerAreaPoints.push_back(bPoint.p4);
    cPoint.objectHigh = 4;
  }
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

float TPPNode::compute_relative_speed_obj2ego(const Vector3_32 rel_v_rel, const Point32 obj_rel)
{
  return compute_scalar_projection_A_onto_B(rel_v_rel.x, rel_v_rel.y, rel_v_rel.z, obj_rel.x, obj_rel.y, obj_rel.z);
}

float TPPNode::compute_radar_absolute_velocity(const float radar_speed_rel, const float box_center_x_abs,
                                               const float box_center_y_abs)
{
  float ego_vx = vel_.get_ego_speed() * std::cos(vel_.get_ego_heading());
  float ego_vy = vel_.get_ego_speed() * std::sin(vel_.get_ego_heading());

  float vecx_ego2obj_abs = box_center_x_abs - vel_.get_ego_x_abs();
  float vecy_ego2obj_abs = box_center_y_abs - vel_.get_ego_y_abs();

  float dist_ego2obj_tmp = euclidean_distance(vecx_ego2obj_abs, vecy_ego2obj_abs);
  float dist_ego2obj = 0;
  assign_value_cannot_zero(dist_ego2obj, dist_ego2obj_tmp);

  float mul = divide(radar_speed_rel, dist_ego2obj);

  float obj_vx_radar_rel = -vecx_ego2obj_abs * mul;
  float obj_vy_radar_rel = -vecy_ego2obj_abs * mul;

  float obj_vx_radar_abs = obj_vx_radar_rel + ego_vx;
  float obj_vy_radar_abs = obj_vy_radar_rel + ego_vy;

  float radar_speed_abs = euclidean_distance(obj_vx_radar_abs, obj_vy_radar_abs);
  return radar_speed_abs;
}

void TPPNode::compute_velocity_kalman(const float ego_dx_abs, const float ego_dy_abs)
{
  for (unsigned i = 0; i < KTs_.tracks_.size(); i++)
  {
    init_velocity(KTs_.tracks_[i].box_.track);

    if (KTs_.get_dt() <= 0)
    {
      LOG_INFO << "Warning: dt = " << KTs_.get_dt() << " ! Illegal time input !" << std::endl;
    }
    else
    {
      // absolute velocity in absolute coordinate
      float abs_vx_abs = 3.6f * KTs_.tracks_[i].kalman_.statePost.at<float>(2);  // km/h
      float abs_vy_abs = 3.6f * KTs_.tracks_[i].kalman_.statePost.at<float>(3);  // km/h

      // compute absolute velocity in relative coordinate (km/h)
      transform_vector_abs2rel(abs_vx_abs, abs_vy_abs, KTs_.tracks_[i].box_.track.absolute_velocity.x,
                               KTs_.tracks_[i].box_.track.absolute_velocity.y, vel_.get_ego_heading());
      KTs_.tracks_[i].box_.track.absolute_velocity.z = 0.f;  // km/h

      // absolute speed
      KTs_.tracks_[i].box_.track.absolute_velocity.speed = euclidean_distance(
          KTs_.tracks_[i].box_.track.absolute_velocity.x, KTs_.tracks_[i].box_.track.absolute_velocity.y);

      // relative velocity in absolute coordinate
      float rel_vx_abs = abs_vx_abs - 3.6f * (ego_dx_abs / KTs_.get_dt());  // km/h
      float rel_vy_abs = abs_vy_abs - 3.6f * (ego_dy_abs / KTs_.get_dt());  // km/h

      // compute relative velocity in relative coordinate (km/h)
      transform_vector_abs2rel(rel_vx_abs, rel_vy_abs, KTs_.tracks_[i].box_.track.relative_velocity.x,
                               KTs_.tracks_[i].box_.track.relative_velocity.y, vel_.get_ego_heading());
      KTs_.tracks_[i].box_.track.relative_velocity.z = 0.f;  // km/h

      // relative speed
      KTs_.tracks_[i].box_.track.relative_velocity.speed = euclidean_distance(
          KTs_.tracks_[i].box_.track.relative_velocity.x, KTs_.tracks_[i].box_.track.relative_velocity.y);
    }

#if DEBUG_VELOCITY
    LOG_INFO << "[Track ID] " << KTs_.tracks_[i].box_.track.id << std::endl;

    LOG_INFO << "relative_velocity on relative coord = ("               //
             << KTs_.tracks_[i].box_.track.relative_velocity.x << ", "  //
             << KTs_.tracks_[i].box_.track.relative_velocity.y << ") "  //
             << KTs_.tracks_[i].box_.track.relative_velocity.speed << " km/h" << std::endl;

    LOG_INFO << "absolute_velocity  on relative coord = ("              //
             << KTs_.tracks_[i].box_.track.absolute_velocity.x << ", "  //
             << KTs_.tracks_[i].box_.track.absolute_velocity.y << ") "  //
             << KTs_.tracks_[i].box_.track.absolute_velocity.speed << " km/h" << std::endl;
#endif

// DetectedObject.absSpeed
#if USE_RADAR_ABS_SPEED == 0
    KTs_.tracks_[i].box_.absSpeed = KTs_.tracks_[i].box_.track.absolute_velocity.speed;  // km/h
#else
    Point32 p_abs;
    box_center_.pos.get_point_abs(p_abs);
    KTs_.tracks_[i].box_.absSpeed = compute_radar_absolute_velocity(KTs_.tracks_[i].box_.relSpeed,  //
                                                                    p_abs.x, p_abs.y);
#endif

// DetectedObject.relSpeed
#if USE_RADAR_REL_SPEED
    if (KTs_.tracks_[i].box_.header.frame_id != "RadarFront")
    {
      Point32 p_rel;
      KTs_.tracks_[i].box_center_.pos.get_point_rel(p_rel);
      Vector3_32 rel_v_rel;
      rel_v_rel.x = KTs_.tracks_[i].box_.track.relative_velocity.x;
      rel_v_rel.y = KTs_.tracks_[i].box_.track.relative_velocity.y;
      rel_v_rel.z = KTs_.tracks_[i].box_.track.relative_velocity.z;
      KTs_.tracks_[i].box_.relSpeed = compute_relative_speed_obj2ego(rel_v_rel, p_rel);  // km/h
    }
#else
    Point32 p_rel;
    KTs_.tracks_[i].box_center_.pos.get_point_rel(p_rel);
    Vector3_32 rel_v_rel;

    rel_v_rel.x = KTs_.tracks_[i].box_.track.relative_velocity.x;
    rel_v_rel.y = KTs_.tracks_[i].box_.track.relative_velocity.y;
    rel_v_rel.z = KTs_.tracks_[i].box_.track.relative_velocity.z;
    KTs_.tracks_[i].box_.relSpeed = compute_relative_speed_obj2ego(rel_v_rel, p_rel);  // km/h
#endif
  }
}

void TPPNode::push_to_vector(BoxCenter a, std::vector<Point32>& b)
{
  Point32 c_rel;
  a.pos.get_point_rel(c_rel);
  b.push_back(c_rel);
}

void TPPNode::publish_tracking()
{
  std::vector<msgs::DetectedObject>().swap(pp_objs_);
  pp_objs_.reserve(KTs_.tracks_.size());

#if FPS_EXTRAPOLATION
  std::vector<Point32>().swap(box_centers_kalman_rel_);
  box_centers_kalman_rel_.reserve(KTs_.tracks_.size());

  std::vector<Point32>().swap(box_centers_kalman_next_rel_);
  box_centers_kalman_next_rel_.reserve(KTs_.tracks_.size());
#endif

  for (unsigned i = 0; i < KTs_.tracks_.size(); i++)
  {
#if REMOVE_IMPULSE_NOISE
    if (KTs_.tracks_[i].tracked_)
    {
#endif  // REMOVE_IMPULSE_NOISE
#if NOT_OUTPUT_SHORT_TERM_TRACK_LOST_BBOX
      if (KTs_.tracks_[i].lost_time_ == 0)
      {
#endif  // NOT_OUTPUT_SHORT_TERM_TRACK_LOST_BBOX

        msgs::DetectedObject box = KTs_.tracks_[i].box_;

        // init max_length, head, is_over_max_length
        box.track.max_length = 10;
        box.track.head = 255;
        box.track.is_over_max_length = false;

        box.track.id = KTs_.tracks_[i].id_;

#if FPS_EXTRAPOLATION
        box.track.tracktime = (KTs_.tracks_[i].tracktime_ - 1) * num_publishs_per_loop + 1;
#else
    	box.track.tracktime = KTs_.tracks_[i].tracktime_;
#endif

        // set max_length
        if (KTs_.tracks_[i].hist_.max_len_ > 0)
        {
          box.track.max_length = KTs_.tracks_[i].hist_.max_len_;
        }

        // set head
        if (KTs_.tracks_[i].hist_.head_ < 255)
        {
          box.track.head = KTs_.tracks_[i].hist_.head_;
        }

        // set is_over_max_length
        if (KTs_.tracks_[i].hist_.len_ >= (unsigned short)KTs_.tracks_[i].hist_.max_len_)
        {
          box.track.is_over_max_length = true;
        }

        // set states
        box.track.states.resize(box.track.max_length);

        for (unsigned k = 0; k < box.track.states.size(); k++)
        {
          box.track.states[k] = KTs_.tracks_[i].hist_.states_[k];
        }

        pp_objs_.push_back(box);

#if FPS_EXTRAPOLATION
        push_to_vector(KTs_.tracks_[i].box_center_kalman_, box_centers_kalman_rel_);
        push_to_vector(KTs_.tracks_[i].box_center_kalman_next_, box_centers_kalman_next_rel_);
#endif

#if NOT_OUTPUT_SHORT_TERM_TRACK_LOST_BBOX
      }
#endif  // NOT_OUTPUT_SHORT_TERM_TRACK_LOST_BBOX
#if REMOVE_IMPULSE_NOISE
    }
#endif  // REMOVE_IMPULSE_NOISE
  }
}

void TPPNode::publish_pp(ros::Publisher pub, std::vector<msgs::DetectedObject>& objs, const unsigned int pub_offset,
                         const float time_offset)
{
  msgs::DetectedObjectArray msg;

  msg.header = objs_header_;
  msg.header.stamp = objs_header_.stamp + ros::Duration((double)time_offset);

  msg.objects.assign(objs.begin(), objs.end());

  for (unsigned i = 0; i < msg.objects.size(); i++)
  {
    msg.objects[i].track.tracktime += pub_offset;
  }

  pub.publish(msg);

  if (gen_markers_)
  {
    mg_.marker_gen_main(msg.header, objs, mc_, ppss);
  }
}

void TPPNode::control_sleep(const double loop_interval)
{
  loop_elapsed = ros::Time::now().toSec() - loop_begin;

#if FPS
  LOG_INFO << "Sleep " << loop_interval - loop_elapsed << " seconds" << std::endl;
#endif

  if (loop_elapsed > 0)
    this_thread::sleep_for(std::chrono::milliseconds((long int)round(1000 * (loop_interval - loop_elapsed))));

  loop_begin = ros::Time::now().toSec();
}

#if FPS_EXTRAPOLATION
void TPPNode::publish_pp_extrapolation(ros::Publisher pub, std::vector<msgs::DetectedObject>& objs,
                                       std::vector<Point32> box_centers_rel, std::vector<Point32> box_centers_next_rel)
{
  float loop_interval = 1.f / output_fps;
  float time_offset = 0.f;
  unsigned int pub_offset = 0;

  publish_pp(pub, objs, pub_offset, time_offset);
  control_sleep((double)loop_interval);

  std::vector<Point32> box_pos_diffs;
  box_pos_diffs.reserve(objs.size());

  Point32 box_pos_diff;

  float scale = 0.1f * loop_interval;

  for (unsigned int i = 0; i < objs.size(); i++)
  {
    box_pos_diff.x = scale * (box_centers_next_rel[i].x - box_centers_rel[i].x);
    box_pos_diff.y = scale * (box_centers_next_rel[i].y - box_centers_rel[i].y);
    box_pos_diff.z = scale * (box_centers_next_rel[i].z - box_centers_rel[i].z);
    box_pos_diffs.push_back(box_pos_diff);
  }

  std::vector<msgs::DetectedObject> objs2;
  objs2.assign(objs.begin(), objs.end());

  for (unsigned int i = 0; i < num_publishs_per_loop - 1; i++)
  {
    for (unsigned int j = 0; j < objs2.size(); j++)
    {
      objs2[j].bPoint.p0 = add_two_Point32s(objs2[j].bPoint.p0, box_pos_diffs[j]);
      objs2[j].bPoint.p1 = add_two_Point32s(objs2[j].bPoint.p1, box_pos_diffs[j]);
      objs2[j].bPoint.p2 = add_two_Point32s(objs2[j].bPoint.p2, box_pos_diffs[j]);
      objs2[j].bPoint.p3 = add_two_Point32s(objs2[j].bPoint.p3, box_pos_diffs[j]);

      objs2[j].bPoint.p4 = add_two_Point32s(objs2[j].bPoint.p4, box_pos_diffs[j]);
      objs2[j].bPoint.p5 = add_two_Point32s(objs2[j].bPoint.p5, box_pos_diffs[j]);
      objs2[j].bPoint.p6 = add_two_Point32s(objs2[j].bPoint.p6, box_pos_diffs[j]);
      objs2[j].bPoint.p7 = add_two_Point32s(objs2[j].bPoint.p7, box_pos_diffs[j]);
    }

    time_offset += loop_interval;
    pub_offset++;

    publish_pp(pub, objs2, pub_offset, time_offset);
    control_sleep((double)loop_interval);
  }
}
#endif

void TPPNode::set_ros_params()
{
  ROSParamsParser par(nh_);

  par.get_ros_param_int("input_source", in_source_);

  //-----------------------------------------------

  par.get_ros_param_double("input_fps", input_fps);
  par.get_ros_param_double("output_fps", output_fps);
  num_publishs_per_loop = std::max((unsigned int)1, (unsigned int)std::floor(std::floor(output_fps / input_fps)));

  //-----------------------------------------------

  par.get_ros_param_double("m_lifetime_sec", mc_.lifetime_sec);

  if (mc_.lifetime_sec == 0)
  {
    mc_.lifetime_sec = 1 / output_fps;
  }

  //-----------------------------------------------

  par.get_ros_param_bool("gen_markers", gen_markers_);
  par.get_ros_param_bool("show_classid", mc_.show_classid);
  par.get_ros_param_bool("show_tracktime", mc_.show_tracktime);
  par.get_ros_param_bool("show_source", mc_.show_source);
  par.get_ros_param_bool("show_distance", mc_.show_distance);
  par.get_ros_param_bool("show_absspeed", mc_.show_distance);
  par.get_ros_param_bool("show_pp", mc_.show_pp);

  //-----------------------------------------------

  set_ColorRGBA(mc_.color_lidar_tpp, 0.f, 0.5f, 0.f, 1.f);
  set_ColorRGBA(mc_.color_radar_tpp, 0.5f, 0.f, 0.f, 1.f);
  set_ColorRGBA(mc_.color_camera_tpp, 0.5f, 0.5f, 0.5f, 1.f);
  set_ColorRGBA(mc_.color_fusion_tpp, 0.f, 1.f, 1.f, 1.f);

  par.get_ros_param_color("color_lidar_tpp", mc_.color_lidar_tpp);
  par.get_ros_param_color("color_radar_tpp", mc_.color_radar_tpp);
  par.get_ros_param_color("color_camera_tpp", mc_.color_camera_tpp);
  par.get_ros_param_color("color_fusion_tpp", mc_.color_fusion_tpp);
}

int TPPNode::run()
{
  set_ros_params();

  subscribe_and_advertise_topics();

  LOG_INFO << "ITRI_Tracking_PP is running! ver. 20190901_1730!" << std::endl;

  signal(SIGINT, signal_handler);

  ros::AsyncSpinner spinner(num_callbacks);
  spinner.start();

  ros::Rate r(output_fps);

  while (ros::ok() && !done_with_profiling() && keep_run)
  {
    std::unique_lock<std::mutex> lock(bbox_mutex);

    if (is_legal_dt_)
    {
#if DEBUG
      LOG_INFO << "Tracking main process start" << std::endl;
#endif

#if FPS
      clock_t begin_time = clock();
#endif

      // Tracking start ==========================================================================

      vel_.compute_position_displacement();
      vel_.update_localization();

      KTs_.kalman_tracker_main(vel_.get_dt(), vel_.get_ego_x_abs(), vel_.get_ego_y_abs(), vel_.get_ego_z_abs(),
                               vel_.get_ego_heading());
      compute_velocity_kalman(vel_.get_ego_dx_abs(), vel_.get_ego_dy_abs());

      publish_tracking();

#if DELAY_TIME
      mc_.module_pubtime_sec = ros::Time::now().toSec();
#endif
      // Tracking end ============================================================================
      // PP start ================================================================================

      pp_.callback_tracking(pp_objs_, vel_.get_ego_x_abs(), vel_.get_ego_y_abs(), vel_.get_ego_z_abs(),
                            vel_.get_ego_heading());
      pp_.main(pp_objs_, ppss, mc_.show_pp);

#if FPS_EXTRAPOLATION
      publish_pp_extrapolation(pp_pub_, pp_objs_, box_centers_kalman_rel_, box_centers_kalman_next_rel_);
#else
      publish_pp(pp_pub_, pp_objs_, 0, 0);
#endif

      // PP end ==================================================================================

      keep_go = false;

#if FPS
      clock_t end_time = clock();
      LOG_INFO << "Running time of Tracking PP main process: " << clock_to_milliseconds(end_time - begin_time) << "ms"
               << std::endl;
#endif
    }

    keep_go_cv.wait(lock);
    r.sleep();
  }

  spinner.stop();
  LOG_INFO << "END ITRI_Tracking_PP" << std::endl;

  return 0;
}
}  // namespace tpp