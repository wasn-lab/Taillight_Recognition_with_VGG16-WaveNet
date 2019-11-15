#include "tpp_node.h"

namespace tpp
{
boost::shared_ptr<ros::AsyncSpinner> g_spinner;

bool g_trigger = false;

void signal_handler(int sig)
{
  if (sig == SIGINT)
  {
    LOG_INFO << "END ITRI_Tracking_PP" << std::endl;
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

void TPPNode::callback_localization(const msgs::LocalizationToVeh::ConstPtr& input)
{
#if DEBUG_CALLBACK
  LOG_INFO << "callback_localization() start" << std::endl;
#endif

  // set ego vehicle's relative pose
  ego_x_m_.update(input->x);
  ego_y_m_.update(input->y);
  ego_heading_rad_.update(input->heading * 0.01745329251f);

  vel_.set_ego_x_rel(ego_x_m_.kf_.get_estimate());
  vel_.set_ego_y_rel(ego_y_m_.kf_.get_estimate());
  vel_.set_ego_heading(ego_heading_rad_.kf_.get_estimate());

#if DEBUG_DATA_IN
  LOG_INFO << "ego_x = " << vel_.get_ego_x_rel() << "  ego_y = " << vel_.get_ego_y_rel()
           << "  ego_heading = " << vel_.get_ego_heading() << std::endl;
#endif
}

void TPPNode::callback_fusion(const msgs::DetectedObjectArray::ConstPtr& input)
{
#if DEBUG_CALLBACK
  LOG_INFO << "callback_fusion() start" << std::endl;
#endif

#if DEBUG_COMPACT
  LOG_INFO << "-----------------------------------------" << std::endl;
#endif

#if FPS_EXTRAPOLATION
  loop_begin = ros::Time::now().toSec();
#endif

#if FPS
  clock_t begin_time = clock();
#endif

  objs_header_prev_ = objs_header_;
  objs_header_ = input->header;

#if VIRTUAL_INPUT
  objs_header_.frame_id = "lidar";
  vel_.set_dt(100000000);  // 0.1s = 100,000,000ns
  is_legal_dt_ = true;
#else
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
#endif

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

    if (in_source_ == 2)
    {
      for (unsigned i = 0; i < KTs_.objs_.size(); i++)
      {
        KTs_.objs_[i].header.frame_id = "RadFront";
      }
    }
    else
    {
      for (unsigned i = 0; i < KTs_.objs_.size(); i++)
      {
        KTs_.objs_[i].header.frame_id = "lidar";
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

  g_trigger = true;

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
    fusion_sub_ = nh_.subscribe("LidarDetection", 1, &TPPNode::callback_fusion, this);
    topic = "PathPredictionOutput/lidar";
    set_ColorRGBA(mc_.color, mc_.color_lidar_tpp);
  }
  else if (in_source_ == 2)
  {
    LOG_INFO << "Input Source: Radar" << std::endl;
    fusion_sub_ = nh_.subscribe("RadarDetection", 1, &TPPNode::callback_fusion, this);
    topic = "PathPredictionOutput/radar";
    set_ColorRGBA(mc_.color, mc_.color_radar_tpp);
  }
  else if (in_source_ == 3)
  {
    LOG_INFO << "Input Source: Camera" << std::endl;
    fusion_sub_ = nh_.subscribe("DetectedObjectArray/cam60_1", 1, &TPPNode::callback_fusion, this);
    topic = "PathPredictionOutput/camera";
    set_ColorRGBA(mc_.color, mc_.color_camera_tpp);
  }
  else if (in_source_ == 4)
  {
    LOG_INFO << "Input Source: Virtual_abs" << std::endl;
    fusion_sub_ = nh_.subscribe("abs_virBB_array", 1, &TPPNode::callback_fusion, this);
    topic = "PathPredictionOutput";
    set_ColorRGBA(mc_.color, mc_.color_fusion_tpp);
  }
  else if (in_source_ == 5)
  {
    LOG_INFO << "Input Source: Virtual_rel" << std::endl;
    fusion_sub_ = nh_.subscribe("rel_virBB_array", 1, &TPPNode::callback_fusion, this);
    topic = "PathPredictionOutput";
    set_ColorRGBA(mc_.color, mc_.color_fusion_tpp);
  }
  else
  {
    LOG_INFO << "Input Source: Fusion" << std::endl;
    fusion_sub_ = nh_.subscribe("SensorFusion", 1, &TPPNode::callback_fusion, this);
    topic = "PathPredictionOutput";
    set_ColorRGBA(mc_.color, mc_.color_fusion_tpp);
  }

  pp_pub_ = nh_.advertise<msgs::DetectedObjectArray>(topic, 2);

  nh2_.setCallbackQueue(&queue_);

  // Note that we use different NodeHandle here
  localization_sub_ = nh2_.subscribe("localization_to_veh", 2, &TPPNode::callback_localization, this);

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

    if (mc_.show_pp > 0)
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
      float scale = 3.6f / KTs_.get_dt();
      float rel_vx_abs = abs_vx_abs - scale * ego_dx_abs;  // km/h
      float rel_vy_abs = abs_vy_abs - scale * ego_dy_abs;  // km/h

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

void TPPNode::save_output_to_txt(const std::vector<msgs::DetectedObject>& objs)
{
  std::ofstream ofs;
  std::stringstream ss;
  ss << "../../../tracking_output.txt";
  std::string fname = ss.str();

  if (objs.empty())
  {
    std::cout << "objs is empty. No output to txt." << std::endl;
    return;
  }

  ofs.open(fname, std::ios_base::app);

  for (size_t i = 0; i < objs.size(); i++)
  {
    if (objs[i].track.is_ready_prediction)
    {
      ofs << std::fixed                                               //
          << objs[i].header.stamp << ", "                             // #1 time stamp
          << objs[i].track.id << ", "                                 // #2 track id
          << objs[i].lidarInfo.boxCenter.x << ", "                    // #3 original bbox center x
          << objs[i].lidarInfo.boxCenter.y << ", "                    // #4 original bbox center y
          << (objs[i].bPoint.p0.x + objs[i].bPoint.p6.x) / 2 << ", "  // #5 modified bbox center x
          << (objs[i].bPoint.p0.y + objs[i].bPoint.p6.y) / 2 << ", "  // #6 modified bbox center y
          << objs[i].track.absolute_velocity.x << ", "                // #7 abs vx
          << objs[i].track.absolute_velocity.y << ", "                // #8 abs vy
          << objs[i].absSpeed << ", "                                 // #9 abs speed
          << objs[i].track.relative_velocity.x << ", "                // #10 rel vx
          << objs[i].track.relative_velocity.y << ", "                // #11 rel vy
          << objs[i].relSpeed;                                        // #12 rel speed

      // #13 ppx in 5 ticks
      // #14 ppy in 5 ticks
      // #15 ppx in 10 ticks
      // #16 ppy in 10 ticks
      // #17 ppx in 15 ticks
      // #18 ppy in 15 ticks
      // #19 ppx in 20 ticks
      // #20 ppy in 20 ticks
      for (size_t j = 0; j < objs[i].track.forecasts.size(); j = j + 5)
      {
        ofs << ", " << objs[i].track.forecasts[j].position.x << ", " << objs[i].track.forecasts[j].position.y;
      }
      ofs << ", "                          //
          << vel_.get_ego_x_abs() << ", "  // #21 kf ego x abs
          << vel_.get_ego_x_rel() << ", "  // #22 kf ego x rel
          << vel_.get_ego_y_abs() << ", "  // #23 kf ego y abs
          << vel_.get_ego_y_rel() << ", "  // #24 kf ego y rel
          << vel_.get_ego_z_abs() << ", "  // #25 kf ego z abs
          << vel_.get_ego_z_rel() << ", "  // #26 kf ego z rel
          << vel_.get_ego_heading();       // #27 kf ego heading
    }
    ofs << "\n";
    std::cout << "[Produced] time = " << objs[i].header.stamp << ", track_id = " << objs[i].track.id << std::endl;
  }

  ofs.close();
}

void TPPNode::publish_pp(ros::Publisher pub, std::vector<msgs::DetectedObject>& objs, const unsigned int pub_offset,
                         const float time_offset)
{
#if SAVE_OUTPUT_TXT
  save_output_to_txt(objs);
#endif

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

  double pp_input_shift_m = 0.;
  par.get_ros_param_double("pp_input_shift_m", pp_input_shift_m);
  pp_.set_input_shift_m((long double)pp_input_shift_m);

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
  par.get_ros_param_uint("show_pp", mc_.show_pp);

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

  LOG_INFO << "ITRI_Tracking_PP is running! ver. 20191111_1500!" << std::endl;

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

#if FPS
      clock_t begin_time = clock();
#endif

      // Tracking start ==========================================================================

      vel_.compute_ego_position_absolute();

      dt_ = vel_.get_dt();
      ego_x_abs_ = vel_.get_ego_x_abs();
      ego_y_abs_ = vel_.get_ego_y_abs();
      ego_z_abs_ = vel_.get_ego_z_abs();
      ego_heading_ = vel_.get_ego_heading();
      ego_dx_abs_ = vel_.get_ego_dx_abs();
      ego_dy_abs_ = vel_.get_ego_dy_abs();

      // std::cout << dt_ << " " << ego_x_abs_ << " " << ego_y_abs_ << " " << ego_z_abs_ << " " <<  ego_heading_ << " "
      // << ego_dx_abs_ << " " << ego_dy_abs_ << std::endl;

      KTs_.kalman_tracker_main(dt_, ego_x_abs_, ego_y_abs_, ego_z_abs_, ego_heading_);
      compute_velocity_kalman(ego_dx_abs_, ego_dy_abs_);

      publish_tracking();

#if DELAY_TIME
      mc_.module_pubtime_sec = ros::Time::now().toSec();
#endif

      // Tracking end && PP start ================================================================

      pp_.callback_tracking(pp_objs_, vel_.get_ego_x_abs(), vel_.get_ego_y_abs(), vel_.get_ego_z_abs(),
                            vel_.get_ego_heading());
      pp_.main(pp_objs_, ppss, mc_.show_pp);

#if FPS_EXTRAPOLATION
      publish_pp_extrapolation(pp_pub_, pp_objs_, box_centers_kalman_rel_, box_centers_kalman_next_rel_);
#else
      publish_pp(pp_pub_, pp_objs_, 0, 0);
#endif

      // PP end ==================================================================================

#if FPS
      clock_t end_time = clock();
      LOG_INFO << "Running time of Tracking PP main process: " << clock_to_milliseconds(end_time - begin_time) << "ms"
               << std::endl;
#endif

      g_trigger = false;
    }

    // Process messages on callback_fusion()
    ros::spinOnce();
    loop_rate.sleep();
  }

  return 0;
}
}  // namespace tpp