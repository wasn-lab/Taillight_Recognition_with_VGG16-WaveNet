#include "tpp_node.h"

namespace tpp
{
boost::shared_ptr<ros::AsyncSpinner> g_spinner;
static double input_fps = 5;    // known callback rate
static double output_fps = 10;  // expected publish rate

static unsigned int num_publishs_per_loop =
    std::max((unsigned int)1, (unsigned int)std::floor(std::floor(output_fps / input_fps)));

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

#if TTC_TEST
void TPPNode::callback_seq(const std_msgs::Int32::ConstPtr& input)
{
  seq_cb_ = input->data;

#if DEBUG_DATA_IN
  std::cout << "seq_cb_ = " << seq_cb_ << std::endl;
#endif
}

void TPPNode::callback_ego_speed_kmph(const std_msgs::Float64::ConstPtr& input)
{
  vel_.set_ego_speed_kmph(input->data);

#if DEBUG_DATA_IN
  LOG_INFO << "ego_speed_kmph = " << vel_.get_ego_speed_kmph() << std::endl;
#endif
}

void TPPNode::callback_localization(const visualization_msgs::Marker::ConstPtr& input)
{
#if DEBUG_CALLBACK
  LOG_INFO << "callback_localization() start" << std::endl;
#endif

  vel_.set_ego_x_abs(input->pose.position.x);
  vel_.set_ego_y_abs(input->pose.position.y);

  double roll, pitch, yaw;
  quaternion_to_rpy(roll, pitch, yaw, input->pose.orientation.x, input->pose.orientation.y, input->pose.orientation.z,
                    input->pose.orientation.w) vel_.set_ego_heading(yaw);

#if DEBUG_DATA_IN
  LOG_INFO << "ego_x = " << vel_.get_ego_x_abs() << "  ego_y = " << vel_.get_ego_y_abs()
           << "  ego_heading = " << vel_.get_ego_heading() << std::endl;
#endif
}
#else  // TTC_TEST == 0
void TPPNode::callback_ego_speed_kmph(const msgs::VehInfo::ConstPtr& input)
{
  vel_.set_ego_speed_kmph(input->ego_speed * 3.6);

#if DEBUG_DATA_IN
  LOG_INFO << "ego_speed_kmph = " << vel_.get_ego_speed_kmph() << std::endl;
#endif
}

void TPPNode::callback_localization(const msgs::LocalizationToVeh::ConstPtr& input)
{
  // #if DEBUG_CALLBACK
  //   LOG_INFO << "callback_localization() start" << std::endl;
  // #endif

  //   vel_.set_ego_x_abs(input->x);
  //   vel_.set_ego_y_abs(input->y);
  //   vel_.set_ego_heading(input->heading * 0.01745329251f);

  // #if DEBUG_DATA_IN
  //   LOG_INFO << "ego_x = " << vel_.get_ego_x_abs() << "  ego_y = " << vel_.get_ego_y_abs()
  //            << "  ego_heading = " << vel_.get_ego_heading() << std::endl;
  // #endif
}
#endif

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
#endif

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
    KTs_.objs_.assign(input->objects.begin(), input->objects.end());

#if VIRTUAL_INPUT
    for (unsigned i = 0; i < KTs_.objs_.size(); i++)
    {
      gt_x_ = KTs_.objs_[i].radarInfo.imgPoint60.x;
      gt_y_ = KTs_.objs_[i].radarInfo.imgPoint60.y;
    }
#endif

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
    fusion_sub_ = nh_.subscribe("CamObjFrontCenter", 1, &TPPNode::callback_fusion, this);
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
#if TTC_TEST
  seq_sub_ = nh2_.subscribe("sequence_ID", 1, &TPPNode::callback_seq, this);
  ego_speed_kmph_sub_ = nh2_.subscribe("player_vehicle_speed", 1, &TPPNode::callback_ego_speed_kmph, this);
  localization_sub_ = nh2_.subscribe("player_vehicle", 1, &TPPNode::callback_localization, this);
#else
  ego_speed_kmph_sub_ = nh2_.subscribe("veh_info", 1, &TPPNode::callback_ego_speed_kmph, this);
  localization_sub_ = nh2_.subscribe("localization_to_veh", 1, &TPPNode::callback_localization, this);
#endif

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

    std::string topic7 = topic + "/vel";
    mc_.pub_vel = nh_.advertise<visualization_msgs::MarkerArray>(topic7, 2);
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
  float ego_vx = ego_speed_kmph_ * std::cos(ego_heading_);
  float ego_vy = ego_speed_kmph_ * std::sin(ego_heading_);

  float vecx_ego2obj_abs = box_center_x_abs - ego_x_abs_;
  float vecy_ego2obj_abs = box_center_y_abs - ego_y_abs_;

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

void TPPNode::compute_velocity_kalman()
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
                               KTs_.tracks_[i].box_.track.absolute_velocity.y, ego_heading_);
      KTs_.tracks_[i].box_.track.absolute_velocity.z = 0.f;  // km/h

      // absolute speed
      KTs_.tracks_[i].box_.track.absolute_velocity.speed = euclidean_distance(
          KTs_.tracks_[i].box_.track.absolute_velocity.x, KTs_.tracks_[i].box_.track.absolute_velocity.y);

      // relative velocity in absolute coordinate
      float rel_vx_abs = abs_vx_abs - ego_velx_abs_kmph_;  // km/h
      float rel_vy_abs = abs_vy_abs - ego_vely_abs_kmph_;  // km/h

      // compute relative velocity in relative coordinate (km/h)
      transform_vector_abs2rel(rel_vx_abs, rel_vy_abs, KTs_.tracks_[i].box_.track.relative_velocity.x,
                               KTs_.tracks_[i].box_.track.relative_velocity.y, ego_heading_);
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
    KTs_.tracks_[i].box_center_.pos.get_point_rel(p_rel);  // m

    Vector3_32 rel_v_rel;
    rel_v_rel.x = KTs_.tracks_[i].box_.track.relative_velocity.x;  // km/h
    rel_v_rel.y = KTs_.tracks_[i].box_.track.relative_velocity.y;  // km/h
    rel_v_rel.z = KTs_.tracks_[i].box_.track.relative_velocity.z;  // km/h

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

inline bool test_file_exist(const std::string& name)
{
  ifstream f(name.c_str());
  return f.good();
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

  if (!test_file_exist(fname))
  {
    ofs.open(fname, std::ios_base::app);

    ofs << "#1 time stamp (s), "  //
        << "#2 track id, "        //
        << "#3 dt (s), "          //
#if VIRTUAL_INPUT
        << "#4-1 GT bbox center x (m), "  //
        << "#4-2 GT bbox center y (m), "  //
#endif
        << "#5-1 input bbox center x (m), "            //
        << "#5-2 input bbox center y (m), "            //
        << "#6-1 kalman-filtered bbox center x (m), "  //
        << "#6-2 kalman-filtered bbox center y (m), "  //
        << "#7 abs vx (km/h), "                        //
        << "#8 abs vy (km/h), "                        //
        << "#9 abs speed (km/h), "                     //
        << "#10 rel vx (km/h), "                       //
        << "#11 rel vy (km/h), "                       //
        << "#12 rel speed (km/h), "                    //
        << "#13 ppx in 5 ticks (m), "                  //
        << "#14 ppy in 5 ticks (m), "                  //
        << "#15 ppx in 10 ticks (m), "                 //
        << "#16 ppy in 10 ticks (m), "                 //
        << "#17 ppx in 15 ticks (m), "                 //
        << "#18 ppy in 15 ticks (m), "                 //
        << "#19 ppx in 20 ticks (m), "                 //
        << "#20 ppy in 20 ticks (m), "                 //
        << "#21 ego x abs (m), "                       //
        << "#22 ego y abs (m), "                       //
        << "#23 ego z abs (m), "                       //
        << "#24 ego heading (rad), "                   //
        << "#25 kf Q1, "                               //
        << "#26 kf Q2, "                               //
        << "#27 kf Q3, "                               //
        << "#28 kf R, "                                //
        << "#29 kf P0\n";
  }
  else
  {
    ofs.open(fname, std::ios_base::app);
  }

  ros::Duration dt_s(0, dt_);

  for (size_t i = 0; i < objs.size(); i++)
  {
    ofs << std::fixed                          //
        << objs_header_.stamp.toSec() << ", "  // #1 time stamp (s)
        << objs[i].track.id << ", "            // #2 track id
        << dt_s.toSec() << ", "                // #3 dt (s)
#if VIRTUAL_INPUT
        << gt_x_ << ", "  // #4-1 GT bbox center x (m)
        << gt_y_ << ", "  // #4-2 GT bbox center y (m)
#endif
        << objs[i].lidarInfo.boxCenter.x << ", "                    // #5-1 input bbox center x (m)
        << objs[i].lidarInfo.boxCenter.y << ", "                    // #5-2 input bbox center y (m)
        << (objs[i].bPoint.p0.x + objs[i].bPoint.p6.x) / 2 << ", "  // #6-1 kalman-filtered bbox center x (m)
        << (objs[i].bPoint.p0.y + objs[i].bPoint.p6.y) / 2 << ", "  // #6-2 kalman-filtered bbox center y (m)
        << objs[i].track.absolute_velocity.x << ", "                // #7 abs vx (km/h)
        << objs[i].track.absolute_velocity.y << ", "                // #8 abs vy (km/h)
        << objs[i].absSpeed << ", "                                 // #9 abs speed (km/h)
        << objs[i].track.relative_velocity.x << ", "                // #10 rel vx (km/h)
        << objs[i].track.relative_velocity.y << ", "                // #11 rel vy (km/h)
        << objs[i].relSpeed;                                        // #12 rel speed (km/h)

    if (objs[i].track.is_ready_prediction)
    {
      // #13 ppx in 5 ticks (m)
      // #14 ppy in 5 ticks (m)
      // #15 ppx in 10 ticks (m)
      // #16 ppy in 10 ticks (m)
      // #17 ppx in 15 ticks (m)
      // #18 ppy in 15 ticks (m)
      // #19 ppx in 20 ticks (m)
      // #20 ppy in 20 ticks (m)
      for (size_t j = 0; j < objs[i].track.forecasts.size(); j = j + 5)
      {
        ofs << ", " << objs[i].track.forecasts[j].position.x << ", " << objs[i].track.forecasts[j].position.y;
      }

      ofs << ", "                //
          << ego_x_abs_ << ", "  // #21 ego x abs
          << ego_y_abs_ << ", "  // #22 ego y abs
          << ego_z_abs_ << ", "  // #23 ego z abs
          << ego_heading_;       // #24 ego heading (rad)

      ofs << ", "                   //
          << KTs_.get_Q1() << ", "  // #25 kf Q1
          << KTs_.get_Q2() << ", "  // #26 kf Q2
          << KTs_.get_Q3() << ", "  // #27 kf Q3
          << KTs_.get_R() << ", "   // #28 kf R
          << KTs_.get_P0();         // #29 kf P0
    }

    ofs << "\n";
    std::cout << "[Produced] time = " << objs[i].header.stamp << ", track_id = " << objs[i].track.id << std::endl;
  }

  ofs.close();
}

#if TTC_TEST
float TPPNode::closest_distance_of_obj_pivot(const msgs::DetectedObject& obj)
{
  float dist_c = euclidean_distance((obj.bPoint.p0.x + obj.bPoint.p6.x) / 2, (obj.bPoint.p0.y + obj.bPoint.p6.y) / 2);
  float dist_p0 = euclidean_distance(obj.bPoint.p0.x, obj.bPoint.p0.y);
  float dist_p3 = euclidean_distance(obj.bPoint.p3.x, obj.bPoint.p3.y);
  float dist_p4 = euclidean_distance(obj.bPoint.p4.x, obj.bPoint.p4.y);
  float dist_p7 = euclidean_distance(obj.bPoint.p7.x, obj.bPoint.p7.y);

  return std::min(std::min(std::min(std::min(dist_c, dist_p0), dist_p3), dist_p4), dist_p7);
}

void TPPNode::save_ttc_to_csv(std::vector<msgs::DetectedObject>& objs)
{
  std::ofstream ofs;
  std::stringstream ss;
  ss << "../../../ttc_output.csv";
  std::string fname = ss.str();

  if (objs.empty())
  {
    std::cout << "objs is empty. No output to .csv." << std::endl;
    return;
  }

  if (!test_file_exist(fname))
  {
    ofs.open(fname, std::ios_base::app);

    ofs << "Frame number,"              //
        << "Timestamp,"                 //
        << "dt (sec),"                  //
        << "Track ID,"                  //
        << "Distance of SV & POV (m),"  //
        << "SV abs. speed (km/h),"      //
        << "POV abs. speed (km/h),"     //
        << "POV rel. speed (km/h),"     //
        << "TTC (sec)\n";
  }
  else
  {
    ofs.open(fname, std::ios_base::app);
  }

  ros::Duration dt_s(0, dt_);

  for (size_t i = 0; i < objs.size(); i++)
  {
    float dist_m = closest_distance_of_obj_pivot(objs[i]);  //  Distance of SV & POV (m)
    double ttc_s = (objs[i].relSpeed < 0) ? (dist_m * 3.6f) / -objs[i].relSpeed : -1.;

    if (ttc_s != -1.)
    {
      ofs << seq_ << ","                        // Frame number
          << objs_header_.stamp.toSec() << ","  // Timestamp
          << dt_s.toSec() << ", "               // dt (sec)
          << objs[i].track.id << ","            // Track ID
          << dist_m << ","                      // Distance of SV & POV (m)
          << ego_speed_kmph_ << ","             // SV abs. speed (km/h)
          << objs[i].absSpeed << ","            // POV abs. speed (km/h)
          << objs[i].relSpeed << ","            // POV rel. speed (km/h)
          << ttc_s << "\n";                     // TTC (sec)

      if (ttc_s >= 0.)
        LOG_INFO << fixed << setprecision(3)  //
                 << "Seq: " << seq_ << "   Track ID: " << objs[i].track.id << "   dist = " << dist_m
                 << "m   TTC: " << ttc_s << "s (rel. speed = " << objs[i].relSpeed << " km/h)" << std::endl;
      else
        LOG_INFO << fixed << setprecision(3)  //
                 << "Seq: " << seq_ << "   Track ID: " << objs[i].track.id << "   dist = " << dist_m
                 << "m   TTC: ERROR!" << std::endl;
    }
    else
    {
      LOG_INFO << fixed << setprecision(3)  //
               << "Seq: " << seq_ << "   Track ID: " << objs[i].track.id << "   dist = " << dist_m << "m   TTC: X"
               << std::endl;
    }
  }

  ofs.close();
}
#endif

void TPPNode::publish_pp(ros::Publisher pub, std::vector<msgs::DetectedObject>& objs, const unsigned int pub_offset,
                         const float time_offset)
{
#if SAVE_OUTPUT_TXT
  save_output_to_txt(objs);
#endif

#if TTC_TEST
  save_ttc_to_csv(objs);
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

void TPPNode::get_current_ego_data(const tf2_ros::Buffer& tf_buffer, const ros::Time fusion_stamp)
{
  geometry_msgs::TransformStamped tf_stamped;

  try
  {
    tf_stamped = tf_buffer.lookupTransform("map", "lidar", fusion_stamp);
  }
  catch (tf2::TransformException& ex)
  {
    ROS_WARN("%s", ex.what());
  }

  vel_.set_ego_x_abs(tf_stamped.transform.translation.x);
  vel_.set_ego_y_abs(tf_stamped.transform.translation.y);

  double roll, pitch, yaw;
  quaternion_to_rpy(roll, pitch, yaw, tf_stamped.transform.rotation.x, tf_stamped.transform.rotation.y,
                    tf_stamped.transform.rotation.z, tf_stamped.transform.rotation.w);
  vel_.set_ego_heading(yaw * 0.01745329251f);

  ego_x_abs_ = vel_.get_ego_x_abs();
  ego_y_abs_ = vel_.get_ego_y_abs();

  std::cout << "ego_x_abs_ " << ego_x_abs_ << " ego_y_abs_ " << ego_y_abs_ << std::endl;

  ego_z_abs_ = vel_.get_ego_z_abs();
  ego_heading_ = vel_.get_ego_heading();
  ego_dx_abs_ = vel_.get_ego_dx_abs();
  ego_dy_abs_ = vel_.get_ego_dy_abs();

  ego_speed_kmph_ = vel_.get_ego_speed_kmph();
  vel_.ego_velx_vely_kmph_abs();
  ego_velx_abs_kmph_ = vel_.get_ego_velx_kmph_abs();
  ego_vely_abs_kmph_ = vel_.get_ego_vely_kmph_abs();
}

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
  par.get_ros_param_bool("show_absspeed", mc_.show_absspeed);
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

#if TTC_TEST == 0
  tf2_ros::Buffer tf_buffer;
  tf2_ros::TransformListener tf_listener(tf_buffer);
#endif

  ros::Rate loop_rate(output_fps);

  while (ros::ok() && !done_with_profiling())
  {
#if DEBUG_CALLBACK
    LOG_INFO << "ROS loop start" << std::endl;
#endif

    if (!is_legal_dt_)
    {
      tf_buffer.clear();
    }

    get_current_ego_data(tf_buffer, KTs_.header_.stamp);  // sync data

    if (g_trigger && is_legal_dt_)
    {
#if DEBUG
      LOG_INFO << "Tracking main process start" << std::endl;
#endif

#if FPS
      clock_t begin_time = clock();
#endif

      // Tracking start ==========================================================================

#if TTC_TEST
      seq_ = seq_cb_;
#endif

      KTs_.kalman_tracker_main(dt_, ego_x_abs_, ego_y_abs_, ego_z_abs_, ego_heading_);
      compute_velocity_kalman();

      publish_tracking();

#if DELAY_TIME
      mc_.module_pubtime_sec = ros::Time::now().toSec();
#endif

      // Tracking end && PP start ================================================================

      pp_.callback_tracking(pp_objs_, ego_x_abs_, ego_y_abs_, ego_z_abs_, ego_heading_);
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

    ros::spinOnce();  // Process callback_fusion()
    loop_rate.sleep();
  }

  return 0;
}
}  // namespace tpp
