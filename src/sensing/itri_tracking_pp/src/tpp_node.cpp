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

void TPPNode::callback_wayarea(const nav_msgs::OccupancyGrid& input)
{
  wayarea_ = input;
}

#if TTC_TEST
void TPPNode::callback_seq(const std_msgs::Int32::ConstPtr& input)
{
  seq_cb_ = input->data;

#if DEBUG_DATA_IN
  std::cout << "seq_cb_ = " << seq_cb_ << std::endl;
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

void TPPNode::callback_ego_speed_kmph(const std_msgs::Float64::ConstPtr& input)
{
  vel_.set_ego_speed_kmph(input->data);

#if DEBUG_DATA_IN
  LOG_INFO << "ego_speed_kmph = " << vel_.get_ego_speed_kmph() << std::endl;
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
#endif

void TPPNode::callback_fusion(const msgs::DetectedObjectArray::ConstPtr& input)
{
#if DEBUG_CALLBACK
  LOG_INFO << "callback_fusion() start" << std::endl;
#endif

#if DEBUG_COMPACT
  LOG_INFO << "-----------------------------------------" << std::endl;
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

#if VIRTUAL_INPUT
    for (unsigned i = 0; i < KTs_.objs_.size(); i++)
    {
      gt_x_ = KTs_.objs_[i].radarInfo.imgPoint60.x;
      gt_y_ = KTs_.objs_[i].radarInfo.imgPoint60.y;
    }
#endif

    for (auto& obj : KTs_.objs_)
    {
      obj.header.frame_id = "lidar";
      obj.absSpeed = 0.f;

      if (input_source_ == InputSource::RadarDet)
      {
        obj.relSpeed = mps_to_kmph(obj.relSpeed);
      }
      else
      {
        obj.relSpeed = 0.f;
      }
    }

#if FILL_CONVEX_HULL
    for (auto& obj : KTs_.objs_)
    {
      fill_convex_hull(obj.bPoint, obj.cPoint, obj.header.frame_id);
    }
#endif

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

#if FPS
  clock_t end_time = clock();
  LOG_INFO << "Running time of callback_fusion(): " << clock_to_milliseconds(end_time - begin_time) << "ms"
           << std::endl;
#endif
}

void TPPNode::subscribe_and_advertise_topics()
{
  std::string topic = "PathPredictionOutput";
  use_tracking2d = false;

  if (input_source_ == InputSource::LidarDet)
  {
    LOG_INFO << "Input Source: Lidar (/LidarDetection)" << std::endl;
    fusion_sub_ = nh_.subscribe("LidarDetection", 1, &TPPNode::callback_fusion, this);
    set_ColorRGBA(mc_.color, mc_.color_lidar_tpp);
  }
  else if (input_source_ == InputSource::RadarDet)
  {
    LOG_INFO << "Input Source: Radar (/RadarDetection)" << std::endl;
    fusion_sub_ = nh_.subscribe("RadarDetection", 1, &TPPNode::callback_fusion, this);
    set_ColorRGBA(mc_.color, mc_.color_radar_tpp);
  }
  else if (input_source_ == InputSource::CameraDetV1)
  {
    LOG_INFO << "Input Source: Camera approach 1 (/cam_obj/front_bottom_60)" << std::endl;
    fusion_sub_ = nh_.subscribe("cam_obj/front_bottom_60", 1, &TPPNode::callback_fusion, this);
    set_ColorRGBA(mc_.color, mc_.color_camera_tpp);
  }
  else if (input_source_ == InputSource::VirtualBBoxAbs)
  {
    LOG_INFO << "Input Source: Virtual_abs (/abs_virBB_array)" << std::endl;
    fusion_sub_ = nh_.subscribe("abs_virBB_array", 1, &TPPNode::callback_fusion, this);
    set_ColorRGBA(mc_.color, mc_.color_fusion_tpp);
  }
  else if (input_source_ == InputSource::VirtualBBoxRel)
  {
    LOG_INFO << "Input Source: Virtual_rel (/rel_virBB_array)" << std::endl;
    fusion_sub_ = nh_.subscribe("rel_virBB_array", 1, &TPPNode::callback_fusion, this);
    set_ColorRGBA(mc_.color, mc_.color_fusion_tpp);
  }
  else if (input_source_ == InputSource::CameraDetV2)
  {
    LOG_INFO << "Input Source: Camera approach 2 (/CameraDetection)" << std::endl;
    fusion_sub_ = nh_.subscribe("CameraDetection", 1, &TPPNode::callback_fusion, this);
    set_ColorRGBA(mc_.color, mc_.color_camera_tpp);
  }
  else if (input_source_ == InputSource::Tracking2D)
  {
    use_tracking2d = true;
    LOG_INFO << "Input Source: Tracking 2D (/Tracking2D/front_bottom_60)" << std::endl;
    fusion_sub_ = nh_.subscribe("Tracking2D/front_bottom_60", 1, &TPPNode::callback_fusion, this);
    set_ColorRGBA(mc_.color, mc_.color_camera_tpp);
  }
  else
  {
    LOG_INFO << "Input Source: Fusion (/SensorFusion)" << std::endl;
    fusion_sub_ = nh_.subscribe("SensorFusion", 1, &TPPNode::callback_fusion, this);
    set_ColorRGBA(mc_.color, mc_.color_fusion_tpp);
  }

  pp_pub_ = nh_.advertise<msgs::DetectedObjectArray>(topic, 2);
#if HEARTBEAT == 1
  pp_pub_heartbeat_ = nh_.advertise<std_msgs::Empty>(topic + std::string("/heartbeat"), 1);
#endif
#if TO_GRIDMAP
  pp_grid_pub_ = nh_.advertise<nav_msgs::OccupancyGrid>("PathPredictionOutput/grid", 2);
#endif

  nh2_.setCallbackQueue(&queue_);

  // Note that we use different NodeHandle(nh2_) here
  if (occ_source_ == OccupancySource::MapBased)
  {
    wayarea_sub_ = nh2_.subscribe("occupancy_wayarea", 1, &TPPNode::callback_wayarea, this);
  }
  else
  {
    wayarea_sub_ = nh2_.subscribe("occupancy_grid_wayarea", 1, &TPPNode::callback_wayarea, this);
  }

#if TTC_TEST
  seq_sub_ = nh2_.subscribe("sequence_ID", 1, &TPPNode::callback_seq, this);
  localization_sub_ = nh2_.subscribe("player_vehicle", 1, &TPPNode::callback_localization, this);
  ego_speed_kmph_sub_ = nh2_.subscribe("player_vehicle_speed", 1, &TPPNode::callback_ego_speed_kmph, this);
#else
  ego_speed_kmph_sub_ = nh2_.subscribe("veh_info", 1, &TPPNode::callback_ego_speed_kmph, this);
#endif

  if (gen_markers_)
  {
    std::string topic1 = topic + "/markers";
    mc_.pub_bbox = nh_.advertise<visualization_msgs::MarkerArray>(topic1, 2);

    std::string topic3 = topic + "/id";
    mc_.pub_id = nh_.advertise<visualization_msgs::MarkerArray>(topic3, 2);

    std::string topic4 = topic + "/speed";
    mc_.pub_speed = nh_.advertise<visualization_msgs::MarkerArray>(topic4, 2);

    if (mc_.show_pp >= 1 && mc_.show_pp <= 3)
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
  if (cPoint.lowerAreaPoints.empty() || input_source_ == InputSource::RadarDet)
  {
    std::vector<MyPoint32>().swap(cPoint.lowerAreaPoints);
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

float TPPNode::compute_relative_speed_obj2ego(const Vector3_32 rel_v_rel, const MyPoint32 obj_rel)
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

// DetectedObject.absSpeed
#if USE_RADAR_ABS_SPEED == 0
    track.box_.absSpeed = track.box_.track.absolute_velocity.speed;  // km/h
#else
    MyPoint32 p_abs;
    box_center_.pos.get_point_abs(p_abs);
    track.box_.absSpeed = compute_radar_absolute_velocity(track.box_.relSpeed,  //
                                                          p_abs.x, p_abs.y);
#endif

    if (std::isnan(track.box_.absSpeed))
    {
      track.box_.absSpeed = 0.f;
    }

    // DetectedObject.relSpeed
    if (input_source_ != InputSource::RadarDet)
    {
      MyPoint32 p_rel;
      track.box_center_.pos.get_point_rel(p_rel);
      Vector3_32 rel_v_rel;
      rel_v_rel.x = track.box_.track.relative_velocity.x;
      rel_v_rel.y = track.box_.track.relative_velocity.y;
      rel_v_rel.z = track.box_.track.relative_velocity.z;
      track.box_.relSpeed = compute_relative_speed_obj2ego(rel_v_rel, p_rel);  // km/h
    }

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
  std::vector<msgs::DetectedObject>().swap(pp_objs_);
  pp_objs_.reserve(KTs_.tracks_.size());

  for (const auto& track : KTs_.tracks_)
  {
#if REMOVE_IMPULSE_NOISE
    if (track.tracked_)
    {
#endif  // REMOVE_IMPULSE_NOISE
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

        // set length
        box.track.length = track.hist_.len_;

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

        pp_objs_.push_back(box);

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

#if OUTPUT_MAP_TF == 1
void TPPNode::convert(msgs::PointXYZ& p, const geometry_msgs::TransformStamped tf_stamped)
{
  // TF (lidar-to-map) for object pose
  geometry_msgs::Pose pose_in_lidar;
  pose_in_lidar.position.x = p.x;
  pose_in_lidar.position.y = p.y;
  pose_in_lidar.position.z = p.z;
  pose_in_lidar.orientation.x = 0;
  pose_in_lidar.orientation.y = 0;
  pose_in_lidar.orientation.z = 0;
  pose_in_lidar.orientation.w = 1;

  geometry_msgs::Pose pose_in_map;
  tf2::doTransform(pose_in_lidar, pose_in_map, tf_stamped);
  p.x = pose_in_map.position.x;
  p.y = pose_in_map.position.y;
  p.z = pose_in_map.position.z;
}

void TPPNode::convert_all_to_map_tf(std::vector<msgs::DetectedObject>& objs)
{
  for (auto& obj : objs)
  {
    if (obj.header.frame_id != frame_id_target_)
    {
      geometry_msgs::TransformStamped tf_stamped;
      
      std::string frame_id_source_ = "base_link";

      if (obj.header.frame_id != "lidar" && obj.header.frame_id != "base_link")
      {
        frame_id_source_ = obj.header.frame_id;
      }

      try
      {
        tf_stamped = tf_buffer_.lookupTransform(frame_id_target_, frame_id_source_, obj.header.stamp);
      }
      catch (tf2::TransformException& ex1)
      {
        ROS_WARN("%s", ex1.what());
        try
        {
          tf_stamped = tf_buffer_.lookupTransform(frame_id_target_, frame_id_source_, ros::Time(0));
        }
        catch (tf2::TransformException& ex2)
        {
          ROS_WARN("%s", ex2.what());
          return;
        }
      }

      convert(obj.bPoint.p0, tf_stamped);
      convert(obj.bPoint.p1, tf_stamped);
      convert(obj.bPoint.p2, tf_stamped);
      convert(obj.bPoint.p3, tf_stamped);
      convert(obj.bPoint.p4, tf_stamped);
      convert(obj.bPoint.p5, tf_stamped);
      convert(obj.bPoint.p6, tf_stamped);
      convert(obj.bPoint.p7, tf_stamped);

      if (!obj.cPoint.lowerAreaPoints.empty())
      {
        for (auto p : obj.cPoint.lowerAreaPoints)
        {
          convert(p, tf_stamped);
        }
      }

      if (obj.track.is_ready_prediction)
      {
        for (unsigned int i = 0; i < num_forecasts_; i++)
        {
          msgs::PointXYZ p;
          p.x = obj.track.forecasts[i].position.x;
          p.y = obj.track.forecasts[i].position.y;
          p.z = 0.;

          convert(p, tf_stamped);

          obj.track.forecasts[i].position.x = p.x;
          obj.track.forecasts[i].position.y = p.y;
        }
      }

      obj.header.frame_id = frame_id_target_;
    }
  }
}
#endif

void TPPNode::save_output_to_txt(const std::vector<msgs::DetectedObject>& objs, const std::string out_filename)
{
  std::ofstream ofs;
  std::stringstream ss;
  ss << out_filename;
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
        << "#2-1 track id, "      //
        << "#2-2 class id, "      //
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
        << "#PPx in 1 tick (m), "                      //
        << "#PPy in 1 tick (m), "                      //
        << "#PPx in 2 ticks (m), "                     //
        << "#PPy in 2 ticks (m), "                     //
        << "#PPx in 3 ticks (m), "                     //
        << "#PPy in 3 ticks (m), "                     //
        << "#PPx in 4 ticks (m), "                     //
        << "#PPy in 4 ticks (m), "                     //
        << "#PPx in 5 ticks (m), "                     //
        << "#PPy in 5 ticks (m), "                     //
        << "#PPx in 6 ticks (m), "                     //
        << "#PPy in 6 ticks (m), "                     //
        << "#PPx in 7 ticks (m), "                     //
        << "#PPy in 7 ticks (m), "                     //
        << "#PPx in 8 ticks (m), "                     //
        << "#PPy in 8 ticks (m), "                     //
        << "#PPx in 9 ticks (m), "                     //
        << "#PPy in 9 ticks (m), "                     //
        << "#PPx in 10 ticks (m), "                    //
        << "#PPy in 10 ticks (m), "                    //
        << "#PPx in 11 ticks (m), "                    //
        << "#PPy in 11 ticks (m), "                    //
        << "#PPx in 12 ticks (m), "                    //
        << "#PPy in 12 ticks (m), "                    //
        << "#PPx in 13 ticks (m), "                    //
        << "#PPy in 13 ticks (m), "                    //
        << "#PPx in 14 ticks (m), "                    //
        << "#PPy in 14 ticks (m), "                    //
        << "#PPx in 15 ticks (m), "                    //
        << "#PPy in 15 ticks (m), "                    //
        << "#PPx in 16 ticks (m), "                    //
        << "#PPy in 16 ticks (m), "                    //
        << "#PPx in 17 ticks (m), "                    //
        << "#PPy in 17 ticks (m), "                    //
        << "#PPx in 18 ticks (m), "                    //
        << "#PPy in 18 ticks (m), "                    //
        << "#PPx in 19 ticks (m), "                    //
        << "#PPy in 19 ticks (m), "                    //
        << "#PPx in 20 ticks (m), "                    //
        << "#PPy in 20 ticks (m), "                    //
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

  for (const auto& obj : objs)
  {
    ofs << std::fixed                          //
        << objs_header_.stamp.toSec() << ", "  // #1 time stamp (s)
        << obj.track.id << ", "                // #2-1 track id
        << obj.classId << ", "                 // #2-2 class id
        << dt_s.toSec() << ", "                // #3 dt (s)
#if VIRTUAL_INPUT
        << gt_x_ << ", "  // #4-1 GT bbox center x (m)
        << gt_y_ << ", "  // #4-2 GT bbox center y (m)
#endif
        << obj.lidarInfo.boxCenter.x << ", "                // #5-1 input bbox center x (m)
        << obj.lidarInfo.boxCenter.y << ", "                // #5-2 input bbox center y (m)
        << (obj.bPoint.p0.x + obj.bPoint.p6.x) / 2 << ", "  // #6-1 kalman-filtered bbox center x (m)
        << (obj.bPoint.p0.y + obj.bPoint.p6.y) / 2 << ", "  // #6-2 kalman-filtered bbox center y (m)
        << obj.track.absolute_velocity.x << ", "            // #7 abs vx (km/h)
        << obj.track.absolute_velocity.y << ", "            // #8 abs vy (km/h)
        << obj.absSpeed << ", "                             // #9 abs speed (km/h)
        << obj.track.relative_velocity.x << ", "            // #10 rel vx (km/h)
        << obj.track.relative_velocity.y << ", "            // #11 rel vy (km/h)
        << obj.relSpeed;                                    // #12 rel speed (km/h)

    if (obj.track.is_ready_prediction)
    {
      // #PP1x~PP20y ppx in 1~20 ticks (m)
      // #PP1y~PP20y ppy in 1~20 ticks (m)
      for (unsigned int i = 0; i < num_forecasts_; i++)
      {
        ofs << ", " << obj.track.forecasts[i].position.x << ", " << obj.track.forecasts[i].position.y;
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
    std::cout << "[Produced] time = " << obj.header.stamp << ", track_id = " << obj.track.id << std::endl;
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

  for (const auto& obj : objs)
  {
    float dist_m = closest_distance_of_obj_pivot(obj);  //  Distance of SV & POV (m)
    double ttc_s = (obj.relSpeed < 0) ? (dist_m * 3.6f) / -obj.relSpeed : -1.;

    if (ttc_s != -1.)
    {
      ofs << seq_ << ","                        // Frame number
          << objs_header_.stamp.toSec() << ","  // Timestamp
          << dt_s.toSec() << ", "               // dt (sec)
          << obj.track.id << ","                // Track ID
          << dist_m << ","                      // Distance of SV & POV (m)
          << ego_speed_kmph_ << ","             // SV abs. speed (km/h)
          << obj.absSpeed << ","                // POV abs. speed (km/h)
          << obj.relSpeed << ","                // POV rel. speed (km/h)
          << ttc_s << "\n";                     // TTC (sec)

      if (ttc_s >= 0.)
        LOG_INFO << fixed << setprecision(3)  //
                 << "Seq: " << seq_ << "   Track ID: " << obj.track.id << "   dist = " << dist_m << "m   TTC: " << ttc_s
                 << "s (rel. speed = " << obj.relSpeed << " km/h)" << std::endl;
      else
        LOG_INFO << fixed << setprecision(3)  //
                 << "Seq: " << seq_ << "   Track ID: " << obj.track.id << "   dist = " << dist_m << "m   TTC: ERROR!"
                 << std::endl;
    }
    else
    {
      LOG_INFO << fixed << setprecision(3)  //
               << "Seq: " << seq_ << "   Track ID: " << obj.track.id << "   dist = " << dist_m << "m   TTC: X"
               << std::endl;
    }
  }

  ofs.close();
}
#endif

#if TO_GRIDMAP
void TPPNode::publish_pp_grid(ros::Publisher pub, const std::vector<msgs::DetectedObject>& objs)
{
  pcl::PointCloud<pcl::PointXYZ>::Ptr in_points(new pcl::PointCloud<pcl::PointXYZ>);

  for (const auto& obj : objs)
  {
    if (obj.track.is_ready_prediction)
    {
      for (unsigned int j = num_forecasts_; j < num_forecasts_ * 5; j++)
      {
        pcl::PointXYZ p;
        p.x = obj.track.forecasts[j].position.x;
        p.y = obj.track.forecasts[j].position.y;
        p.z = 0.f;
        in_points->push_back(p);
      }
    }
  }

  grid_map::GridMap gridmap;
  gridmap.setFrameId("base_link");
  gridmap.setGeometry(grid_map::Length(50, 30), 0.2, grid_map::Position(10, 0));
  gridmap.add("pp_layer", grid_map::Matrix::Constant(gridmap.getSize()(0), gridmap.getSize()(1), 0.0));

  gridmap["pp_layer"] = PointsToCostmap().makeCostmapFromSensorPoints(5, -5, 0.0, 1.0, gridmap, "pp_layer", in_points);
  nav_msgs::OccupancyGrid occ_grid_msg;
  grid_map::GridMapRosConverter::toOccupancyGrid(gridmap, "pp_layer", 0, 1, occ_grid_msg);
  occ_grid_msg.header = objs_header_;
  pub.publish(occ_grid_msg);
}
#endif

void TPPNode::publish_pp(ros::Publisher pub, std::vector<msgs::DetectedObject>& objs, const unsigned int pub_offset,
                         const float time_offset)
{
#if SAVE_OUTPUT_TXT == 1
  save_output_to_txt(objs, "../../../tracking_rpp_output_tf_lidar.txt");
#endif

#if OUTPUT_MAP_TF == 1
  convert_all_to_map_tf(objs);
#if SAVE_OUTPUT_TXT == 1
  save_output_to_txt(objs, "../../../tracking_rpp_output_tf_map.txt");
#endif
#endif

#if TTC_TEST
  save_ttc_to_csv(objs);
#endif

  msgs::DetectedObjectArray msg;

  msg.header = objs_header_;
  msg.header.stamp = objs_header_.stamp + ros::Duration((double)time_offset);

  msg.objects.assign(objs.begin(), objs.end());

  for (auto& obj : msg.objects)
  {
    obj.track.tracktime += pub_offset;
  }

  pub.publish(msg);
#if HEARTBEAT == 1
  std_msgs::Empty msg_heartbeat;
  pp_pub_heartbeat_.publish(msg_heartbeat);
#endif

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
  {
    this_thread::sleep_for(std::chrono::milliseconds((long int)round(1000 * (loop_interval - loop_elapsed))));
  }

  loop_begin = ros::Time::now().toSec();
}

void TPPNode::get_current_ego_data_main()
{
  ego_x_abs_ = vel_.get_ego_x_abs();
  ego_y_abs_ = vel_.get_ego_y_abs();

  ego_z_abs_ = vel_.get_ego_z_abs();
  ego_heading_ = vel_.get_ego_heading();
  ego_dx_abs_ = vel_.get_ego_dx_abs();
  ego_dy_abs_ = vel_.get_ego_dy_abs();

  ego_speed_kmph_ = vel_.get_ego_speed_kmph();
  vel_.ego_velx_vely_kmph_abs();
  ego_velx_abs_kmph_ = vel_.get_ego_velx_kmph_abs();
  ego_vely_abs_kmph_ = vel_.get_ego_vely_kmph_abs();
}

void TPPNode::get_current_ego_data(const ros::Time fusion_stamp)
{
  geometry_msgs::TransformStamped tf_stamped;
  bool is_warning = false;

  try
  {
    tf_stamped = tf_buffer_.lookupTransform("map", "lidar", fusion_stamp);
  }
  catch (tf2::TransformException& ex)
  {
    ROS_WARN("%s", ex.what());
    try
    {
      tf_stamped = tf_buffer_.lookupTransform("map", "lidar", ros::Time(0));
    }
    catch (tf2::TransformException& ex)
    {
      ROS_WARN("%s", ex.what());
      is_warning = true;
    }
  }

  if (!is_warning)
  {
    vel_.set_ego_x_abs(tf_stamped.transform.translation.x);
    vel_.set_ego_y_abs(tf_stamped.transform.translation.y);

    double roll, pitch, yaw;
    quaternion_to_rpy(roll, pitch, yaw, tf_stamped.transform.rotation.x, tf_stamped.transform.rotation.y,
                      tf_stamped.transform.rotation.z, tf_stamped.transform.rotation.w);
    vel_.set_ego_heading(yaw);
  }
  else
  {
    vel_.set_ego_x_abs(0.f);
    vel_.set_ego_y_abs(0.f);
    vel_.set_ego_heading(0.f);
  }

  get_current_ego_data_main();
}

void TPPNode::set_ros_params()
{
  std::string domain = "/itri_tracking_pp/";
  nh_.param<int>(domain + "input_source", input_source_, InputSource::CameraDetV2);
  nh_.param<int>(domain + "occ_source", occ_source_, OccupancySource::PlannedPathBased);

  nh_.param<double>(domain + "input_fps", input_fps, 10.);
  nh_.param<double>(domain + "output_fps", output_fps, 10.);
  num_publishs_per_loop = std::max((unsigned int)1, (unsigned int)std::floor(std::floor(output_fps / input_fps)));

  double pp_input_shift_m = 0.;
  nh_.param<double>(domain + "pp_input_shift_m", pp_input_shift_m, 150.);
  pp_.set_input_shift_m((long double)pp_input_shift_m);

  nh_.param<double>(domain + "m_lifetime_sec", mc_.lifetime_sec, 0.);
  mc_.lifetime_sec = (mc_.lifetime_sec == 0.) ? 1. / output_fps : mc_.lifetime_sec;

  nh_.param<bool>(domain + "gen_markers", gen_markers_, true);
  nh_.param<bool>(domain + "show_classid", mc_.show_classid, false);
  nh_.param<bool>(domain + "show_tracktime", mc_.show_tracktime, false);
  nh_.param<bool>(domain + "show_source", mc_.show_source, false);
  nh_.param<bool>(domain + "show_distance", mc_.show_distance, false);
  nh_.param<bool>(domain + "show_absspeed", mc_.show_absspeed, false);

  int show_pp_int = 0;
  nh_.param<int>(domain + "show_pp", show_pp_int, 0);
  mc_.show_pp = (unsigned int)show_pp_int;

  int num_pp_input_min = 0;
  nh_.param<int>(domain + "num_pp_input_min", num_pp_input_min, 0);
  pp_.set_num_pp_input_min((std::size_t)std::max(num_pp_input_min, 0));

  double pp_obj_min_kmph = 0.;
  nh_.param<double>(domain + "pp_obj_min_kmph", pp_obj_min_kmph, 3.);
  pp_.set_pp_obj_min_kmph(pp_obj_min_kmph);

  double pp_obj_max_kmph = 0.;
  nh_.param<double>(domain + "pp_obj_max_kmph", pp_obj_max_kmph, 50.);
  pp_.set_pp_obj_max_kmph(pp_obj_max_kmph);

  set_ColorRGBA(mc_.color_lidar_tpp, 1.f, 1.f, 0.4f, 1.f);  // Unmellow Yellow (255, 255, 102)
  set_ColorRGBA(mc_.color_radar_tpp, 1.f, 1.f, 0.4f, 1.f);
  set_ColorRGBA(mc_.color_camera_tpp, 1.f, 1.f, 0.4f, 1.f);
  set_ColorRGBA(mc_.color_fusion_tpp, 1.f, 1.f, 0.4f, 1.f);
}

int TPPNode::run()
{
  set_ros_params();

  subscribe_and_advertise_topics();

  LOG_INFO << "ITRI_Tracking_PP is running!" << std::endl;

  signal(SIGINT, signal_handler);

  // Create AsyncSpinner, run it on all available cores and make it process custom callback queue
  g_spinner.reset(new ros::AsyncSpinner(0, &queue_));
  g_spinner->start();

  g_trigger = true;

#if TTC_TEST == 0
  tf2_ros::TransformListener tf_listener(tf_buffer_);
#endif

  ros::Rate loop_rate(output_fps);

  while (ros::ok() && !done_with_profiling())
  {
#if DEBUG_CALLBACK
    LOG_INFO << "ROS loop start" << std::endl;
#endif

    if (!is_legal_dt_)
    {
      tf_buffer_.clear();
    }

    if (g_trigger && is_legal_dt_)
    {
      get_current_ego_data(KTs_.header_.stamp);  // sync data

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

      // MOT: SORT algorithm
      KTs_.kalman_tracker_main(dt_, ego_x_abs_, ego_y_abs_, ego_z_abs_, ego_heading_, use_tracking2d);
      compute_velocity_kalman();

      publish_tracking();

#if DELAY_TIME
      mc_.module_pubtime_sec = ros::Time::now().toSec();
#endif

      // Tracking --> PP =========================================================================

      pp_.callback_tracking(pp_objs_, ego_x_abs_, ego_y_abs_, ego_z_abs_, ego_heading_, input_source_);
      pp_.main(pp_objs_, ppss, mc_.show_pp, wayarea_);  // PP: autoregression of order 1 -- AR(1)

      publish_pp(pp_pub_, pp_objs_, 0, 0);
#if TO_GRIDMAP
      publish_pp_grid(pp_grid_pub_, pp_objs_);
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
