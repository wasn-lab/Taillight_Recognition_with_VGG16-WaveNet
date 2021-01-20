#include "tpp_node.h"

namespace tpp
{
boost::shared_ptr<ros::AsyncSpinner> g_spinner;
static double g_input_fps = 5;    // known callback rate
static double g_output_fps = 10;  // expected publish rate

static unsigned int g_num_publishs_per_loop =
    std::max((unsigned int)1, (unsigned int)std::floor(std::floor(g_output_fps / g_input_fps)));

bool g_trigger = false;

void signal_handler(int sig)
{
  if (sig == SIGINT)
  {
    LOG_INFO << "END ITRI_Tracking_3D" << std::endl;
    g_spinner->stop();
    ros::shutdown();
  }
}

static bool done_with_profiling()
{
#if ENABLE_PROFILING_MODE
  static int num_loop = 0;
  if (num_loop < 60 * g_output_fps)
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

void TPPNode::callback_ego_speed_kmph(const msgs::VehInfo::ConstPtr& input)
{
  vel_.set_ego_speed_kmph(input->ego_speed * 3.6);

#if DEBUG_DATA_IN
  LOG_INFO << "ego_speed_kmph = " << vel_.get_ego_speed_kmph() << std::endl;
#endif
}

void TPPNode::callback_lanelet2_route(const visualization_msgs::MarkerArray::ConstPtr& input)
{
  std::cout << *input << std::endl;

  ros::Time start;
  start = ros::Time::now();

  std::vector<cv::Point3f>().swap(lanelet2_route_left);
  std::vector<cv::Point3f>().swap(lanelet2_route_right);
  lanelet2_route_left.reserve(200);
  lanelet2_route_right.reserve(200);

  for (auto const& obj : input->markers)
  {
    if (obj.ns.compare("left_lane_bound") == 0)
    {
      for (auto const& obj_point : obj.points)
      {
        cv::Point3f point;
        point.x = obj_point.x;
        point.y = obj_point.y;
        point.z = obj_point.z;
        bool push_or_not = true;
        for (size_t i = 0; i < lanelet2_route_left.size(); i++)
        {
          if (lanelet2_route_left[i].x == point.x && lanelet2_route_left[i].y == point.y &&
              lanelet2_route_left[i].z == point.z)
          {
            push_or_not = false;
          }
        }
        if (push_or_not)
        {
          lanelet2_route_left.push_back(point);
        }
      }
    }
    else if (obj.ns.compare("right_lane_bound") == 0)
    {
      for (auto const& obj_point : obj.points)
      {
        cv::Point3f point;
        point.x = obj_point.x;
        point.y = obj_point.y;
        point.z = obj_point.z;
        bool push_or_not = true;
        for (size_t i = 0; i < lanelet2_route_left.size(); i++)
        {
          if (lanelet2_route_left[i].x == point.x && lanelet2_route_left[i].y == point.y &&
              lanelet2_route_left[i].z == point.z)
          {
            push_or_not = false;
          }
        }
        if (push_or_not)
        {
          lanelet2_route_right.push_back(point);
        }
      }
    }
  }
}

void TPPNode::create_bbox_from_polygon(msgs::DetectedObject& obj)
{
  if (!obj.cPoint.lowerAreaPoints.empty())
  {
    float xmin = std::numeric_limits<float>::max();
    float xmax = -std::numeric_limits<float>::max();
    float ymin = std::numeric_limits<float>::max();
    float ymax = -std::numeric_limits<float>::max();
    float zmin = std::numeric_limits<float>::max();
    float zmax = -std::numeric_limits<float>::max();

    for (auto p : obj.cPoint.lowerAreaPoints)
    {
      xmin = (p.x < xmin) ? p.x : xmin;
      xmax = (p.x > xmax) ? p.x : xmax;
      ymin = (p.y < ymin) ? p.y : ymin;
      ymax = (p.y > ymax) ? p.y : ymax;
      zmin = (p.z < zmin) ? p.z : zmin;
      zmax = (p.z > zmax) ? p.z : zmax;
    }

    init_BoxPoint(obj.bPoint.p0, xmin, ymax, zmin);
    init_BoxPoint(obj.bPoint.p1, xmin, ymax, zmax);
    init_BoxPoint(obj.bPoint.p2, xmin, ymin, zmax);
    init_BoxPoint(obj.bPoint.p3, xmin, ymin, zmin);
    init_BoxPoint(obj.bPoint.p4, xmax, ymax, zmin);
    init_BoxPoint(obj.bPoint.p5, xmax, ymax, zmax);
    init_BoxPoint(obj.bPoint.p6, xmax, ymin, zmax);
    init_BoxPoint(obj.bPoint.p7, xmax, ymin, zmin);
  }
}
void TPPNode::create_polygon_from_bbox(const msgs::BoxPoint& bPoint, msgs::ConvexPoint& cPoint,
                                       const std::string frame_id)
{
  if (cPoint.lowerAreaPoints.size() == 0)
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

void TPPNode::callback_fusion(const msgs::DetectedObjectArray::ConstPtr& input)
{
  clock_t begin_time = -1;
  clock_t end_time = -1;

  if (show_runtime_)
  {
    begin_time = clock();
  }

#if DEBUG_CALLBACK
  LOG_INFO << "callback_fusion() start" << std::endl;
#endif

#if DEBUG_COMPACT
  LOG_INFO << "-----------------------------------------" << std::endl;
#endif

  objs_header_prev_ = objs_header_;
  objs_header_ = input->header;

#if VIRTUAL_INPUT
  objs_header_.frame_id = "lidar";
#endif

  frame_id_source_ = "base_link";

  if (objs_header_.frame_id != "lidar" && objs_header_.frame_id != "base_link")
  {
    frame_id_source_ = objs_header_.frame_id;
  }

  double objs_header_stamp = objs_header_.stamp.toSec();
  double objs_header_stamp_prev = objs_header_prev_.stamp.toSec();

  is_legal_dt_ = objs_header_stamp_prev > 0 && vel_.init_time(objs_header_stamp, objs_header_stamp_prev) == 0;

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

    KTs_.objs_.reserve(input->objects.size());

    for (const auto& obj : input->objects)
    {
      if (obj.bPoint.p0.x == 0 && obj.bPoint.p0.y == 0 && obj.bPoint.p0.z == 0 && obj.bPoint.p6.x == 0 &&
          obj.bPoint.p6.y == 0 && obj.bPoint.p6.z == 0)
      {
        continue;
      }

#if INPUT_ALL_CLASS
      KTs_.objs_.push_back(obj);
#else
      if (obj.classId == sensor_msgs_itri::DetectedObjectClassId::Person ||
          obj.classId == sensor_msgs_itri::DetectedObjectClassId::Bicycle ||
          obj.classId == sensor_msgs_itri::DetectedObjectClassId::Motobike)
      {
        KTs_.objs_.push_back(obj);
      }
#endif
    }

    for (auto& obj : KTs_.objs_)
    {
      obj.speed_abs = 0.f;
      obj.speed_rel = 0.f;

      if (create_bbox_from_polygon_)
      {
        create_bbox_from_polygon(obj);
      }
      if (create_polygon_from_bbox_)
      {
        create_polygon_from_bbox(obj.bPoint, obj.cPoint, obj.header.frame_id);
      }
    }

#if VIRTUAL_INPUT
    for (unsigned i = 0; i < KTs_.objs_.size(); i++)
    {
      gt_x_ = KTs_.objs_[i].radarInfo.imgPoint60.x;
      gt_y_ = KTs_.objs_[i].radarInfo.imgPoint60.y;
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

  if (show_runtime_)
  {
    end_time = clock();
    LOG_INFO << "[RunTime] Callback: " << CYAN_CHAR << clock_to_milliseconds(end_time - begin_time) << "ms"
             << WHITE_CHAR << std::endl;
  }
}

void TPPNode::subscribe_and_advertise_topics()
{
  std::string topic = "Tracking3D";
  use_tracking2d = false;

  if (in_source_ == InputSource::LidarDet)
  {
    LOG_INFO << "Input Source: Lidar (/LidarDetection)" << std::endl;
    fusion_sub_ = nh_.subscribe("LidarDetection", 1, &TPPNode::callback_fusion, this);
  }
  else if (in_source_ == InputSource::LidarDet_PointPillars_Car)
  {
    LOG_INFO << "Input Source: Lidar PointPillars -- Car model (/LidarDetection/Car)" << std::endl;
    fusion_sub_ = nh_.subscribe("LidarDetection/Car", 1, &TPPNode::callback_fusion, this);
  }
  else if (in_source_ == InputSource::LidarDet_PointPillars_Ped_Cyc)
  {
    LOG_INFO << "Input Source: Lidar PointPillars -- Ped & Cycle model (/LidarDetection/Ped_Cyc)" << std::endl;
    fusion_sub_ = nh_.subscribe("LidarDetection/Ped_Cyc", 1, &TPPNode::callback_fusion, this);
  }
  else if (in_source_ == InputSource::VirtualBBoxAbs)
  {
    LOG_INFO << "Input Source: Virtual_abs (/abs_virBB_array)" << std::endl;
    fusion_sub_ = nh_.subscribe("abs_virBB_array", 1, &TPPNode::callback_fusion, this);
  }
  else if (in_source_ == InputSource::VirtualBBoxRel)
  {
    LOG_INFO << "Input Source: Virtual_rel (/rel_virBB_array)" << std::endl;
    fusion_sub_ = nh_.subscribe("rel_virBB_array", 1, &TPPNode::callback_fusion, this);
  }
  else if (in_source_ == InputSource::CameraDetV2)
  {
    LOG_INFO << "Input Source: Camera approach 2 (/CameraDetection)" << std::endl;
    fusion_sub_ = nh_.subscribe("CameraDetection", 1, &TPPNode::callback_fusion, this);
  }
  else if (in_source_ == InputSource::Tracking2D)
  {
    use_tracking2d = true;
    LOG_INFO << "Input Source: Tracking 2D (/Tracking2D/front_bottom_60)" << std::endl;
    fusion_sub_ = nh_.subscribe("Tracking2D/front_bottom_60", 1, &TPPNode::callback_fusion, this);
  }
  else
  {
    LOG_INFO << "Input Source: Fusion (/SensorFusion)" << std::endl;
    fusion_sub_ = nh_.subscribe("SensorFusion", 1, &TPPNode::callback_fusion, this);
  }

  track3d_pub_ = nh_.advertise<msgs::DetectedObjectArray>(topic, 2);

#if HEARTBEAT == 1
  track3d_pub_heartbeat_ = nh_.advertise<std_msgs::Empty>(topic + std::string("/heartbeat"), 1);
#endif

  nh2_.setCallbackQueue(&queue_);

  // Note that we use different NodeHandle(nh2_) here
  ego_speed_kmph_sub_ = nh2_.subscribe("veh_info", 1, &TPPNode::callback_ego_speed_kmph, this);
  lanelet2_route_sub_ =
      nh2_.subscribe("planning/mission_planning/route_marker", 1, &TPPNode::callback_lanelet2_route, this);

  if (gen_markers_)
  {
    std::string topic2 = topic + "/id";
    mc_.pub_id = nh_.advertise<visualization_msgs::MarkerArray>(topic2, 2);

    std::string topic3 = topic + "/speed";
    mc_.pub_speed = nh_.advertise<visualization_msgs::MarkerArray>(topic3, 2);

    std::string topic4 = topic + "/vel";
    mc_.pub_vel = nh_.advertise<visualization_msgs::MarkerArray>(topic4, 2);
  }

  std::string topic5 = topic + "/drivable";
  drivable_area_pub_ = nh_.advertise<geometry_msgs::PolygonStamped>(topic5, 2);
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

    // DetectedObject.speed_abs
    track.box_.speed_abs = track.box_.track.absolute_velocity.speed;  // km/h

    if (std::isnan(track.box_.speed_abs))
    {
      track.box_.speed_abs = 0.f;
    }

    // DetectedObject.speed_rel
    MyPoint32 p_rel;
    track.box_center_.pos.get_point_rel(p_rel);  // m

    Vector3_32 rel_v_rel;
    rel_v_rel.x = track.box_.track.relative_velocity.x;  // km/h
    rel_v_rel.y = track.box_.track.relative_velocity.y;  // km/h
    rel_v_rel.z = track.box_.track.relative_velocity.z;  // km/h

    track.box_.speed_rel = compute_relative_speed_obj2ego(rel_v_rel, p_rel);  // km/h

    if (std::isnan(track.box_.speed_rel))
    {
      track.box_.speed_rel = 0.f;
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
  std::vector<msgs::DetectedObject>().swap(track3d_objs_);
  track3d_objs_.reserve(KTs_.tracks_.size());

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
        box.track.is_over_max_length = 0u;

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
          box.track.is_over_max_length = 1u;
        }

        // set states
        box.track.states.resize(box.track.max_length);

        for (unsigned k = 0; k < box.track.states.size(); k++)
        {
          box.track.states[k] = track.hist_.states_[k];
        }

        track3d_objs_.push_back(box);

#if NOT_OUTPUT_SHORT_TERM_TRACK_LOST_BBOX
      }
#endif  // NOT_OUTPUT_SHORT_TERM_TRACK_LOST_BBOX
#if REMOVE_IMPULSE_NOISE
    }
#endif  // REMOVE_IMPULSE_NOISE
  }
}

geometry_msgs::Point TPPNode::get_transform_coordinate(geometry_msgs::Point origin_point, double yaw,
                                                       geometry_msgs::Vector3 translation)
{
  geometry_msgs::Point new_point;
  new_point.x = translation.x + std::cos(yaw) * origin_point.x - std::sin(yaw) * origin_point.y;
  new_point.y = translation.y + std::sin(yaw) * origin_point.x + std::cos(yaw) * origin_point.y;
  return new_point;
}

bool TPPNode::check_in_polygon(cv::Point2f position, std::vector<cv::Point2f>& polygon)
{
  std::vector<double> vertx;
  std::vector<double> verty;
  vertx.reserve(polygon.size());
  verty.reserve(polygon.size());

  for (auto const& obj : polygon)
  {
    vertx.push_back(obj.x);
    verty.push_back(obj.y);
  }

  int nvert = polygon.size();
  double testx = position.x;
  double testy = position.y;

  int i, j, c = 0;
  for (i = 0, j = nvert - 1; i < nvert; j = i++)
  {
    if (((verty[i] > testy) != (verty[j] > testy)) &&
        (testx < (vertx[j] - vertx[i]) * (testy - verty[i]) / (verty[j] - verty[i]) + vertx[i]))
    {
      c = 1 + c;
    }
  }

  if (c % 2 == 0)
  {
    return true;
  }
  else
  {
    return false;
  }
}

bool TPPNode::drivable_area_filter(const msgs::BoxPoint box_point)
{
  cv::Point2f position;
  position.x = box_point.p0.x;
  position.y = box_point.p0.y;

  if (lanelet2_route_left.empty() || lanelet2_route_right.empty())
  {
    return false;
  }

  double yaw = tf2::getYaw(tf_stamped_.transform.rotation);

  geometry_msgs::PoseStamped point_in;
  point_in.pose.position.x = position.x;
  point_in.pose.position.y = position.y;
  point_in.pose.position.z = 0;

  geometry_msgs::PoseStamped point_out;
  point_out.pose.position = get_transform_coordinate(point_in.pose.position, yaw, tf_stamped_.transform.translation);
  position.x = point_out.pose.position.x;
  position.y = point_out.pose.position.y;

  std::vector<cv::Point3f> lanelet2_route_left_temp(lanelet2_route_left);
  std::vector<cv::Point3f> lanelet2_route_right_temp(lanelet2_route_right);
  std::vector<cv::Point2f> route_left_transformed;
  std::vector<cv::Point2f> route_right_transformed;

  for (auto const& obj : lanelet2_route_left_temp)
  {
    cv::Point2f point;
    point.x = obj.x;
    point.y = obj.y;
    route_left_transformed.push_back(point);
  }

  for (auto const& obj : lanelet2_route_right_temp)
  {
    cv::Point2f point;
    point.x = obj.x;
    point.y = obj.y;
    route_right_transformed.push_back(point);
  }

  std::vector<cv::Point3f>().swap(lanelet2_route_left_temp);
  std::vector<cv::Point3f>().swap(lanelet2_route_right_temp);

  // expand warning zone for left bound
  std::vector<cv::Point2f>().swap(expanded_route_left);
  for (size_t i = 0; i < route_left_transformed.size(); i++)
  {
    if (i == 0)
    {
      double diff_x;
      double diff_y;
      diff_x = route_left_transformed[0].x - route_right_transformed[0].x;
      diff_y = route_left_transformed[0].y - route_right_transformed[0].y;
      double distance = sqrt(pow(diff_x, 2) + pow(diff_y, 2));
      cv::Point2f expand_point = route_left_transformed[i];
      expand_point.x = expand_point.x + diff_x / distance * expand_left_;
      expand_point.y = expand_point.y + diff_y / distance * expand_left_;
      expanded_route_left.push_back(expand_point);
    }
    else if (i == route_left_transformed.size() - 1)
    {
      double diff_x;
      double diff_y;
      diff_x = route_left_transformed.back().x - route_right_transformed.back().x;
      diff_y = route_left_transformed.back().y - route_right_transformed.back().y;
      double distance = sqrt(pow(diff_x, 2) + pow(diff_y, 2));
      cv::Point2f expand_point = route_left_transformed[i];
      expand_point.x = expand_point.x + diff_x / distance * expand_left_;
      expand_point.y = expand_point.y + diff_y / distance * expand_left_;
      expanded_route_left.push_back(expand_point);
    }
    else
    {
      double diff_x;
      double diff_y;
      diff_x = route_left_transformed[i + 1].x - route_left_transformed[i - 1].x;
      diff_y = route_left_transformed[i + 1].y - route_left_transformed[i - 1].y;
      double N_x = (-1) * diff_y;
      double N_y = diff_x;
      double distance = sqrt(pow(N_x, 2) + pow(N_y, 2));
      cv::Point2f expand_point = route_left_transformed[i];
      expand_point.x = expand_point.x + N_x / distance * expand_left_;
      expand_point.y = expand_point.y + N_y / distance * expand_left_;
      expanded_route_left.push_back(expand_point);
    }
  }
  // expand warning zone for right bound
  std::vector<cv::Point2f>().swap(expanded_route_right);
  for (size_t i = 0; i < route_right_transformed.size(); i++)
  {
    if (i == 0)
    {
      double diff_x;
      double diff_y;
      diff_x = route_right_transformed[0].x - route_left_transformed[0].x;
      diff_y = route_right_transformed[0].y - route_left_transformed[0].y;
      double distance = sqrt(pow(diff_x, 2) + pow(diff_y, 2));
      cv::Point2f expand_point = route_right_transformed[i];
      expand_point.x = expand_point.x + diff_x / distance * expand_right_;
      expand_point.y = expand_point.y + diff_y / distance * expand_right_;
      expanded_route_right.push_back(expand_point);
    }
    else if (i == route_left_transformed.size() - 1)
    {
      double diff_x;
      double diff_y;
      diff_x = route_right_transformed.back().x - route_left_transformed.back().x;
      diff_y = route_right_transformed.back().y - route_left_transformed.back().y;
      double distance = sqrt(pow(diff_x, 2) + pow(diff_y, 2));
      cv::Point2f expand_point = route_right_transformed[i];
      expand_point.x = expand_point.x + diff_x / distance * expand_right_;
      expand_point.y = expand_point.y + diff_y / distance * expand_right_;
      expanded_route_right.push_back(expand_point);
    }
    else
    {
      double diff_x;
      double diff_y;
      diff_x = route_right_transformed[i + 1].x - route_right_transformed[i - 1].x;
      diff_y = route_right_transformed[i + 1].y - route_right_transformed[i - 1].y;
      double N_x = diff_y;
      double N_y = (-1) * diff_x;
      double distance = sqrt(pow(N_x, 2) + pow(N_y, 2));
      cv::Point2f expand_point = route_right_transformed[i];
      expand_point.x = expand_point.x + N_x / distance * expand_right_;
      expand_point.y = expand_point.y + N_y / distance * expand_right_;
      expanded_route_right.push_back(expand_point);
    }
  }

  // route_right_transformed add into route_left_transformed reversed
  while (!expanded_route_right.empty())
  {
    expanded_route_left.push_back(expanded_route_right.back());
    expanded_route_right.pop_back();
  }
  expanded_route_left.push_back(expanded_route_left[0]);  // close the polygon

  geometry_msgs::PolygonStamped polygon_marker;
  polygon_marker.header.frame_id = "map";

  for (auto const& obj : expanded_route_left)
  {
    geometry_msgs::Point32 polygon_point;
    polygon_point.x = obj.x;
    polygon_point.y = obj.y;
    polygon_point.z = ground_z_;
    polygon_marker.polygon.points.push_back(polygon_point);
  }

  drivable_area_pub_.publish(polygon_marker);

  // all route, check ped in polygon or not
  // no need to filter peds in warning zone
  if (check_in_polygon(position, expanded_route_left))
  {
    return true;
  }
  else
  {
    return false;
  }
  // no need to filter
  return false;
}

inline bool test_file_exist(const std::string& name)
{
  ifstream f(name.c_str());
  return f.good();
}

void TPPNode::convert(msgs::PointXYZ& p, geometry_msgs::Quaternion& q)
{
  // TF (lidar-to-map) for object pose
  geometry_msgs::Pose pose_in_lidar;
  pose_in_lidar.position.x = p.x;
  pose_in_lidar.position.y = p.y;
  pose_in_lidar.position.z = p.z;
  pose_in_lidar.orientation = q;

  geometry_msgs::Pose pose_in_map;
  tf2::doTransform(pose_in_lidar, pose_in_map, tf_stamped_);
  p.x = pose_in_map.position.x;
  p.y = pose_in_map.position.y;
  p.z = pose_in_map.position.z;
  q = pose_in_map.orientation;
}

void TPPNode::heading_enu(std::vector<msgs::DetectedObject>& objs)
{
  for (auto& obj : objs)
  {
    msgs::PointXYZ p;
    p.x = 0.;
    p.y = 0.;
    p.z = 0.;
    geometry_msgs::Quaternion q;
    q.x = obj.heading.x;
    q.y = obj.heading.y;
    q.z = obj.heading.z;
    q.w = obj.heading.w;
    convert(p, q);
    obj.heading_enu.x = q.x;
    obj.heading_enu.y = q.y;
    obj.heading_enu.z = q.z;
    obj.heading_enu.w = q.w;
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
        << "#5-1 bbox center x -- input (m), "            //
        << "#5-2 bbox center y -- input (m), "            //
        << "#5-3 bbox center z -- input (m), "            //
        << "#6-1 bbox center x -- kalman-filtered (m), "  //
        << "#6-2 bbox center y -- kalman-filtered (m), "  //
        << "#6-3 bbox center z -- kalman-filtered (m), "  //
        << "#7 abs vx (km/h), "                           //
        << "#8 abs vy (km/h), "                           //
        << "#9 abs speed (km/h), "                        //
        << "#10 rel vx (km/h), "                          //
        << "#11 rel vy (km/h), "                          //
        << "#12 rel speed (km/h), "                       //
        << "#21 ego x abs (m), "                          //
        << "#22 ego y abs (m), "                          //
        << "#23 ego z abs (m), "                          //
        << "#24 ego heading (rad), "                      //
        << "#25 kf Q1, "                                  //
        << "#26 kf Q2, "                                  //
        << "#27 kf Q3, "                                  //
        << "#28 kf R, "                                   //
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
        << obj.track.id << ", "                // #2 track id
        << dt_s.toSec() << ", "                // #3 dt (s)
#if VIRTUAL_INPUT
        << gt_x_ << ", "  // #4-1 GT bbox center x (m)
        << gt_y_ << ", "  // #4-2 GT bbox center y (m)
#endif
        << obj.lidarInfo.boxCenter.x << ", "      // #5-1 bbox center x -- input (m)
        << obj.lidarInfo.boxCenter.y << ", "      // #5-2 bbox center y -- input (m)
        << obj.lidarInfo.boxCenter.z << ", "      // #5-3 bbox center z -- input (m)
        << obj.center_point.x << ", "             // #6-1 bbox center x -- kalman-filtered (m)
        << obj.center_point.y << ", "             // #6-2 bbox center y -- kalman-filtered (m)
        << obj.center_point.z << ", "             // #6-3 bbox center z -- kalman-filtered (m)
        << obj.track.absolute_velocity.x << ", "  // #7 abs vx (km/h)
        << obj.track.absolute_velocity.y << ", "  // #8 abs vy (km/h)
        << obj.speed_abs << ", "                  // #9 abs speed (km/h)
        << obj.track.relative_velocity.x << ", "  // #10 rel vx (km/h)
        << obj.track.relative_velocity.y << ", "  // #11 rel vy (km/h)
        << obj.speed_rel << ", "                  // #12 rel speed (km/h)
        << ego_x_abs_ << ", "                     // #21 ego x abs
        << ego_y_abs_ << ", "                     // #22 ego y abs
        << ego_z_abs_ << ", "                     // #23 ego z abs
        << ego_heading_ << ", "                   // #24 ego heading (rad)
        << KTs_.get_Q1() << ", "                  // #25 kf Q1
        << KTs_.get_Q2() << ", "                  // #26 kf Q2
        << KTs_.get_Q3() << ", "                  // #27 kf Q3
        << KTs_.get_R() << ", "                   // #28 kf R
        << KTs_.get_P0() << "\n";                 // #29 kf P0

    std::cout << "[Produced] time = " << obj.header.stamp << ", track_id = " << obj.track.id << std::endl;
  }

  ofs.close();
}

void TPPNode::publish_tracking2(const ros::Publisher& pub, std::vector<msgs::DetectedObject>& objs,
                                const unsigned int pub_offset, const float time_offset)
{
#if SAVE_OUTPUT_TXT
  save_output_to_txt(objs);
#endif

  msgs::DetectedObjectArray msg;

  msg.header = objs_header_;
  msg.header.stamp = objs_header_.stamp + ros::Duration((double)time_offset);

  if (drivable_area_filter_)
  {
    msg.objects.reserve(objs.size());

    for (auto& obj : objs)
    {
      if (drivable_area_filter(obj.bPoint))
      {
        continue;
      }
      else
      {
        msg.objects.push_back(obj);
      }
    }
  }
  else
  {
    msg.objects.assign(objs.begin(), objs.end());
  }

  for (auto& obj : msg.objects)
  {
    obj.track.tracktime += pub_offset;
  }

  pub.publish(msg);

#if HEARTBEAT == 1
  std_msgs::Empty msg_heartbeat;
  track3d_pub_heartbeat_.publish(msg_heartbeat);
#endif

  if (gen_markers_)
  {
    mg_.marker_gen_main(msg.header, objs, mc_);
  }
}

void TPPNode::control_sleep(const double loop_interval)
{
  loop_elapsed = ros::Time::now().toSec() - loop_begin;

  if (show_runtime_)
  {
    LOG_INFO << "Sleep " << loop_interval - loop_elapsed << " seconds" << std::endl;
  }

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
  bool is_warning = false;

  try
  {
    tf_stamped_ = tf_buffer_.lookupTransform(frame_id_target_, frame_id_source_, fusion_stamp);
  }
  catch (tf2::TransformException& ex)
  {
    ROS_WARN("%s", ex.what());
    try
    {
      tf_stamped_ = tf_buffer_.lookupTransform(frame_id_target_, frame_id_source_, ros::Time(0));
    }
    catch (tf2::TransformException& ex)
    {
      ROS_WARN("%s", ex.what());
      is_warning = true;
    }
  }

  if (!is_warning)
  {
    vel_.set_ego_x_abs(tf_stamped_.transform.translation.x);
    vel_.set_ego_y_abs(tf_stamped_.transform.translation.y);
    vel_.set_ego_heading(tf2::getYaw(tf_stamped_.transform.rotation));
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
  std::string domain = "/itri_tracking_3d/";
  nh_.param<int>(domain + "input_source", in_source_, InputSource::LidarDet);

  nh_.param<double>(domain + "g_input_fps", g_input_fps, 10.);
  nh_.param<double>(domain + "g_output_fps", g_output_fps, 10.);
  g_num_publishs_per_loop = std::max((unsigned int)1, (unsigned int)std::floor(std::floor(g_output_fps / g_input_fps)));

  nh_.param<bool>(domain + "show_runtime", show_runtime_, false);

  nh_.param<bool>(domain + "create_bbox_from_polygon", create_bbox_from_polygon_, false);
  nh_.param<bool>(domain + "create_polygon_from_bbox", create_polygon_from_bbox_, false);

  nh_.param<bool>(domain + "drivable_area_filter", drivable_area_filter_, true);
  nh_.param<double>(domain + "expand_left", expand_left_, 2.2);
  nh_.param<double>(domain + "expand_right", expand_right_, 0.);
  nh_.param<double>(domain + "ground_z", ground_z_, -3.1);

  nh_.param<double>(domain + "m_lifetime_sec", mc_.lifetime_sec, 0.);
  mc_.lifetime_sec = (mc_.lifetime_sec == 0.) ? 1. / g_output_fps : mc_.lifetime_sec;

  nh_.param<bool>(domain + "gen_markers", gen_markers_, true);
  nh_.param<bool>(domain + "show_classid", mc_.show_classid, false);
  nh_.param<bool>(domain + "show_tracktime", mc_.show_tracktime, false);
  nh_.param<bool>(domain + "show_source", mc_.show_source, false);
  nh_.param<bool>(domain + "show_distance", mc_.show_distance, false);
  nh_.param<bool>(domain + "show_absspeed", mc_.show_absspeed, false);

  set_ColorRGBA(mc_.color, 1.f, 1.f, 0.4f, 1.f);  // Unmellow Yellow (255, 255, 102)
}

int TPPNode::run()
{
  set_ros_params();

  subscribe_and_advertise_topics();

  LOG_INFO << "ITRI_Tracking_3D is running!" << std::endl;

  signal(SIGINT, signal_handler);

  // Create AsyncSpinner, run it on all available cores and make it process custom callback queue
  g_spinner.reset(new ros::AsyncSpinner(0, &queue_));
  g_spinner->start();

  g_trigger = true;

  tf2_ros::TransformListener tf_listener(tf_buffer_);

  ros::Rate loop_rate(g_output_fps);

  while (ros::ok() && !done_with_profiling())
  {
    clock_t begin_time = -1;
    clock_t end_time = -1;

    if (show_runtime_)
    {
      begin_time = clock();
    }

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
      heading_enu(KTs_.objs_);                   // compute heading_enu (tf_map)

#if DEBUG
      LOG_INFO << "Tracking main process start" << std::endl;
#endif

      // Tracking start ==========================================================================

      // MOT: SORT algorithm
      KTs_.kalman_tracker_main(dt_, ego_x_abs_, ego_y_abs_, ego_z_abs_, ego_heading_, use_tracking2d);
      compute_velocity_kalman();

      publish_tracking();
      publish_tracking2(track3d_pub_, track3d_objs_, 0, 0);

      // Tracking end ==================================================================================

      g_trigger = false;

      if (show_runtime_)
      {
        end_time = clock();
        LOG_INFO << "[RunTime] 3D Tracking (" << KTs_.objs_.size() << " objs): " << CYAN_CHAR
                 << clock_to_milliseconds(end_time - begin_time) << "ms" << WHITE_CHAR << std::endl;
      }
    }

    ros::spinOnce();  // Process callback_fusion()
    loop_rate.sleep();
  }

  return 0;
}
}  // namespace tpp
