#include "daf_node.h"

namespace daf
{
boost::shared_ptr<ros::AsyncSpinner> g_spinner;

bool g_trigger = false;

void signal_handler(int sig)
{
  if (sig == SIGINT)
  {
    LOG_INFO << "END Drivable_Area_Filter" << std::endl;
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

void DAFNode::callback_lanelet2_route(const visualization_msgs::MarkerArray::ConstPtr& input)
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

void DAFNode::callback_detected_objs(const msgs::DetectedObjectArray::ConstPtr& input)
{
  objs_header_ = input->header;

  frame_id_source_ = "base_link";
  if (objs_header_.frame_id != "lidar" && objs_header_.frame_id != "base_link")
  {
    frame_id_source_ = objs_header_.frame_id;
  }

  get_current_ego_data(objs_header_.stamp);

  std::vector<msgs::DetectedObject>().swap(objs_);
  objs_.reserve(input->objects.size());

  for (const auto& obj : input->objects)
  {
    if (obj.bPoint.p0.x == 0 && obj.bPoint.p0.y == 0 && obj.bPoint.p0.z == 0 && obj.bPoint.p6.x == 0 &&
        obj.bPoint.p6.y == 0 && obj.bPoint.p6.z == 0)
    {
      continue;
    }

    if (use_filter_)
    {
      if (drivable_area_filter(obj.bPoint))
      {
        continue;
      }
      else
      {
        objs_.push_back(obj);
      }
    }
    else
    {
      objs_.push_back(obj);
    }
  }

  g_trigger = true;
}

void DAFNode::subscribe_and_advertise_topics()
{
  if (input_source_ == InputSource::LidarDet)
  {
    LOG_INFO << "Input Source: LiDAR (/LidarDetection)" << std::endl;
    sub_input_ = nh_.subscribe("LidarDetection", 1, &DAFNode::callback_detected_objs, this);
  }
  else if (input_source_ == InputSource::LidarDet_PointPillars_Car)
  {
    LOG_INFO << "Input Source: LiDAR PointPillars -- Car model (/LidarDetection/Car)" << std::endl;
    sub_input_ = nh_.subscribe("LidarDetection/Car", 1, &DAFNode::callback_detected_objs, this);
  }
  else if (input_source_ == InputSource::LidarDet_PointPillars_Ped_Cyc)
  {
    LOG_INFO << "Input Source: LiDAR PointPillars -- Ped & Cycle model (/LidarDetection/Ped_Cyc)" << std::endl;
    sub_input_ = nh_.subscribe("LidarDetection/Ped_Cyc", 1, &DAFNode::callback_detected_objs, this);
  }
  else if (input_source_ == InputSource::VirtualBBoxAbs)
  {
    LOG_INFO << "Input Source: Virtual_abs (/abs_virBB_array)" << std::endl;
    sub_input_ = nh_.subscribe("abs_virBB_array", 1, &DAFNode::callback_detected_objs, this);
  }
  else if (input_source_ == InputSource::VirtualBBoxRel)
  {
    LOG_INFO << "Input Source: Virtual_rel (/rel_virBB_array)" << std::endl;
    sub_input_ = nh_.subscribe("rel_virBB_array", 1, &DAFNode::callback_detected_objs, this);
  }
  else if (input_source_ == InputSource::CameraDetV2)
  {
    LOG_INFO << "Input Source: Camera approach 2 (/CameraDetection)" << std::endl;
    sub_input_ = nh_.subscribe("CameraDetection", 1, &DAFNode::callback_detected_objs, this);
  }
  else if (input_source_ == InputSource::Tracking2D)
  {
    LOG_INFO << "Input Source: Tracking 2D (/Tracking2D/front_bottom_60)" << std::endl;
    sub_input_ = nh_.subscribe("Tracking2D/front_bottom_60", 1, &DAFNode::callback_detected_objs, this);
  }
  else
  {
    LOG_INFO << "Input Source: Fusion (/SensorFusion)" << std::endl;
    sub_input_ = nh_.subscribe("SensorFusion", 1, &DAFNode::callback_detected_objs, this);
  }

  std::string topic = "filtered_objs";
  pub_daf_ = nh_.advertise<msgs::DetectedObjectArray>(topic, 1);
  pub_daf_heartbeat_ = nh_.advertise<std_msgs::Empty>(topic + std::string("/heartbeat"), 1);

  // Use NodeHandle(nh2_) here
  nh2_.setCallbackQueue(&queue_);
  sub_lanelet2_route_ =
      nh2_.subscribe("planning/mission_planning/route_marker", 1, &DAFNode::callback_lanelet2_route, this);

  std::string topic5 = topic + "/drivable";
  pub_drivable_area_ = nh_.advertise<geometry_msgs::PolygonStamped>(topic5, 2);
}

void DAFNode::get_current_ego_data(const ros::Time input_stamp)
{
  try
  {
    tf_stamped_ = tf_buffer_.lookupTransform(frame_id_target_, frame_id_source_, input_stamp);
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
    }
  }
}

geometry_msgs::Point DAFNode::get_transform_coordinate(geometry_msgs::Point origin_point, double yaw,
                                                       geometry_msgs::Vector3 translation)
{
  geometry_msgs::Point new_point;
  new_point.x = translation.x + std::cos(yaw) * origin_point.x - std::sin(yaw) * origin_point.y;
  new_point.y = translation.y + std::sin(yaw) * origin_point.x + std::cos(yaw) * origin_point.y;
  return new_point;
}

bool DAFNode::check_in_polygon(cv::Point2f position, std::vector<cv::Point2f>& polygon)
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

bool DAFNode::drivable_area_filter(const msgs::BoxPoint box_point)
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

  pub_drivable_area_.publish(polygon_marker);

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

void DAFNode::publish_filtered_objs()
{
  msgs::DetectedObjectArray msg;

  msg.header = objs_header_;
  msg.objects.assign(objs_.begin(), objs_.end());
  pub_daf_.publish(msg);

  std_msgs::Empty msg_heartbeat;
  pub_daf_heartbeat_.publish(msg_heartbeat);
}

void DAFNode::set_ros_params()
{
  std::string domain = "/drivable_area_filter/";
  nh_.param<bool>(domain + "use_filter", use_filter_, true);
  nh_.param<double>(domain + "expand_left", expand_left_, 2.2);
  nh_.param<double>(domain + "expand_right", expand_right_, 0.0);
  nh_.param<double>(domain + "ground_z", ground_z_, -3.1);
}

int DAFNode::run()
{
  set_ros_params();

  subscribe_and_advertise_topics();

  LOG_INFO << "Drivable_Area_Filter is running!" << std::endl;

  signal(SIGINT, signal_handler);

  // Create AsyncSpinner, run it on all available cores and make it process custom callback queue
  g_spinner.reset(new ros::AsyncSpinner(0, &queue_));
  g_spinner->start();

  g_trigger = true;

  tf2_ros::TransformListener tf_listener(tf_buffer_);

  ros::Rate loop_rate(10.0);

  while (ros::ok() && !done_with_profiling())
  {
    if (g_trigger)
    {
      publish_filtered_objs();
      g_trigger = false;
    }

    ros::spinOnce();  // Process callback_detected_objs()
    loop_rate.sleep();
  }

  return 0;
}
}  // namespace daf
