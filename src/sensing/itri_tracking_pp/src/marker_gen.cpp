#include "marker_gen.h"

namespace tpp
{
geometry_msgs::Point MarkerGen::init_Point(const double x, const double y, const double z)
{
  geometry_msgs::Point p;
  p.x = x;
  p.y = y;
  p.z = z;
  return p;
}

void MarkerGen::set_marker_attr(visualization_msgs::Marker& marker, const geometry_msgs::Point point)
{
  marker.lifetime = ros::Duration(mc_.lifetime_sec);

  set_ColorRGBA(marker.color, mc_.color);

  marker.pose.position.x = point.x;
  marker.pose.position.y = point.y;
  marker.pose.position.z = point.z;

  marker.pose.orientation.x = 0.0;
  marker.pose.orientation.y = 0.0;
  marker.pose.orientation.z = 0.0;
  marker.pose.orientation.w = 1.0;
}

visualization_msgs::Marker MarkerGen::create_seq_marker(const unsigned int idx, const geometry_msgs::Point point)
{
  visualization_msgs::Marker marker;

  marker.header = header_;
  marker.ns = "PPOutput_q";
  marker.action = visualization_msgs::Marker::ADD;
  marker.id = idx;
  marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
  // marker.scale.x = 10.0;
  // marker.scale.y = 1.0;
  marker.scale.z = 1.7;

  std::ostringstream ss;
  ss << "seq = " << seq_;
  std::string str = ss.str();
  marker.text = str;

  set_marker_attr(marker, point);

  return marker;
}

geometry_msgs::Point MarkerGen::text_marker_position(const MyPoint32 p1, const MyPoint32 p2, const double z_offset)
{
  geometry_msgs::Point p;
  p.x = (p1.x + p2.x) * 0.5;
  p.y = (p1.y + p2.y) * 0.5;
  p.z = (p1.z + p2.z) * 0.5 + z_offset;
  return p;
}

visualization_msgs::Marker MarkerGen::create_box_marker(const unsigned int idx, const msgs::BoxPoint bbox,
                                                        std_msgs::Header obj_header)
{
  visualization_msgs::Marker marker;

#if SAME_OBJ_MARKER_HEADER
  marker.header = header_;
#else
  marker.header = obj_header;
#endif
  marker.ns = "PPOutput_b";
  marker.action = visualization_msgs::Marker::ADD;
  marker.pose.orientation.w = 1.0;
  marker.id = idx;
  marker.type = visualization_msgs::Marker::LINE_LIST;
  marker.scale.x = 0.2;
  marker.lifetime = ros::Duration(mc_.lifetime_sec);
  set_ColorRGBA(marker.color, mc_.color);

  std::vector<MyPoint32> point_list = { bbox.p0, bbox.p1, bbox.p2, bbox.p3, bbox.p4, bbox.p5, bbox.p6, bbox.p7 };

  marker.points.reserve(box_order.size());
  for (const auto i : box_order)
  {
    geometry_msgs::Point point_msg;
    convert_MyPoint32_to_Point(point_msg, point_list[i]);
    marker.points.push_back(point_msg);
  }

  return marker;
}

std::string MarkerGen::parse_class_id(unsigned int class_id)
{
  std::string class_name;
  switch (class_id)
  {
    case 1:
      class_name = "Person";
      break;
    case 2:
      class_name = "Bicycle";
      break;
    case 3:
      class_name = "Motobike";
      break;
    case 4:
      class_name = "Car";
      break;
    case 5:
      class_name = "Bus";
      break;
    case 6:
      class_name = "Truck";
      break;
    case 7:
      class_name = "Sign";
      break;
    case 8:
      class_name = "Light";
      break;
    case 9:
      class_name = "Park";
      break;
    default:
      class_name = "Unknown";
  }

  return class_name;
}

std::string MarkerGen::parse_source_id(unsigned int source_id)
{
  std::string source_name;
  switch (source_id)
  {
    case 0:
      source_name = "C";  // Camera
      break;
    case 1:
      source_name = "R";  // Radar
      break;
    case 2:
      source_name = "L";  // Lidar
      break;
    default:
      source_name = "X";  // Undefined
  }

  return source_name;
}

visualization_msgs::Marker MarkerGen::create_trackid_marker(const unsigned int idx, const geometry_msgs::Point point,
                                                            const msgs::DetectedObject& obj)
{
  visualization_msgs::Marker marker;

#if SAME_OBJ_MARKER_HEADER
  marker.header = header_;
#else
  marker.header = obj.header;
#endif
  marker.ns = "PPOutput_i";
  marker.action = visualization_msgs::Marker::ADD;
  marker.id = idx;
  marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
  // marker.scale.x = 10.0;
  // marker.scale.y = 1.0;
  marker.scale.z = 1.7;

  std::ostringstream ss;

  if (mc_.show_classid)
  {
    std::string class_name = parse_class_id(obj.classId);
    ss << class_name << "_";
  }

  ss << obj.track.id % 1000 + 1;

  if (mc_.show_tracktime)
  {
    ss << "(" << obj.track.tracktime << ")";
  }

  if (mc_.show_source)
  {
    std::string source_name = parse_source_id(obj.fusionSourceId);
    ss << source_name;
  }

  if (mc_.show_distance)
  {
    ss << "_" << std::setprecision(1) << std::fixed << obj.distance << "m";
  }

  std::string str = ss.str();
  marker.text = str;

  set_marker_attr(marker, point);

  return marker;
}

visualization_msgs::Marker MarkerGen::create_speed_marker(const unsigned int idx, const geometry_msgs::Point point,
                                                          std_msgs::Header obj_header, const float relspeed,
                                                          const float absspeed)
{
  visualization_msgs::Marker marker;

#if SAME_OBJ_MARKER_HEADER
  marker.header = header_;
#else
  marker.header = obj_header;
#endif
  marker.ns = "PPOutput_s";
  marker.action = visualization_msgs::Marker::ADD;
  marker.id = idx;
  marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
  // marker.scale.x = 10.0;
  // marker.scale.y = 1.0;
  marker.scale.z = 1.7;

  std::ostringstream ss;

  if (mc_.show_absspeed)
  {
    ss << "abs:" << std::setprecision(1) << std::fixed << absspeed;
  }
  else
  {
    ss << "rel:" << std::setprecision(1) << std::fixed << relspeed;
  }

  std::string str = ss.str();
  marker.text = str;

  set_marker_attr(marker, point);

  return marker;
}

visualization_msgs::Marker MarkerGen::create_pp_marker_ellipse(const unsigned int idx, const msgs::PointXY pos,
                                                               std_msgs::Header obj_header, const PPLongDouble pp,
                                                               const unsigned int forecast_seq,
                                                               const float abs_speed_kmph)
{
  visualization_msgs::Marker marker;

#if SAME_OBJ_MARKER_HEADER
  marker.header = header_;
#else
  marker.header = obj_header;
#endif
  marker.ns = "PPOutput_pp1";
  marker.action = visualization_msgs::Marker::ADD;
  marker.id = idx;
  marker.type = visualization_msgs::Marker::CYLINDER;

  double scale = abs_speed_kmph * (forecast_seq + 1) / 120.;
  marker.scale.x = scale;
  marker.scale.y = scale;
  marker.scale.z = 0.1;

  marker.pose.position.x = pos.x;
  marker.pose.position.y = pos.y;
  marker.pose.position.z = 0.;

  marker.pose.orientation = tf2::toMsg(pp.q1);

  marker.lifetime = ros::Duration(mc_.lifetime_sec);

  marker.color.r = 0.8;
  marker.color.g = 0.9 - forecast_seq * 0.035;
  marker.color.b = 0.0;
  marker.color.a = 0.3 - forecast_seq * 0.005;

  return marker;
}

visualization_msgs::Marker MarkerGen::create_pp_marker_point(const unsigned int idx, const msgs::PointXY pos,
                                                             std_msgs::Header obj_header)
{
  visualization_msgs::Marker marker;

#if SAME_OBJ_MARKER_HEADER
  marker.header = header_;
#else
  marker.header = obj_header;
#endif
  marker.ns = "PPOutput_pp2";
  marker.action = visualization_msgs::Marker::ADD;
  marker.id = idx;
  marker.type = visualization_msgs::Marker::CYLINDER;

  marker.scale.x = 0.3;
  marker.scale.y = 0.3;
  marker.scale.z = 0.1;

  marker.pose.position.x = pos.x;
  marker.pose.position.y = pos.y;
  marker.pose.position.z = 0.1;

  marker.pose.orientation.x = 0;
  marker.pose.orientation.y = 0;
  marker.pose.orientation.z = 0;
  marker.pose.orientation.w = 1;

  marker.lifetime = ros::Duration(mc_.lifetime_sec);

  marker.color.r = 1.0;
  marker.color.g = 1.0;
  marker.color.b = 0.0;
  marker.color.a = 1.0;

  return marker;
}

visualization_msgs::Marker MarkerGen::create_vel_marker(const unsigned int idx, const geometry_msgs::Point point,
                                                        const float vx, const float vy, std_msgs::Header obj_header)
{
  visualization_msgs::Marker marker;

#if SAME_OBJ_MARKER_HEADER
  marker.header = header_;
#else
  marker.header = obj_header;
#endif
  marker.ns = "PPOutput_pp";
  marker.action = visualization_msgs::Marker::ADD;
  marker.id = idx;
  marker.type = visualization_msgs::Marker::LINE_LIST;

  marker.scale.x = 0.25;

  marker.lifetime = ros::Duration(mc_.lifetime_sec);

  marker.color.r = 1.0;
  marker.color.g = 0.0;
  marker.color.b = 1.0;
  marker.color.a = 1.0;

  marker.points.push_back(point);

  geometry_msgs::Point p;
  double scale = 0.5555555556;  // 5 / 9
  p.x = point.x + vx * scale;
  p.y = point.y + vy * scale;
  p.z = point.z;

  marker.points.push_back(p);

  return marker;
}

void MarkerGen::process_text_marker(unsigned int& idx, const std::vector<msgs::DetectedObject>& objs)
{
  std::vector<visualization_msgs::Marker>().swap(m_id_.markers);
  std::vector<visualization_msgs::Marker>().swap(m_speed_.markers);

  m_id_.markers.reserve(objs.size());
  m_speed_.markers.reserve(objs.size());

  for (const auto& obj : objs)
  {
    geometry_msgs::Point point = text_marker_position(obj.bPoint.p1, obj.bPoint.p2, 2.);
    m_id_.markers.push_back(create_trackid_marker(idx++, point, obj));
    m_speed_.markers.push_back(create_speed_marker(idx++, point, obj.header, obj.relSpeed, obj.absSpeed));
  }

  geometry_msgs::Point point_seq = init_Point(-20, 0, 0);
  m_id_.markers.push_back(create_seq_marker(idx++, point_seq));

  mc_.pub_id.publish(m_id_);
  mc_.pub_speed.publish(m_speed_);
}

void MarkerGen::process_box_marker(unsigned int& idx, const std::vector<msgs::DetectedObject>& objs)
{
  std::vector<visualization_msgs::Marker>().swap(m_box_.markers);
  m_box_.markers.reserve(objs.size());

  for (const auto& obj : objs)
  {
    m_box_.markers.push_back(create_box_marker(idx++, obj.bPoint, obj.header));
  }

  mc_.pub_bbox.publish(m_box_);
}

void MarkerGen::process_pp_marker(unsigned int& idx, const std::vector<msgs::DetectedObject>& objs,
                                  std::vector<std::vector<PPLongDouble> >& ppss)
{
  std::vector<visualization_msgs::Marker>().swap(m_pp_.markers);

  if (mc_.show_pp == 1 || mc_.show_pp == 2)
  {
    m_pp_.markers.reserve(objs.size() * num_forecasts_);
  }
  else if (mc_.show_pp == 3)
  {
    m_pp_.markers.reserve(objs.size() * num_forecasts_ * 5);
  }
  else
  {
    return;
  }

  for (unsigned i = 0; i < objs.size(); i++)
  {
    if (objs[i].track.is_ready_prediction)
    {
      for (int j = (int)num_forecasts_ - 1; j >= 0; j--)
      {
        if (mc_.show_pp == 1)
        {
          m_pp_.markers.push_back(create_pp_marker_point(idx++, objs[i].track.forecasts[j].position, objs[i].header));
        }
        else if (mc_.show_pp == 2 || mc_.show_pp == 3)
        {
          m_pp_.markers.push_back(create_pp_marker_ellipse(idx++, objs[i].track.forecasts[j].position, objs[i].header,
                                                           ppss[i][j], j, objs[i].absSpeed));
        }
      }

      if (mc_.show_pp == 3)
      {
        for (unsigned int j = num_forecasts_; j < num_forecasts_ * 5; j++)
        {
          m_pp_.markers.push_back(create_pp_marker_point(idx++, objs[i].track.forecasts[j].position, objs[i].header));
        }
      }
    }
  }

  mc_.pub_pp.publish(m_pp_);
}

void MarkerGen::process_vel_marker(unsigned int& idx, const std::vector<msgs::DetectedObject>& objs)
{
  std::vector<visualization_msgs::Marker>().swap(m_vel_.markers);
  m_vel_.markers.reserve(objs.size());

  for (const auto& obj : objs)
  {
    geometry_msgs::Point point = text_marker_position(obj.bPoint.p0, obj.bPoint.p6, 0.);
    m_vel_.markers.push_back(
        create_vel_marker(idx++, point, obj.track.absolute_velocity.x, obj.track.absolute_velocity.y, obj.header));
  }

  mc_.pub_vel.publish(m_vel_);
}

void MarkerGen::marker_gen_main(const std_msgs::Header header, const std::vector<msgs::DetectedObject>& objs,
                                MarkerConfig mc, std::vector<std::vector<PPLongDouble> >& ppss)
{
  set_config(mc, mc_);
  header_ = header;
  increase_uint(seq_);

  unsigned int idx = 1;

  process_text_marker(idx, objs);
  process_box_marker(idx, objs);

  if (mc_.show_pp > 0)
  {
    process_pp_marker(idx, objs, ppss);
  }

  process_vel_marker(idx, objs);
}
}  // namespace tpp