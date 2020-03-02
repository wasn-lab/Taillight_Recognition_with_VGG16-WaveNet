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
  for (unsigned i = 0; i < box_order.size(); i++)
  {
    geometry_msgs::Point point_msg;
    convert_MyPoint32_to_Point(point_msg, point_list[box_order[i]]);
    marker.points.push_back(point_msg);
  }

  return marker;
}

visualization_msgs::Marker MarkerGen::create_polygon_marker(const unsigned int idx, const msgs::ConvexPoint& cPoint,
                                                            std_msgs::Header obj_header)
{
  visualization_msgs::Marker marker;

#if SAME_OBJ_MARKER_HEADER
  marker.header = header_;
#else
  marker.header = obj_header;
#endif
  marker.ns = "PPOutput_p";
  marker.action = visualization_msgs::Marker::ADD;
  marker.pose.orientation.x = 0.0;
  marker.pose.orientation.y = 0.0;
  marker.pose.orientation.z = 0.0;
  marker.pose.orientation.w = 1.0;

  marker.id = idx;
  marker.type = visualization_msgs::Marker::LINE_STRIP;

  if (cPoint.lowerAreaPoints.size() <= 4)
  {
    marker.scale.x = 0.35;
    marker.scale.y = 0.35;
    marker.scale.z = 0.7;
  }
  else
  {
    marker.scale.x = 0.17;
    marker.scale.y = 0.17;
    marker.scale.z = 0.4;
  }
  marker.lifetime = ros::Duration(mc_.lifetime_sec);
  set_ColorRGBA(marker.color, mc_.color);

  marker.points.reserve(cPoint.lowerAreaPoints.size());

  if (!cPoint.lowerAreaPoints.empty())
  {
    for (unsigned i = 0; i < cPoint.lowerAreaPoints.size(); i++)
    {
      geometry_msgs::Point p;
      convert_MyPoint32_to_Point(p, cPoint.lowerAreaPoints[i]);
      marker.points.push_back(p);
    }
    geometry_msgs::Point p;
    convert_MyPoint32_to_Point(p, cPoint.lowerAreaPoints[0]);
    marker.points.push_back(p);
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
    ss << "rel:";
  }
  ss << std::setprecision(1) << std::fixed << relspeed << "km/h";

  if (mc_.show_absspeed)
  {
    ss << std::endl << "abs:" << std::setprecision(1) << std::fixed << absspeed << "km/h";
  }
  std::string str = ss.str();
  marker.text = str;

  set_marker_attr(marker, point);

  return marker;
}

visualization_msgs::Marker MarkerGen::create_delay_marker(const unsigned int idx, const geometry_msgs::Point point,
                                                          std_msgs::Header obj_header)
{
  visualization_msgs::Marker marker;

#if SAME_OBJ_MARKER_HEADER
  marker.header = header_;
#else
  marker.header = obj_header;
#endif
  marker.ns = "PPOutput_d";
  marker.action = visualization_msgs::Marker::ADD;
  marker.id = idx;
  marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
  // marker.scale.x = 10.0;
  // marker.scale.y = 1.0;
  marker.scale.z = 1.7;

  std::ostringstream ss;
  double start_time_sec = header_.stamp.toSec();
  double delaytime_sec = mc_.module_pubtime_sec - start_time_sec;
  ss << std::fixed << std::setprecision(1) << delaytime_sec * 1000.0 << "ms";
  std::string str = ss.str();
  marker.text = str;

#if DEBUG
  LOG_INFO << std::fixed << "Start_Time: " << start_time_sec << "  Module_PubTime: " << mc_.module_pubtime_sec
           << "  Delay_Time: " << delaytime_sec << std::endl;
#endif

  set_marker_attr(marker, point);

  return marker;
}

visualization_msgs::Marker MarkerGen::create_pp_marker1(const unsigned int idx, const msgs::PointXY pos,
                                                        std_msgs::Header obj_header, const PPLongDouble pp,
                                                        const unsigned int forecast_seq, const float abs_speed_kmph)
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

  double scale = abs_speed_kmph * (forecast_seq + 1) / 36.;
  marker.scale.x = scale;
  marker.scale.y = scale / 2;
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

visualization_msgs::Marker MarkerGen::create_pp_marker2(const unsigned int idx, const msgs::PointXY pos,
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
  marker.pose.position.z = 0.;

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
  std::vector<visualization_msgs::Marker>().swap(m_delay_.markers);

  m_id_.markers.reserve(objs.size());
  m_speed_.markers.reserve(objs.size());
  m_delay_.markers.reserve(objs.size());

  for (unsigned i = 0; i < objs.size(); i++)
  {
    geometry_msgs::Point point = text_marker_position(objs[i].bPoint.p1, objs[i].bPoint.p2, 2.);
    m_id_.markers.push_back(create_trackid_marker(idx++, point, objs[i]));
    m_speed_.markers.push_back(create_speed_marker(idx++, point, objs[i].header, objs[i].relSpeed, objs[i].absSpeed));
    m_delay_.markers.push_back(create_delay_marker(idx++, point, objs[i].header));
  }

  geometry_msgs::Point point_seq = init_Point(-20, 0, 0);
  m_id_.markers.push_back(create_seq_marker(idx++, point_seq));

  mc_.pub_id.publish(m_id_);
  mc_.pub_speed.publish(m_speed_);
  mc_.pub_delay.publish(m_delay_);
}

void MarkerGen::process_box_marker(unsigned int& idx, const std::vector<msgs::DetectedObject>& objs)
{
  std::vector<visualization_msgs::Marker>().swap(m_box_.markers);
  m_box_.markers.reserve(objs.size());

  for (unsigned i = 0; i < objs.size(); i++)
  {
    m_box_.markers.push_back(create_box_marker(idx++, objs[i].bPoint, objs[i].header));
  }

  mc_.pub_bbox.publish(m_box_);
}

void MarkerGen::process_polygon_marker(unsigned int& idx, const std::vector<msgs::DetectedObject>& objs)
{
  std::vector<visualization_msgs::Marker>().swap(m_polygon_.markers);
  m_polygon_.markers.reserve(objs.size());

  for (unsigned i = 0; i < objs.size(); i++)
  {
    m_polygon_.markers.push_back(create_polygon_marker(idx++, objs[i].cPoint, objs[i].header));
  }

  mc_.pub_polygon.publish(m_polygon_);
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
          m_pp_.markers.push_back(create_pp_marker2(idx++, objs[i].track.forecasts[j].position, objs[i].header));
        }
        else if (mc_.show_pp == 2 || mc_.show_pp == 3)
        {
          m_pp_.markers.push_back(create_pp_marker1(idx++, objs[i].track.forecasts[j].position, objs[i].header,
                                                    ppss[i][j], j, objs[i].absSpeed));
        }
      }

      if (mc_.show_pp == 3)
      {
        for (unsigned int j = num_forecasts_; j < num_forecasts_ * 5; j++)
        {
          m_pp_.markers.push_back(create_pp_marker2(idx++, objs[i].track.forecasts[j].position, objs[i].header));
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

  for (unsigned i = 0; i < objs.size(); i++)
  {
    geometry_msgs::Point point = text_marker_position(objs[i].bPoint.p0, objs[i].bPoint.p6, 0.);
    m_vel_.markers.push_back(create_vel_marker(idx++, point, objs[i].track.absolute_velocity.x,
                                               objs[i].track.absolute_velocity.y, objs[i].header));
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

  process_polygon_marker(idx, objs);

  if (mc_.show_pp > 0)
  {
    process_pp_marker(idx, objs, ppss);
  }

  process_vel_marker(idx, objs);
}
}  // namespace tpp