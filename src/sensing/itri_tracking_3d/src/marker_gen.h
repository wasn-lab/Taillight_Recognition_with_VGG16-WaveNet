#ifndef __MARKER_GEN_H__
#define __MARKER_GEN_H__

#include "tpp.h"
#include "utils.h"
#include <geometry_msgs/Point.h>
#include <visualization_msgs/MarkerArray.h>
#include <visualization_msgs/Marker.h>

namespace tpp
{
class MarkerGen
{
public:
  MarkerGen()
  {
  }
  ~MarkerGen()
  {
  }

  void marker_gen_main(const std_msgs::Header header, const std::vector<msgs::DetectedObject>& objs, MarkerConfig mc);

private:
  DISALLOW_COPY_AND_ASSIGN(MarkerGen);

  std_msgs::Header header_;
  unsigned int seq_ = 0;
  MarkerConfig mc_;

  const std::vector<unsigned int> box_order = { 0, 1, 1, 2, 2, 3, 3, 0,

                                                4, 5, 5, 6, 6, 7, 7, 4,

                                                0, 4, 1, 5, 2, 6, 3, 7 };

  geometry_msgs::Point init_Point(const double x, const double y, const double z);

  geometry_msgs::Point text_marker_position(const MyPoint32 p1, const MyPoint32 p2, const double z_offset);

  void set_marker_attr(visualization_msgs::Marker& marker, const geometry_msgs::Point point);

  // marker arrays
  visualization_msgs::MarkerArray m_id_;
  visualization_msgs::MarkerArray m_speed_;
  visualization_msgs::MarkerArray m_box_;
  visualization_msgs::MarkerArray m_vel_;

  // create markers
  visualization_msgs::Marker create_seq_marker(const unsigned int idx, const geometry_msgs::Point point);

  visualization_msgs::Marker create_box_marker(const unsigned int idx, const msgs::BoxPoint bbox,
                                               std_msgs::Header obj_header);

  std::string parse_class_id(unsigned int class_id);
  std::string parse_source_id(unsigned int source_id);

  visualization_msgs::Marker create_trackid_marker(const unsigned int idx, const geometry_msgs::Point point,
                                                   const msgs::DetectedObject& obj);

  visualization_msgs::Marker create_speed_marker(const unsigned int idx, const geometry_msgs::Point point,
                                                 std_msgs::Header obj_header, const float relspeed,
                                                 const float absspeed);

  visualization_msgs::Marker create_vel_marker(const unsigned int idx, const geometry_msgs::Point point, const float vx,
                                               const float vy, std_msgs::Header obj_header);

  // process markers
  void process_text_marker(unsigned int& idx, const std::vector<msgs::DetectedObject>& objs);

  void process_box_marker(unsigned int& idx, const std::vector<msgs::DetectedObject>& objs);

  void process_vel_marker(unsigned int& idx, const std::vector<msgs::DetectedObject>& objs);
};
}  // namespace tpp

#endif  // __MARKER_GEN_H__
