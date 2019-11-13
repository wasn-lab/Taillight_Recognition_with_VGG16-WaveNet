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

  void marker_gen_main(const std_msgs::Header header, const std::vector<msgs::DetectedObject>& objs, MarkerConfig mc,
                       std::vector<std::vector<PPLongDouble> >& ppss);

private:
  DISALLOW_COPY_AND_ASSIGN(MarkerGen);

  std_msgs::Header header_;
  unsigned int seq_ = 0;
  MarkerConfig mc_;

  const std::vector<unsigned int> box_order = { 0, 1, 1, 2, 2, 3, 3, 0,

                                                4, 5, 5, 6, 6, 7, 7, 4,

                                                0, 4, 1, 5, 2, 6, 3, 7 };

  visualization_msgs::MarkerArray m_id_;
  visualization_msgs::MarkerArray m_speed_;
  visualization_msgs::MarkerArray m_delay_;

  visualization_msgs::MarkerArray m_polygon_;
  visualization_msgs::MarkerArray m_box_;
  visualization_msgs::MarkerArray m_pp_;

  geometry_msgs::Point init_Point(const double x, const double y, const double z);

  geometry_msgs::Point text_marker_position(const msgs::BoxPoint bbox);

  void set_marker_attr(visualization_msgs::Marker& marker, const geometry_msgs::Point point);

  visualization_msgs::Marker create_seq_marker(const unsigned int idx, const geometry_msgs::Point point);

  visualization_msgs::Marker create_box_marker(const unsigned int idx, const msgs::BoxPoint bbox,
                                               std_msgs::Header obj_header);

  visualization_msgs::Marker create_polygon_marker(const unsigned int idx, const msgs::ConvexPoint& cPoint,
                                                   std_msgs::Header obj_header);

  std::string parse_class_id(unsigned int class_id);
  std::string parse_source_id(unsigned int source_id);

  visualization_msgs::Marker create_trackid_marker(const unsigned int idx, const geometry_msgs::Point point,
                                                   const msgs::DetectedObject& obj);

  visualization_msgs::Marker create_speed_marker(const unsigned int idx, const geometry_msgs::Point point,
                                                 std_msgs::Header obj_header, const float relspeed,
                                                 const float absspeed);

  visualization_msgs::Marker create_delay_marker(const unsigned int idx, const geometry_msgs::Point point,
                                                 std_msgs::Header obj_header);

  visualization_msgs::Marker create_pp_marker(const unsigned int idx, const float x, const float y,
                                              std_msgs::Header obj_header, const PPLongDouble pp,
                                              const unsigned int forecast_seq);

  void process_text_marker(unsigned int& idx, const std::vector<msgs::DetectedObject>& objs);

  void process_box_marker(unsigned int& idx, const std::vector<msgs::DetectedObject>& objs);

  void process_polygon_marker(unsigned int& idx, const std::vector<msgs::DetectedObject>& objs);

  void process_pp_marker(unsigned int& idx, const std::vector<msgs::DetectedObject>& objs,
                         std::vector<std::vector<PPLongDouble> >& ppss);
};
}  // namespace tpp

#endif  // __MARKER_GEN_H__
