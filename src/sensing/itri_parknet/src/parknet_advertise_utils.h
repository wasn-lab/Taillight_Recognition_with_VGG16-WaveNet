/*
   CREATER: ICL U300
   DATE: May, 2019
 */

#ifndef __PARKNET_ADVERTISE_H__
#define __PARKNET_ADVERTISE_H__
#include "rect_class_score.h"
#include "msgs/MarkingPoint.h"
#include "msgs/ParkingSlot.h"
#include "msgs/ParkingSlotResult.h"
class CameraDistanceMapper;

namespace parknet
{
constexpr float short_edge_in_meters = 3.6;
constexpr float long_edge_in_meters = 9.0;
constexpr float edge_length_error_in_meters = 0.25;

// From corners to ParkingSlot
msgs::ParkingSlotResult convert_5_or_more_corners_to_parking_slot_result(
    const int cam_id, const std::vector<RectClassScore<float> >& corners, const CameraDistanceMapper& dist_mapper);
msgs::ParkingSlotResult convert_4_corners_to_parking_slot_result(const int cam_id,
                                                                 const std::vector<RectClassScore<float> >& corners,
                                                                 const CameraDistanceMapper& dist_mapper);
msgs::ParkingSlotResult convert_3_corners_to_parking_slot_result(const int cam_id,
                                                                 const std::vector<RectClassScore<float> >& corners,
                                                                 const CameraDistanceMapper& dist_mapper);
msgs::ParkingSlotResult convert_2_corners_to_parking_slot_result_in_front_120(
    const int cam_id, const std::vector<RectClassScore<float> >& corners, const CameraDistanceMapper& dist_mapper);

msgs::ParkingSlotResult convert_2_corners_to_parking_slot_result_in_right_120(
    const int cam_id, const std::vector<RectClassScore<float> >& corners, const CameraDistanceMapper& dist_mapper);

bool convert_3_marking_points_to_parking_slot(const msgs::MarkingPoint mps[], msgs::ParkingSlot* pslot);
bool convert_2_marking_points_to_parking_slot(const msgs::MarkingPoint mps[], msgs::ParkingSlot* pslot);
bool infer_4th_marking_point(const msgs::MarkingPoint mps[], msgs::MarkingPoint* mp_out);

// helper functions for ParkingSlot
int print_parking_slot(const msgs::ParkingSlot& pslot, const std::string& prefix);
bool is_valid_parking_slot(const msgs::ParkingSlot& pslot);
bool is_valid_edge_length(const float edge_length);
bool is_valid_short_edge_length(const float edge_length);
bool is_valid_long_edge_length(const float edge_length);
msgs::ParkingSlot sort_corners_couterclockwise(const msgs::ParkingSlot& pslot);
msgs::MarkingPoint parking_slot_centroid(const msgs::ParkingSlot& pslot);
float short_edge_length_in_meters(const msgs::ParkingSlot& pslot);
float long_edge_length_in_meters(const msgs::ParkingSlot& pslot);
float area_of_parking_slot(const msgs::ParkingSlot& pslot);
bool is_marking_point_near_image_border(const int cam_id, const msgs::MarkingPoint& mp);
double get_sx_compensation(const int cam_id);
double get_sy_compensation(const int cam_id);

// corner to marking_point
msgs::MarkingPoint convert_corner_to_marking_point(const RectClassScore<float>& corner,
                                                   const CameraDistanceMapper& dist_mapper);
int convert_corner_to_yolov3_image_xy(const RectClassScore<float>& corner, int* im_x, int* im_y);
float euclean_distance(float x1, float y1, float x2, float y2);
float euclean_distance(const msgs::MarkingPoint& p1, const msgs::MarkingPoint& p2);
};  // namespace parknet

#endif  // __PARKNET_ADVERTISE_H__
