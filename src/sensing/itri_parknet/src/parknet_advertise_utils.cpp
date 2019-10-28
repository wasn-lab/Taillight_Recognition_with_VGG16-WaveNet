/*
   CREATER: ICL U300
   DATE: May, 2019
 */
#include <cmath>
#include <glog/logging.h>
#include <opencv2/core.hpp>
#include "camera_params.h"
#include "camera_utils.h"
#include "camera_distance_mapper.h"
#include "parknet_advertise_utils.h"
#include "parknet_camera.h"
#include "parknet_args_parser.h"
#include "rect_class_score.h"
#include "msgs/ParkingSlotResult.h"
#include "msgs/ParkingSlot.h"
#include "msgs/MarkingPoint.h"

namespace parknet
{
/*
 * Return the quadrant of |mp|.
 *
 *            ^ x
 *            |
 *       I    |   IV
 *            |
 *  y <---------------
 *            |
 *      II    |   III
 *            |
 *
 *
 **/
static int quadrant_of(const msgs::MarkingPoint& mp)
{
  if ((mp.x >= 0) && (mp.y >= 0))
  {
    return 1;
  }
  else if ((mp.x < 0) && (mp.y >= 0))
  {
    return 2;
  }
  else if ((mp.x < 0) && (mp.y < 0))
  {
    return 3;
  }
  else
  {
    return 4;
  }
}

static double clockwise_angle_of(const msgs::MarkingPoint& mp)
{
  return -1.0 * atan2(mp.x, -1 * mp.y);
}

static bool compare_marking_points(const msgs::MarkingPoint& mp1, const msgs::MarkingPoint& mp2)
{
  int q1 = quadrant_of(mp1);
  int q2 = quadrant_of(mp2);
  if (q1 != q2)
  {
    if (q1 == 2)
    {
      return true;
    }
    else if ((q1 == 3) && (q2 != 2))
    {
      return true;
    }
    else if ((q1 == 4) && ((q2 != 2) && (q2 != 3)))
    {
      return true;
    }
    else
    {
      return false;
    }
  }
  return clockwise_angle_of(mp1) < clockwise_angle_of(mp2);
}

bool is_marking_point_near_image_border(const int cam_id, const msgs::MarkingPoint& mp)
{
  if (cam_id == camera::right_120_e)
  {
    if (mp.x <= -8.0)
    {
      return true;
    }
    else if (std::abs(mp.x) <= 0.05)
    {
      return true;
    }
    else
    {
      return false;
    }
  }
  else if (cam_id == camera::front_120_e)
  {
    return bool(mp.x < 1e-8);
  }
  else
  {
    return false;
  }
}

static bool all_corners_at_right_half_image(const std::vector<RectClassScore<float> >& corners)
{
  int im_x = 0, im_y = 0;
  for (const auto& corner : corners)
  {
    convert_corner_to_yolov3_image_xy(corner, &im_x, &im_y);
    if (im_x <= ::camera::yolov3_image_center_x)
    {
      return false;
    }
  }
  return true;
}

static bool all_corners_at_left_half_image(const std::vector<RectClassScore<float> >& corners)
{
  int im_x = 0, im_y = 0;
  for (const auto& corner : corners)
  {
    convert_corner_to_yolov3_image_xy(corner, &im_x, &im_y);
    if (im_x >= ::camera::yolov3_image_center_x)
    {
      return false;
    }
  }
  return true;
}

/**
 * Construct ParkingSlotResult, given >=5 corners.
 *
 * @param[in] cam_id
 * @param[in] corners
 * @param[in] dist_mapper
 */
msgs::ParkingSlotResult convert_5_or_more_corners_to_parking_slot_result(
    const int cam_id, const std::vector<RectClassScore<float> >& corners, const CameraDistanceMapper& dist_mapper)
{
  msgs::ParkingSlotResult res;
  const auto num_corners = corners.size();
  if (num_corners <= 3)
  {
    return res;
  }
  LOG(INFO) << parknet::camera_names[cam_id] << ": corners: " << num_corners;
  for (size_t i0 = 0; i0 < num_corners; i0++)
  {
    for (size_t i1 = i0 + 1; i1 < num_corners; i1++)
    {
      for (size_t i2 = i1 + 1; i2 < num_corners; i2++)
      {
        for (size_t i3 = i2 + 1; i3 < num_corners; i3++)
        {
          msgs::ParkingSlot pslot;
          pslot.marking_points.emplace_back(convert_corner_to_marking_point(corners[i0], dist_mapper));
          pslot.marking_points.emplace_back(convert_corner_to_marking_point(corners[i1], dist_mapper));
          pslot.marking_points.emplace_back(convert_corner_to_marking_point(corners[i2], dist_mapper));
          pslot.marking_points.emplace_back(convert_corner_to_marking_point(corners[i3], dist_mapper));
          pslot = sort_corners_couterclockwise(pslot);
          print_parking_slot(pslot, parknet::camera_names[cam_id] + ": possible parking slot: ");
          if (is_valid_parking_slot(pslot))
          {
            res.parking_slots.emplace_back(pslot);
            print_parking_slot(pslot, parknet::camera_names[cam_id] + ": valid parking slot: ");
          }
        }
      }
    }
  }

  return res;
}

/**
 * Construct ParkingSlotResult, given 4 corners.
 * When there are 4 corners, we can further relax the assumption of parking
 * slot size to tell vehicle control where to park.
 *
 * @param[in] cam_id
 * @param[in] corners
 * @param[in] dist_mapper
 */
msgs::ParkingSlotResult convert_4_corners_to_parking_slot_result(const int cam_id,
                                                                 const std::vector<RectClassScore<float> >& corners,
                                                                 const CameraDistanceMapper& dist_mapper)
{
  msgs::ParkingSlotResult res;
  const auto num_corners = corners.size();
  msgs::ParkingSlot pslot;
  if (num_corners != 4)
  {
    return res;
  }
  LOG(INFO) << parknet::camera_names[cam_id] << ": corners: " << num_corners;
  for (size_t i = 0; i < 4; i++)
  {
    pslot.marking_points.emplace_back(convert_corner_to_marking_point(corners[i], dist_mapper));
  }
  for (int i = 0; i < 4; i++)
  {
    if (is_marking_point_near_image_border(cam_id, pslot.marking_points[i]))
    {
      VLOG(1) << parknet::camera_names[cam_id] << ": cannot rely on marking points near image borders: ("
              << pslot.marking_points[i].x << ", " << pslot.marking_points[i].y << ")";
      return res;
    }
  }

  pslot = sort_corners_couterclockwise(pslot);
  print_parking_slot(pslot, parknet::camera_names[cam_id] + ": possible parking slot: ");
  if (is_valid_parking_slot(pslot))
  {
    // Good case
    res.parking_slots.emplace_back(pslot);
    print_parking_slot(pslot, parknet::camera_names[cam_id] + ": valid parking slot: ");
    return res;
  }
  // Distance estimation may be unreliable if corners are close to image border.
  // In this case, we allow more margin of errors.
  bool is_near_border = false;
  for (size_t i = 0; i < 4; i++)
  {
    if (is_marking_point_near_image_border(cam_id, pslot.marking_points[i]))
    {
      is_near_border = true;
      break;
    }
  }
  if (is_near_border && all_corners_at_right_half_image(corners))
  {
    if ((cam_id == camera::right_120_e) || (cam_id == camera::front_120_e))
    {
      res.parking_slots.emplace_back(pslot);
      print_parking_slot(pslot, parknet::camera_names[cam_id] + ": valid parking slot: ");
      return res;
    }
  }

  return res;
}

/**
 * Construct ParkingSlotResult, given 3 corners.
 *
 * @param[in] cam_id
 * @param[in] corners
 * @param[in] dist_mapper
 */
msgs::ParkingSlotResult convert_3_corners_to_parking_slot_result(const int cam_id,
                                                                 const std::vector<RectClassScore<float> >& corners,
                                                                 const CameraDistanceMapper& dist_mapper)
{
  msgs::ParkingSlotResult res;
  const auto num_corners = corners.size();
  if (num_corners != 3)
  {
    return res;
  }
  msgs::ParkingSlot pslot;
  msgs::MarkingPoint mps[3];
  mps[0] = convert_corner_to_marking_point(corners[0], dist_mapper);
  mps[1] = convert_corner_to_marking_point(corners[1], dist_mapper);
  mps[2] = convert_corner_to_marking_point(corners[2], dist_mapper);
  bool succ = convert_3_marking_points_to_parking_slot(mps, &pslot);
  if (!succ)
  {
    return res;
  }
  pslot = sort_corners_couterclockwise(pslot);
  pslot.type = 0;
  res.parking_slots.emplace_back(pslot);
  return res;
}

/**
 * Construct ParkingSlotResult in front_120 camera, given 2 corners.
 * The parking slot is too large to fit into the front_120 image. In this
 case, deduce the other two corners and form a parking slot.
 *
 * @param[in] cam_id
 * @param[in] corners
 * @param[in] dist_mapper
 */
msgs::ParkingSlotResult convert_2_corners_to_parking_slot_result_in_front_120(
    const int cam_id, const std::vector<RectClassScore<float> >& corners, const CameraDistanceMapper& dist_mapper)
{
  msgs::ParkingSlotResult res;

  if (cam_id != camera::front_120_e)
  {
    LOG(WARNING) << __FUNCTION__ << " only works for front_120 camera. Your cam_id is " << cam_id;
  }
  const auto num_corners = corners.size();
  int extend_to_y_plus = 0;
  if (num_corners != 2)
  {
    return res;
  }

  msgs::ParkingSlot pslot;
  msgs::MarkingPoint mps[4];
  mps[0] = convert_corner_to_marking_point(corners[0], dist_mapper);
  mps[1] = convert_corner_to_marking_point(corners[1], dist_mapper);

  for (int i = 0; i < 2; i++)
  {
    if (is_marking_point_near_image_border(cam_id, mps[i]))
    {
      VLOG(2) << camera_names[cam_id] << ": Ignore near border marking point: (" << mps[i].x << ", " << mps[i].y << ")";
      return res;
    }
  }

  if (all_corners_at_right_half_image(corners))
  {
    extend_to_y_plus = -1;
  }
  else if (all_corners_at_left_half_image(corners))
  {
    extend_to_y_plus = 1;
  }

  if (extend_to_y_plus == 0)
  {
    VLOG(2) << camera_names[cam_id] << ": cannot decide where to extend in y direction.";
    return res;
  }

  const float dist = euclean_distance(mps[0], mps[1]);
  float infered_edge_length = 0;
  if (is_valid_short_edge_length(dist))
  {
    infered_edge_length = long_edge_in_meters;
  }
  else if (is_valid_long_edge_length(dist))
  {
    infered_edge_length = short_edge_in_meters;
  }
  else
  {
    LOG(INFO) << "Invalid edge length: " << dist << "(" << mps[0].x << ", " << mps[0].y << "),"
              << "(" << mps[1].x << ", " << mps[1].y << ")";
    return res;
  }

  // OK, the corners can form a parking slot.
  for (int i = 0; i < 2; i++)
  {
    const auto j = 2 + i;
    mps[j].x = mps[i].x;
    mps[j].y = mps[i].y + infered_edge_length * extend_to_y_plus;
    mps[j].z = mps[i].z;
  }

  for (int i = 0; i < 4; i++)
  {
    pslot.marking_points.emplace_back(mps[i]);
  }

  pslot = sort_corners_couterclockwise(pslot);
  pslot.type = 0;  // TODO: fill in the right type.
  res.parking_slots.emplace_back(pslot);
  print_parking_slot(pslot, "front_120 uses two corners to find a parking slot: ");
  return res;
}

/**
 * Construct ParkingSlotResult in right_120 camera, given 2 corners.
 *
 * @param[in] cam_id
 * @param[in] corners
 * @param[in] dist_mapper
 */
msgs::ParkingSlotResult convert_2_corners_to_parking_slot_result_in_right_120(
    const int cam_id, const std::vector<RectClassScore<float> >& corners, const CameraDistanceMapper& dist_mapper)
{
  msgs::ParkingSlotResult res;

  if (cam_id != camera::right_120_e)
  {
    LOG(WARNING) << __FUNCTION__ << " only works for right_120 camera. Your cam_id is " << cam_id;
  }
  const auto num_corners = corners.size();
  if (num_corners != 2)
  {
    return res;
  }

  msgs::ParkingSlot pslot;
  msgs::MarkingPoint mps[4];
  if (corners[0].x < corners[1].x)
  {
    mps[0] = convert_corner_to_marking_point(corners[0], dist_mapper);
    mps[1] = convert_corner_to_marking_point(corners[1], dist_mapper);
  }
  else
  {
    mps[0] = convert_corner_to_marking_point(corners[1], dist_mapper);
    mps[1] = convert_corner_to_marking_point(corners[0], dist_mapper);
  }

  if (is_marking_point_near_image_border(cam_id, mps[0]) || is_marking_point_near_image_border(cam_id, mps[1]))
  {
    VLOG(2) << "Cannot form a valid parking slot due to reaching the limit of distance estimation: "
            << "(" << mps[0].x << ", " << mps[0].y << "),"
            << "(" << mps[1].x << ", " << mps[1].y << ")";
    return res;
  }

  const float dist = euclean_distance(mps[0], mps[1]);
  if (!is_valid_short_edge_length(dist))
  {
    LOG(INFO) << camera_names[cam_id] << ": Invalid short edge length: " << dist << "(" << mps[0].x << ", " << mps[0].y
              << "),"
              << "(" << mps[1].x << ", " << mps[1].y << ") ";
    return res;
  }
  // OK, the corners can form a parking slot.
  const float unit_orth_vec_y = -(mps[1].x - mps[0].x) / dist;
  const float unit_orth_vec_x = (mps[1].y - mps[0].y) / dist;

  // The deduced marking points are in the direction of y- relative to mps[0..1].
  float direction = -1;
  if (unit_orth_vec_y < 0)
  {
    direction = 1;
  }
  for (int i = 2; i < 4; i++)
  {
    const auto j = i - 2;
    mps[i].x = mps[j].x + unit_orth_vec_x * long_edge_in_meters * direction;
    mps[i].y = mps[j].y + unit_orth_vec_y * long_edge_in_meters * direction;
    mps[i].z = mps[j].z;
  }

  for (int i = 0; i < 4; i++)
  {
    pslot.marking_points.emplace_back(mps[i]);
  }

  pslot = sort_corners_couterclockwise(pslot);
  pslot.type = 0;
  res.parking_slots.emplace_back(pslot);
  print_parking_slot(pslot, "right_120 uses two corners to find a parking slot: ");
  return res;
}

/**
 * Given 3 marking points, form a valid parking slot if possible.
 *
 * @param[in] mps 3 marking points
 * @param[out] pslot
 * @return true if a valid parking slot is found. false otherwise.
 *
 **/
bool convert_3_marking_points_to_parking_slot(const msgs::MarkingPoint mps[], msgs::ParkingSlot* pslot)
{
  assert(pslot->marking_points.size() == 0);
  msgs::MarkingPoint mp4;
  msgs::ParkingSlot temp_pslot;
  bool found = infer_4th_marking_point(mps, &mp4);

  if (not found)
  {
    return false;
  }
  for (int i = 0; i < 3; i++)
  {
    temp_pslot.marking_points.emplace_back(mps[i]);
  }
  temp_pslot.marking_points.emplace_back(mp4);
  temp_pslot = sort_corners_couterclockwise(temp_pslot);
  if (!is_valid_parking_slot(temp_pslot))
  {
    return false;
  }
  for (int i = 0; i < 4; i++)
  {
    pslot->marking_points.emplace_back(temp_pslot.marking_points[i]);
  }
  pslot->type = 0;
  return true;
}

/**
 * Given 2 marking points, form a valid parking slot if possible.
 *
 * This function is used when the car is backing into the parking slot,
 * each camera has only a parital view of the slot. The
 * left-hand-side camera sees only one corner, so does the right-hand-side one.
 * Under this circumstance, parknet still needs to send the parking slot
 * coordinates to the vehicle control module to maneuver to the final position.
 *       x
 *       ^
 *       |
 * y <---+
 *
 *   left mp   right mp
 *        +----+
 *        |    |
 *        |    |
 *        |    | 9m
 *        |    |
 *        +----+
 *         3.6m
 * @param[in] mps 2 marking points
 * @param[out] pslot
 * @return true if a valid parking slot is found. false otherwise.
 *
 **/
bool convert_2_marking_points_to_parking_slot(const msgs::MarkingPoint mps[], msgs::ParkingSlot* pslot)
{
  float edge_length = euclean_distance(mps[0], mps[1]);
  msgs::MarkingPoint mp_bottom_left, mp_bottom_right;
  int left = -1;
  int right = -1;
  if (!is_valid_short_edge_length(edge_length))
  {
    return false;
  }

  if (mps[0].y > mps[1].y)
  {
    // mps[0] is to the left-hand-side of mps[1]
    left = 0;
  }
  else
  {
    left = 1;
  }
  right = 1 - left;

  // orthogonal vector for mps[left] -> mps[right]
  const float vec_x = mps[right].x - mps[left].x;
  const float vec_y = mps[right].y - mps[left].y;

  // The inference mps should have less x.
  float direction = 1.0;
  if (vec_y < 0)
  {
    direction = -1.0;
  }

  const float orth_vec_x = -vec_y * (9.0 / edge_length) * direction;  // orth_vec has length 9.0m
  const float orth_vec_y = vec_x * (9.0 / edge_length) * direction;
  pslot->marking_points.emplace_back(mps[right]);
  pslot->marking_points.emplace_back(mps[left]);
  mp_bottom_left.x = mps[left].x + orth_vec_x;
  mp_bottom_left.y = mps[left].y + orth_vec_y;
  mp_bottom_right.x = mps[right].x + orth_vec_x;
  mp_bottom_right.y = mps[right].y + orth_vec_y;
  pslot->marking_points.emplace_back(mp_bottom_left);
  pslot->marking_points.emplace_back(mp_bottom_right);
  return true;
}

/**
 * Given 3 marking points, find the 4th marking point.
 *
 * @param[in] mps[]
 * @param[out] mp_out
 * @return true: successfully find the 4th marking point. false otherwise.
 **/
bool infer_4th_marking_point(const msgs::MarkingPoint mps[], msgs::MarkingPoint* mp_out)
{
  bool found = false;
  int right_angle_index = -1;

  // Find the right angle point
  for (int i = 0; i < 3; i++)
  {
    const int i1 = (i + 1) % 3;
    const int i2 = (i + 2) % 3;
    float inner_product =
        (mps[i1].x - mps[i].x) * (mps[i1].y - mps[i].x) + (mps[i2].x - mps[i].x) * (mps[i2].y - mps[i].x);
    float length1 = euclean_distance(mps[i], mps[i1]);
    float length2 = euclean_distance(mps[i], mps[i2]);
    float cos_theta = inner_product / (length1 * length2);
    // Mathematically, cos_theta = 0 if and only if mps[i] is a right angle point.
    // Here we allow a degree of error.
    if (std::abs(cos_theta) <= 0.1)
    {
      found = true;
      right_angle_index = i;
      break;
    }
  }

  if (found)
  {
    const int i1 = (right_angle_index + 1) % 3;
    const int i2 = (right_angle_index + 2) % 3;
    mp_out->x = mps[i1].x + mps[i2].x - mps[right_angle_index].x;
    mp_out->y = mps[i1].y + mps[i2].y - mps[right_angle_index].y;
  }
  return bool(found);
}

int print_parking_slot(const msgs::ParkingSlot& pslot, const std::string& prefix)
{
  assert(pslot.marking_points.size() == 4);
  const auto mps = pslot.marking_points;
  LOG(INFO) << prefix << cv::format("(%.2f, %.2f)", mps[0].x, mps[0].y)
            << cv::format(", (%.2f, %.2f)", mps[1].x, mps[1].y) << cv::format(", (%.2f, %.2f)", mps[2].x, mps[2].y)
            << cv::format(", (%.2f, %.2f)", mps[3].x, mps[3].y);
  return 0;
}

bool is_valid_parking_slot(const msgs::ParkingSlot& pslot)
{
  return is_valid_short_edge_length(short_edge_length_in_meters(pslot)) &&
         is_valid_long_edge_length(long_edge_length_in_meters(pslot));
}

bool is_valid_edge_length(const float edge_length)
{
  return is_valid_short_edge_length(edge_length) || is_valid_long_edge_length(edge_length);
}

bool is_valid_short_edge_length(const float edge_length)
{
  return (std::abs(edge_length - short_edge_in_meters) <= edge_length_error_in_meters);
}

bool is_valid_long_edge_length(const float edge_length)
{
  return (std::abs(edge_length - long_edge_in_meters) <= edge_length_error_in_meters);
}

float short_edge_length_in_meters(const msgs::ParkingSlot& pslot)
{
  assert(pslot.marking_points.size() == 4);
  float min_dist = 1000000.0;
  for (int i = 0; i < 4; i++)
  {
    const auto next_i = (i + 1) % 4;
    const auto& cur_mp = pslot.marking_points[i];
    const auto& next_mp = pslot.marking_points[next_i];
    const float dist = euclean_distance(cur_mp, next_mp);
    if (dist < min_dist)
    {
      min_dist = dist;
    }
  }
  return min_dist;
}

float long_edge_length_in_meters(const msgs::ParkingSlot& pslot)
{
  assert(pslot.marking_points.size() == 4);
  float max_dist = 0;
  for (int i = 0; i < 4; i++)
  {
    const auto next_i = (i + 1) % 4;
    const auto& cur_mp = pslot.marking_points[i];
    const auto& next_mp = pslot.marking_points[next_i];
    const float dist = euclean_distance(cur_mp, next_mp);
    if (dist > max_dist)
    {
      max_dist = dist;
    }
  }
  return max_dist;
}

float area_of_parking_slot(const msgs::ParkingSlot& pslot)
{
  assert(pslot.marking_points.size() == 4);
  // TODO: relax the assumption of rectangle parking lot.
  return short_edge_length_in_meters(pslot) * long_edge_length_in_meters(pslot);
}

/**
 * Sort marking_points of |pslot| in counterclockwise order. The sorted result is
 *
 *       x
 *       ^
 *       |
 * y <---+
 *
 *   p3          p2
 *    +----------+
 *    |          |
 *    |          |
 *    +----------+
 *   p0          p1
 *
 * @return sorted pslot with marking_points = {p0, p1, p2, p3}.
 */
msgs::ParkingSlot sort_corners_couterclockwise(const msgs::ParkingSlot& pslot)
{
  assert(pslot.marking_points.size() == 4);

  msgs::ParkingSlot res;
  const auto centroid = parking_slot_centroid(pslot);

  for (int i = 0; i < 4; i++)
  {
    VLOG(2) << "point " << i << ": " << pslot.marking_points[i].x << " " << pslot.marking_points[i].y << " "
            << pslot.marking_points[i].z;
  }
  for (int i = 0; i < 4; i++)
  {
    res.marking_points.push_back(pslot.marking_points[i]);
    res.marking_points[i].x -= centroid.x;
    res.marking_points[i].y -= centroid.y;
    res.marking_points[i].z -= centroid.z;
  }

  std::sort(res.marking_points.begin(), res.marking_points.end(), compare_marking_points);
  for (int i = 0; i < 4; i++)
  {
    res.marking_points[i].x += centroid.x;
    res.marking_points[i].y += centroid.y;
    res.marking_points[i].z += centroid.z;
    VLOG(2) << "res " << i << ": " << res.marking_points[i].x << " " << res.marking_points[i].y << " "
            << res.marking_points[i].z;
  }

  res.id = pslot.id;
  res.type = pslot.type;

  return res;
}

msgs::MarkingPoint parking_slot_centroid(const msgs::ParkingSlot& pslot)
{
  msgs::MarkingPoint centroid;
  float sum_x = 0, sum_y = 0, sum_z = 0;
  for (int i = 0; i < 4; i++)
  {
    sum_x += pslot.marking_points[i].x;
    sum_y += pslot.marking_points[i].y;
    sum_z += pslot.marking_points[i].z;
  }
  centroid.x = sum_x / 4;
  centroid.y = sum_y / 4;
  centroid.z = sum_z / 4;
  return centroid;
}

/**
 * X-axis compensation between distance estimation and vehicle control.
 */
double get_sx_compensation(const int cam_id)
{
  if (cam_id == front_120_e)
  {
    return get_front_120_sx_compensation();
  }
  if (cam_id == left_120_e)
  {
    return get_left_120_sx_compensation();
  }
  if (cam_id == right_120_e)
  {
    return get_right_120_sx_compensation();
  }
  return 0;
}

/**
 * Y-axis compensation between distance estimation and vehicle control.
 */
double get_sy_compensation(const int cam_id)
{
  if (cam_id == front_120_e)
  {
    return get_front_120_sy_compensation();
  }
  if (cam_id == left_120_e)
  {
    return get_left_120_sy_compensation();
  }
  if (cam_id == right_120_e)
  {
    return get_right_120_sy_compensation();
  }
  return 0;
}

/**
 * Map a corner in 608x608 image to 3D spatial coordinates.
 *
 * @param[in] corner with fields {x, y, w, h}
 * @param[in] dist_mapper
 * @return marking_point spatial {x, y, z}
 */
msgs::MarkingPoint convert_corner_to_marking_point(const RectClassScore<float>& corner,
                                                   const CameraDistanceMapper& dist_mapper)
{
  int im_yolov3_x = 0, im_yolov3_y = 0;
  int im_x = 0, im_y = 0;
  msgs::MarkingPoint mp;
  convert_corner_to_yolov3_image_xy(corner, &im_yolov3_x, &im_yolov3_y);
  ::camera::yolov3_to_camera_xy(im_yolov3_x, im_yolov3_y, &im_x, &im_y);

  dist_mapper.get_distance_raw_1920x1208(im_x, im_y, &mp.x, &mp.y, &mp.z);
  mp.z = -3.46;  // -3.46 meters from roottop of the car.

  return mp;
}

/*
 * Convert corner bounding box to (im_x, im_y) in yolov3 image.
 * @param[in] corner with fields {x, y, w, h}
 * @param[out] im_x
 * @param[out] im_y
 **/
int convert_corner_to_yolov3_image_xy(const RectClassScore<float>& corner, int* im_x, int* im_y)
{
  const int x = corner.x;
  const int y = corner.y;
  const int w = corner.w;
  const int h = corner.h;
  *im_x = x + w / 2;
  *im_y = y + h / 2;
  return 0;
}

float euclean_distance(float x1, float y1, float x2, float y2)
{
  const auto dx = x1 - x2;
  const auto dy = y1 - y2;

  return sqrt(dx * dx + dy * dy);
}

float euclean_distance(const msgs::MarkingPoint& p1, const msgs::MarkingPoint& p2)
{
  const auto dx = p1.x - p2.x;
  const auto dy = p1.y - p2.y;

  return sqrt(dx * dx + dy * dy);
}

};  // namespace parknet
