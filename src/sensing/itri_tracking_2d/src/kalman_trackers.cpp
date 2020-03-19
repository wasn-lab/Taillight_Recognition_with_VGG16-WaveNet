#include "kalman_trackers.h"

namespace tpp
{
void KalmanTrackers::new_tracker(const msgs::DetectedObject& box, BoxCenter& box_center,
                                 const std::vector<BoxCorner>& box_corners)
{
  KalmanTracker track;

  track.box_ = box;

  track.id_ = box.track.id;

  track.tracktime_ = 1;

  track.kalman_.init(num_vars_, num_measures_, num_controls_, CV_32F);

  track.kalman_.transitionMatrix =
      (cv::Mat_<float>(num_vars_, num_vars_) << 1.f, 0.f, dt_, 0.f, half_dt_square_, 0.f,  //
       0.f, 1.f, 0.f, dt_, 0.f, half_dt_square_,                                           //
       0.f, 0.f, 1.f, 0.f, dt_, 0.f,                                                       //
       0.f, 0.f, 0.f, 1.f, 0.f, dt_,                                                       //
       0.f, 0.f, 0.f, 0.f, 1.f, 0.f,                                                       //
       0.f, 0.f, 0.f, 0.f, 0.f, 1.f);

  track.kalman_.processNoiseCov = (cv::Mat_<float>(num_vars_, num_vars_) << Q1, 0.f, 0.f, 0.f, 0.f, 0.f,  //
                                   0.f, Q1, 0.f, 0.f, 0.f, 0.f,                                           //
                                   0.f, 0.f, Q2, 0.f, 0.f, 0.f,                                           //
                                   0.f, 0.f, 0.f, Q2, 0.f, 0.f,                                           //
                                   0.f, 0.f, 0.f, 0.f, Q3, 0.f,                                           //
                                   0.f, 0.f, 0.f, 0.f, 0.f, Q3);

  cv::setIdentity(track.kalman_.measurementMatrix);
  cv::setIdentity(track.kalman_.measurementNoiseCov, cv::Scalar::all(R));
  cv::setIdentity(track.kalman_.errorCovPost, cv::Scalar::all(1.f));

  MyPoint32 p_rel;
  box_center.pos.get_point_rel(p_rel);

  track.kalman_.statePost.at<float>(0) = p_rel.x;
  track.kalman_.statePost.at<float>(1) = p_rel.y;
  track.kalman_.statePost.at<float>(2) = 0.f;
  track.kalman_.statePost.at<float>(3) = 0.f;
  track.kalman_.statePost.at<float>(4) = 0.f;
  track.kalman_.statePost.at<float>(5) = 0.f;

  track.hist_.header_ = box.header;

  track.hist_.set_for_first_element(track.id_, track.tracktime_, p_rel.x, p_rel.y,                               //
                                    track.kalman_.statePost.at<float>(0), track.kalman_.statePost.at<float>(1),  //
                                    track.kalman_.statePost.at<float>(2), track.kalman_.statePost.at<float>(3));

  track.box_center_ = box_center;
  track.box_corners_ = box_corners;

  tracks_.push_back(track);
}

void KalmanTrackers::set_time_displacement(const long long dt)
{
  dt_ = dt / 1000000000.f;

  half_dt_square_ = 0.5f * std::pow(dt_, 2);

  for (unsigned i = 0; i < tracks_.size(); i++)
  {
    tracks_[i].kalman_.transitionMatrix.at<float>(0, 2) = dt_;
    tracks_[i].kalman_.transitionMatrix.at<float>(0, 4) = half_dt_square_;

    tracks_[i].kalman_.transitionMatrix.at<float>(1, 3) = dt_;
    tracks_[i].kalman_.transitionMatrix.at<float>(1, 5) = half_dt_square_;

    tracks_[i].kalman_.transitionMatrix.at<float>(2, 4) = dt_;
    tracks_[i].kalman_.transitionMatrix.at<float>(3, 5) = dt_;
  }
}

void KalmanTrackers::extract_box_center(BoxCenter& box_center, const msgs::BoxPoint& box)
{
  box_center.id = 0;

  MyPoint32 p_rel;
  p_rel.x = 0.5f * (box.p0.x + box.p6.x);
  p_rel.y = 0.5f * (box.p0.y + box.p6.y);
  p_rel.z = 0.5f * (box.p0.z + box.p6.z);

  box_center.pos.set_point_rel(p_rel);

  box_center.x_length = euclidean_distance(box.p3.x - box.p0.x, box.p3.y - box.p0.y);
  box_center.y_length = euclidean_distance(box.p4.x - box.p0.x, box.p4.y - box.p0.y);
  box_center.z_length = box.p1.z - box.p0.z;

  float area_tmp = box_center.x_length * box_center.y_length;
  assign_value_cannot_zero(box_center.area, area_tmp);

  float volumn_tmp = box_center.area * box_center.z_length;
  assign_value_cannot_zero(box_center.volumn, volumn_tmp);
}

void KalmanTrackers::extract_box_centers()
{
  std::vector<BoxCenter>().swap(box_centers_);
  box_centers_.reserve(objs_.size());

  for (unsigned i = 0; i < objs_.size(); i++)
  {
    BoxCenter box_center;
    extract_box_center(box_center, objs_[i].bPoint);
    box_centers_.push_back(box_center);

    objs_[i].lidarInfo.boxCenter.x = (objs_[i].bPoint.p0.x + objs_[i].bPoint.p6.x) / 2;
    objs_[i].lidarInfo.boxCenter.y = (objs_[i].bPoint.p0.y + objs_[i].bPoint.p6.y) / 2;
    objs_[i].lidarInfo.boxCenter.z = (objs_[i].bPoint.p0.z + objs_[i].bPoint.p6.z) / 2;
  }
}

void KalmanTrackers::extract_box_corner(BoxCorner& box_corner, const MyPoint32& corner, const signed char order)
{
  float x_rel = corner.x;
  float y_rel = corner.y;
  float z_rel = corner.z;

  box_corner.id = 0;
  box_corner.order = order;

  box_corner.x_rel = x_rel;
  box_corner.y_rel = y_rel;
  box_corner.z_rel = z_rel;

  box_corner.new_x_rel = 0.f;
  box_corner.new_y_rel = 0.f;
  box_corner.new_z_rel = 0.f;
}

void KalmanTrackers::extract_box_corners_of_boxes()
{
  std::vector<std::vector<BoxCorner> >().swap(box_corners_of_boxes_);
  box_corners_of_boxes_.reserve(objs_.size());

  for (unsigned i = 0; i < objs_.size(); i++)
  {
    std::vector<BoxCorner> box_corners;
    box_corners.reserve(num_2dbox_corners);

    BoxCorner box_corner0;
    BoxCorner box_corner1;
    BoxCorner box_corner2;
    BoxCorner box_corner3;

    extract_box_corner(box_corner0, objs_[i].bPoint.p0, 0);
    extract_box_corner(box_corner1, objs_[i].bPoint.p3, 1);
    extract_box_corner(box_corner2, objs_[i].bPoint.p7, 2);
    extract_box_corner(box_corner3, objs_[i].bPoint.p4, 3);

    box_corners.push_back(box_corner0);
    box_corners.push_back(box_corner1);
    box_corners.push_back(box_corner2);
    box_corners.push_back(box_corner3);

    box_corners_of_boxes_.push_back(box_corners);
  }
}

void KalmanTrackers::correct_tracker(KalmanTracker& track, const float x_measure, const float y_measure)
{
  cv::Mat measurement = cv::Mat::zeros(num_measures_, 1, CV_32F);
  measurement.at<float>(0) = x_measure;
  measurement.at<float>(1) = y_measure;

  track.kalman_.correct(measurement);
}

void KalmanTrackers::update_associated_trackers()
{
  for (unsigned i = 0; i < objs_.size(); i++)
  {
    for (unsigned j = 0; j < tracks_.size(); j++)
    {
      if (objs_[i].track.id == tracks_[j].id_)
      {
        tracks_[j].tracked_ = true;

        tracks_[j].lost_time_ = 0;
        tracks_[j].lost_ = false;

        tracks_[j].box_ = objs_[i];

        tracks_[j].box_center_prev_ = tracks_[j].box_center_;
        tracks_[j].box_center_ = box_centers_[i];

#if SPEEDUP_KALMAN_VEL_EST
        if (tracks_[j].tracktime_ == 2)
        {
          MyPoint32 p_rel;
          tracks_[j].box_center_.pos.get_point_rel(p_rel);
          MyPoint32 p_rel_prev;
          tracks_[j].box_center_prev_.pos.get_point_rel(p_rel_prev);

          tracks_[j].kalman_.statePre.at<float>(2) = (p_rel.x - p_rel_prev.x) / dt_;
          tracks_[j].kalman_.statePre.at<float>(3) = (p_rel.y - p_rel_prev.y) / dt_;
        }
#endif

        tracks_[j].box_corners_ = box_corners_of_boxes_[i];

        MyPoint32 p_rel;
        box_centers_[i].pos.get_point_rel(p_rel);
        correct_tracker(tracks_[j], p_rel.x, p_rel.y);

        tracks_[j].hist_.header_ = tracks_[j].box_.header;

        increase_uint(tracks_[j].tracktime_);

        tracks_[j].hist_.set_for_successive_element(
            tracks_[j].tracktime_, p_rel.x, p_rel.y,                                               //
            tracks_[j].kalman_.statePost.at<float>(0), tracks_[j].kalman_.statePost.at<float>(1),  //
            tracks_[j].kalman_.statePost.at<float>(2), tracks_[j].kalman_.statePost.at<float>(3));
      }
    }
  }
}

void KalmanTrackers::mark_lost_trackers()
{
  for (unsigned i = 0; i < tracks_.size(); i++)
  {
    bool match = false;
    for (unsigned j = 0; j < objs_.size(); j++)
    {
      if (tracks_[i].id_ == objs_[j].track.id)
      {
        match = true;
      }
    }

    if (!match)
    {
      tracks_[i].lost_time_++;
      tracks_[i].lost_ = (tracks_[i].lost_time_ > tracks_[i].lost_time_max_) ? true : false;

      if (!tracks_[i].lost_)
      {
        tracks_[i].hist_.header_ = tracks_[i].box_.header;

        increase_uint(tracks_[i].tracktime_);

        tracks_[i].hist_.set_for_successive_element(
            tracks_[i].tracktime_, tracks_[i].x_predict_, tracks_[i].y_predict_,                   //
            tracks_[i].kalman_.statePost.at<float>(0), tracks_[i].kalman_.statePost.at<float>(1),  //
            tracks_[i].kalman_.statePost.at<float>(2), tracks_[i].kalman_.statePost.at<float>(3));
      }
    }
  }
}

void KalmanTrackers::delete_lost_trackers()
{
  std::vector<KalmanTracker> reserve_elements;
  reserve_elements.reserve(tracks_.size());

  for (unsigned i = 0; i < tracks_.size(); i++)
  {
    if (!tracks_[i].lost_)
    {
      reserve_elements.push_back(tracks_[i]);
    }
    else
    {
#if DEBUG_TRACKTIME
      LOG_INFO << "Tracker " << tracks_[i].id_ << " lost! (Lasted " << tracks_[i].tracktime_ << " frames.)"
               << std::endl;
#endif
    }
  }

  std::vector<KalmanTracker>().swap(tracks_);
  tracks_.assign(reserve_elements.begin(), reserve_elements.end());
}

void KalmanTrackers::add_new_trackers()
{
  for (unsigned i = 0; i < objs_.size(); i++)
  {
    bool match = false;

    for (unsigned j = 0; j < tracks_.size(); j++)
    {
      if (objs_[i].track.id == tracks_[j].id_)
      {
        match = true;
      }
    }

    if (!match)
    {
      new_tracker(objs_[i], box_centers_[i], box_corners_of_boxes_[i]);
    }
  }
}

void KalmanTrackers::init_objs()
{
  for (unsigned i = 0; i < objs_.size(); i++)
  {
    objs_[i].track.id = 0;
  }
}

void KalmanTrackers::init_distance_table()
{
  std::vector<std::vector<float> >().swap(distance_table_);
  distance_table_.resize(tracks_.size(), std::vector<float>(objs_.size(), TRACK_INVALID));
}

void KalmanTrackers::compute_distance_table()
{
#if DEBUG_HUNGARIAN_DIST
  LOG_INFO << "== Hungarian Distance ==" << std::endl;
#endif

  for (unsigned i = 0; i < tracks_.size(); i++)
  {
    for (unsigned j = 0; j < objs_.size(); j++)
    {
      // filter unlikely track jump
      float track_range_sed =
          (tracks_[i].tracktime_ <= tracks_[i].warmup_time_) ? TRACK_RANGE_SED_WARMUP : TRACK_RANGE_SED;

      MyPoint32 p_rel;
      box_centers_[j].pos.get_point_rel(p_rel);
      float cost_box_dist = squared_euclidean_distance(tracks_[i].x_predict_, tracks_[i].y_predict_, p_rel.x, p_rel.y);

#if DEBUG_HUNGARIAN_DIST
      LOG_INFO << "i = " << i << " j = " << j << "  box_dist_diff: " << box_dist_diff << std::endl;
#endif

      if (cost_box_dist > track_range_sed)
      {
        continue;
      }

      float box_vol_ratio = std::max(box_centers_[j].area, BOX_VOL_MIN_FOR_RATIO) /
                            std::max(tracks_[i].box_center_.area, BOX_VOL_MIN_FOR_RATIO);
      float box_vol_ratio_larger = (box_vol_ratio >= 1.f) ? box_vol_ratio : 1.f / box_vol_ratio;

      if (box_vol_ratio_larger > BOX_VOL_RATIO_MAX)
      {
        continue;
      }

      float cost_box_vol_ratio = box_vol_ratio_larger * track_range_sed / BOX_VOL_RATIO_MAX;

      float cost_final = COST_BOX_DIST_W * cost_box_dist + COST_BOX_VOL_RATIO_W * cost_box_vol_ratio;

      if (cost_final <= track_range_sed)
      {
        distance_table_[i][j] = cost_final;
      }

#if DEBUG_HUNGARIAN_DIST
      LOG_INFO << "i = " << i << " j = " << j << std::endl;
      LOG_INFO << "factor_dist_to_obj_avg: " << factor_dist_to_obj_avg << " box_vol_diff_ratio: " << box_vol_diff_ratio
               << std::endl;
      LOG_INFO << "box_vol_same_ratio1: " << box_vol_same_ratio1 << " box_vol_same_ratio2: " << box_vol_same_ratio2
               << std::endl;
      LOG_INFO << "factor_box_vol: " << factor_box_vol << " factor_final: " << factor_final << std::endl;
      LOG_INFO << "------------------------------------------------------------" << std::endl;
#endif
    }
  }
}

void KalmanTrackers::associate_data()
{
  // data association
  unsigned costMatSize = std::max(tracks_.size(), objs_.size());
  std::vector<std::vector<double> > costMatrix(costMatSize, vector<double>(costMatSize, 0));

  for (unsigned i = 0; i < tracks_.size(); i++)
  {
    for (unsigned j = 0; j < objs_.size(); j++)
    {
      costMatrix[i][j] = (double)distance_table_[i][j];
    }
  }

#if DEBUG
  LOG_INFO << "cost matrix: " << std::endl;

  for (unsigned i = 0; i < costMatrix.size(); i++)
  {
    for (unsigned j = 0; j < costMatrix.size(); j++)
    {
      LOG_INFO << costMatrix[i][j] << ", ";
    }
    LOG_INFO << std::endl;
  }
#endif

  Hungarian hun;
  std::vector<int> assignment;

#if DEBUG
  double cost = hun.solve(costMatrix, assignment);

  for (unsigned i = 0; i < costMatrix.size(); i++)
  {
    LOG_INFO << i << "," << assignment[i] << "\t";
  }

  LOG_INFO << "\ncost: " << cost << std::endl;
#else
  hun.solve(costMatrix, assignment);
#endif

  for (unsigned i = 0; i < tracks_.size(); i++)
  {
    if ((unsigned)assignment[i] < objs_.size())
    {
      if (costMatrix[i][assignment[i]] < TRACK_RANGE_SED)
      {
        objs_[assignment[i]].track.id = tracks_[i].id_;
      }
    }
  }
}

void KalmanTrackers::give_ids_to_unassociated_objs()
{
  // check unassociated objs_
  for (unsigned i = 0; i < objs_.size(); i++)
  {
    if (objs_[i].track.id < TRACK_ID_MIN)
    {
      objs_[i].track.id = trackid_new_;
      increase_track_id();
    }
  }
}

void KalmanTrackers::correct_duplicate_track_ids()
{
  for (unsigned i = 0; i < objs_.size(); i++)
  {
    for (unsigned j = i + 1; j < objs_.size(); j++)
    {
      if (objs_[j].track.id == objs_[i].track.id)
      {
        objs_[j].track.id = trackid_new_;
        increase_track_id();
      }
    }
  }
}

void KalmanTrackers::increase_track_id()
{
  if (trackid_new_ == TRACK_ID_MAX)
  {
    trackid_new_ = TRACK_ID_MIN;
  }
  else
  {
    trackid_new_++;
  }
}

void KalmanTrackers::increase_tracktime()
{
  for (unsigned i = 0; i < tracks_.size(); i++)
  {
    increase_uint(tracks_[i].tracktime_);
  }
}

void KalmanTrackers::kalman_tracker_main(const long long dt)
{
  if (objs_.size() == 0 && tracks_.size() == 0)
  {
    return;
  }

  // feature extraction: bbox center
  extract_box_centers();
  extract_box_corners_of_boxes();
  extract_box_two_axes_of_boxes();

  // kalman filter: prediction step
  for (unsigned i = 0; i < tracks_.size(); i++)
  {
    tracks_[i].predict();
  }

  // data association: hungarian algorithm
  init_objs();
  init_distance_table();
  compute_distance_table();
  associate_data();

  // track id management
  give_ids_to_unassociated_objs();
  correct_duplicate_track_ids();

  // sync objs_ and tracks_
  set_time_displacement(dt);
  update_associated_trackers();  // including kalman filter: update step
  mark_lost_trackers();
  delete_lost_trackers();
  add_new_trackers();

  return;
}
}  // namespace tpp