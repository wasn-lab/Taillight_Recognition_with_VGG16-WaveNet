#include "path_predict.h"

namespace tpp
{
void PathPredict::callback_tracking(std::vector<msgs::DetectedObject>& pp_objs_, const float ego_x_abs,
                                    const float ego_y_abs, const float ego_z_abs, const float ego_heading,
                                    const int input_source)
{
  ego_x_abs_ = ego_x_abs;
  ego_y_abs_ = ego_y_abs;
  ego_z_abs_ = ego_z_abs;
  ego_heading_ = ego_heading;
  input_source_ = input_source;

#if DEBUG_PP
  LOG_INFO << "num_pp_input_in_use_ =" << num_pp_input_in_use_ << std::endl;
#endif

  for (unsigned i = 0; i < pp_objs_.size(); i++)
  {
    // bound num_pp_input_in_use_ range
    num_pp_input_in_use_ =
        std::max(std::min(num_pp_input_max_, (std::size_t)pp_objs_[i].track.max_length), num_pp_input_min_);

    pp_objs_[i].track.is_ready_prediction = false;

    // check track.head value
    if (pp_objs_[i].track.head < 255 && pp_objs_[i].track.head < pp_objs_[i].track.max_length)
    {
      // check enough record of track history for pp
      if (pp_objs_[i].track.head >= (int)(num_pp_input_min_ - 1) || pp_objs_[i].track.is_over_max_length == true)
      {
        pp_objs_[i].track.is_ready_prediction = true;

        // PP filter 1: object absolute speed
        if (pp_objs_[i].speed_abs < pp_obj_min_kmph_ && pp_objs_[i].speed_abs > pp_obj_max_kmph_)
        {
          pp_objs_[i].track.is_ready_prediction = false;
          continue;
        }

        float box_center_x = (pp_objs_[i].bPoint.p0.x + pp_objs_[i].bPoint.p6.x) / 2;
        float box_center_y = (pp_objs_[i].bPoint.p0.y + pp_objs_[i].bPoint.p6.y) / 2;

        float box_x_length = std::abs(pp_objs_[i].bPoint.p6.x - pp_objs_[i].bPoint.p0.x);
        float box_y_length = std::abs(pp_objs_[i].bPoint.p6.y - pp_objs_[i].bPoint.p0.y);
        float box_z_length = std::abs(pp_objs_[i].bPoint.p6.z - pp_objs_[i].bPoint.p0.z);

        // PP filter 2: object position x
        if (box_center_x < pp_allow_x_min_m || box_center_x > pp_allow_x_max_m)
        {
          pp_objs_[i].track.is_ready_prediction = false;
          continue;
        }

        // PP filter 3: object position y
        if (box_center_y < pp_allow_y_min_m || box_center_y > pp_allow_y_max_m)
        {
          pp_objs_[i].track.is_ready_prediction = false;
          continue;
        }

        if (input_source_ != InputSource::RadarDet)
        {
          // PP filter 4: object size z
          if (box_z_length < box_length_thr_z)
          {
            pp_objs_[i].track.is_ready_prediction = false;
            continue;
          }

          // PP filter 5: object size x & y
          if (pp_objs_[i].fusionSourceId == sensor_msgs_itri::DetectedObjectClassId::Person ||
              pp_objs_[i].fusionSourceId == sensor_msgs_itri::DetectedObjectClassId::Bicycle ||
              pp_objs_[i].fusionSourceId == sensor_msgs_itri::DetectedObjectClassId::Motobike)
          {
            if (box_x_length < box_length_thr_xy_thin && box_y_length < box_length_thr_xy_thin)
            {
              pp_objs_[i].track.is_ready_prediction = false;
              continue;
            }
          }
          else
          {
            if (box_x_length < box_length_thr_xy && box_y_length < box_length_thr_xy)
            {
              pp_objs_[i].track.is_ready_prediction = false;
              continue;
            }
          }
        }

#if DEBUG_PP
        if (pp_objs_[i].track.is_ready_prediction == true)
        {
          std::cout << "PP_ready" << std::endl;
        }
#endif
      }
    }

#if PP_VERTICES_VIA_SPEED == 1
    pp_objs_[i].track.forecasts.resize(num_forecasts_ * 5);
#else
    pp_objs_[i].track.forecasts.resize(num_forecasts_);
#endif

#if PP_VERTICES_VIA_SPEED == 1
    for (unsigned j = 0; j < num_forecasts_ * 5; j++)
#else
    for (unsigned j = 0; j < num_forecasts_; j++)
#endif
    {
      pp_objs_[i].track.forecasts[j].position.x = 0;
      pp_objs_[i].track.forecasts[j].position.y = 0;
      pp_objs_[i].track.forecasts[j].covariance_xx = 0;
      pp_objs_[i].track.forecasts[j].covariance_yy = 0;
      pp_objs_[i].track.forecasts[j].covariance_xy = 0;
      pp_objs_[i].track.forecasts[j].correlation_xy = 0;
    }
  }
}

void PathPredict::compute_pos_offset(const std::vector<long double>& data_x, const std::vector<long double>& data_y)
{
  PointLD min;
  min.x = std::numeric_limits<long double>::max();
  min.y = std::numeric_limits<long double>::max();

  for (unsigned i = 0; i < data_x.size(); i++)
  {
    if (data_x[i] < min.x)
    {
      min.x = data_x[i];
    }

    if (data_y[i] < min.y)
    {
      min.y = data_y[i];
    }
  }

  PointLD offset;
  offset.x = -min.x + input_shift_m_;
  offset.y = -min.y + input_shift_m_;
  offset.z = 0.;

  offsets_.push_back(offset);
#if DEBUG_PP
  LOG_INFO << "PP offset = x:" << offsets_.back().x << " y:" << offsets_.back().y << " z:" << offsets_.back().z
           << std::endl;
#endif
}

void PathPredict::normalize_pos(std::vector<long double>& data_x, std::vector<long double>& data_y)
{
#if DEBUG_PP
  LOG_INFO << "== Before & After PP Data Normalization ==" << std::endl;
#endif
  for (unsigned i = 0; i < data_x.size(); i++)
  {
#if DEBUG_PP
    LOG_INFO << i + 1 << "  data_x = " << data_x[i] << "  data_y = " << data_y[i] << std::endl;
#endif
    data_x[i] += offsets_.back().x;
    data_y[i] += offsets_.back().y;
#if DEBUG_PP
    LOG_INFO << i + 1 << "  data_x = " << data_x[i] << "  data_y = " << data_y[i] << std::endl;
#endif
  }
}

void PathPredict::create_pp_input(const MyPoint32 point, std::vector<long double>& data_x,
                                  std::vector<long double>& data_y)
{
  float x_abs = point.x;
  float y_abs = point.y;
  float z_abs = point.z;

  float x_rel = 0.f;
  float y_rel = 0.f;
  float z_rel = 0.f;

  transform_point_abs2rel(x_abs, y_abs, z_abs, x_rel, y_rel, z_rel, ego_x_abs_, ego_y_abs_, ego_z_abs_, ego_heading_);

  data_x.push_back(x_rel);
  data_y.push_back(y_rel);
}

void PathPredict::create_pp_input_main(const msgs::TrackInfo& track, std::vector<long double>& data_x,
                                       std::vector<long double>& data_y)
{
  int start = track.head - num_pp_input_in_use_ + 1;

#if DEBUG_PP_TRAJ
  LOG_INFO << std::endl;
#endif

  for (int i = start; i <= track.head; i++)
  {
    if (i < 0)
    {
      if (track.is_over_max_length)
      {
        create_pp_input(track.states[i + track.max_length].estimated_position, data_x, data_y);
#if DEBUG_PP_TRAJ
        LOG_INFO << "Traj " << i << " " << data_x.back() << " " << data_y.back()
                 << " (abs_coord: " << track.states[i + track.max_length].estimated_position.x << " "
                 << track.states[i + track.max_length].estimated_position.y << ")" << std::endl;
#endif
      }
    }
    else
    {
      create_pp_input(track.states[i].estimated_position, data_x, data_y);
#if DEBUG_PP_TRAJ
      LOG_INFO << "Traj " << i << " " << data_x.back() << " " << data_y.back()
               << " (abs_coord: " << track.states[i].estimated_position.x << " " << track.states[i].estimated_position.y
               << ")" << std::endl;
#endif
    }
  }
}

long double PathPredict::variance(const std::vector<long double>& samples, const long double mean)
{
  if (samples.size() < 2)
  {
    return 0;
  }

  long double variance = 0;
  long double diff = 0;
  for (const long double sample : samples)
  {
    diff = sample - mean;
    variance += std::pow(diff, 2);
  }

  return variance / (samples.size() - 1);
}

long double PathPredict::standard_deviation(const long double covariance)
{
  return std::sqrt(covariance);
}

long double PathPredict::covariance(const std::vector<long double>& samples_x,
                                    const std::vector<long double>& samples_y, const long double mean_x,
                                    const long double mean_y)
{
  if (samples_x.size() != samples_y.size())
  {
    return 0;
  }

  if (samples_x.size() < 2)
  {
    return 0;
  }

  long double covariance = 0;
  long double diff_x = 0;
  long double diff_y = 0;
  for (unsigned i = 0; i < samples_x.size(); i++)
  {
    diff_x = samples_x[i] - mean_x;
    diff_y = samples_y[i] - mean_y;
    covariance += (diff_x * diff_y);
  }

  return covariance / (samples_x.size() - 1);
}

long double PathPredict::correlation(const long double cov_xy, const long double stdev_x, const long double stdev_y)
{
  long double stdev_xy = 0;
  assign_value_cannot_zero(stdev_xy, stdev_x * stdev_y);

  return cov_xy / stdev_xy;
}

void PathPredict::confidence_ellipse(PPLongDouble& pp, const float alpha)
{
  // bound cov_xx/xy/yy to prevent module fail
  long double cov_min = -100;
  long double cov_max = 100;
  long double cov_xx = std::max(std::min(pp.cov_xx, cov_max), cov_min);
  long double cov_xy = std::max(std::min(pp.cov_xy, cov_max), cov_min);
  long double cov_yy = std::max(std::min(pp.cov_yy, cov_max), cov_min);

  Eigen::Matrix<float, 2, 2> sigma;
  sigma << cov_xx, cov_xy, cov_xy, cov_yy;

  Eigen::EigenSolver<Eigen::Matrix<float, 2, 2> > sol(sigma);

  float norm_eigenvector1 =
      euclidean_distance(sol.eigenvectors().col(0)[0].real(), sol.eigenvectors().col(0)[1].real());
  float norm_eigenvector2 =
      euclidean_distance(sol.eigenvectors().col(1)[0].real(), sol.eigenvectors().col(1)[1].real());

  // use abs to prevent sqrt of negative value (nan)
  float sqrt_eigenvalue1 = std::sqrt(std::abs(sol.eigenvalues()[0].real()));
  float sqrt_eigenvalue2 = std::sqrt(std::abs(sol.eigenvalues()[1].real()));

  pp.a1 = alpha * divide(sqrt_eigenvalue1, norm_eigenvector1);
  pp.a2 = alpha * divide(sqrt_eigenvalue2, norm_eigenvector2);

  float diff1x = pp.a1 * sol.eigenvectors().col(0)[0].real();
  float diff1y = pp.a1 * sol.eigenvectors().col(0)[1].real();

  float yaw1 = std::atan2(diff1y, diff1x);
  pp.q1.setRPY(0, 0, yaw1);

  float diff2x = pp.a2 * sol.eigenvectors().col(1)[0].real();
  float diff2y = pp.a2 * sol.eigenvectors().col(1)[1].real();

  float yaw2 = std::atan2(diff2y, diff2x);
  pp.q2.setRPY(0, 0, yaw2);

  std::vector<msgs::PointXY> vertices(4);

  vertices[0].x = pp.pos_x - diff1x;
  vertices[0].y = pp.pos_y - diff1y;
  vertices[1].x = pp.pos_x + diff1x;
  vertices[1].y = pp.pos_y + diff1y;

  vertices[2].x = pp.pos_x - diff2x;
  vertices[2].y = pp.pos_y - diff2y;
  vertices[3].x = pp.pos_x + diff2x;
  vertices[3].y = pp.pos_y + diff2y;

#if DEBUG_CONF_E
  LOG_INFO << "pp.cov_xx: " << pp.cov_xx << "  pp.cov_yy: " << pp.cov_yy << "  pp.cov_xy: " << pp.cov_xy << std::endl;
  LOG_INFO << "cov_xx: " << cov_xx << "  cov_yy: " << cov_yy << "  cov_xy: " << cov_xy << std::endl;

  LOG_INFO << "Eigenvalues:" << std::endl;
  LOG_INFO << sol.eigenvalues() << std::endl;
  LOG_INFO << "Eigenvectors:" << std::endl;
  LOG_INFO << sol.eigenvectors() << std::endl;

  LOG_INFO << "norm_eigenvector1: " << norm_eigenvector1 << "  norm_eigenvector2: " << norm_eigenvector2 << std::endl;
  LOG_INFO << "sqrt_eigenvalue1: " << sqrt_eigenvalue1 << "  sqrt_eigenvalue2: " << sqrt_eigenvalue2 << std::endl;
  LOG_INFO << "pp.a1: " << pp.a1 << "  pp.a2: " << pp.a2 << std::endl;

  LOG_INFO << "yaw1: " << yaw1 << "  yaw2: " << yaw2 << std::endl;
  LOG_INFO << "diff1: (" << diff1x << ", " << diff1y << ")" << std::endl;
  LOG_INFO << "diff2: (" << diff2x << ", " << diff2y << ")" << std::endl;
  LOG_INFO << "pp.q1: " << pp.q1[0] << " " << pp.q1[1] << " " << pp.q1[2] << " " << pp.q1[3] << std::endl;
  LOG_INFO << "pp.q2: " << pp.q2[0] << " " << pp.q2[1] << " " << pp.q2[2] << " " << pp.q2[3] << std::endl;

  for (unsigned int i = 0; i < 4; i++)
    LOG_INFO << "vertices[" << i << "]: (" << vertices[i].x << ", " << vertices[i].y << ")" << std::endl;
#endif
}

int PathPredict::ar1_params_main(PPLongDouble& pp, std::vector<long double>& data_x, std::vector<long double>& data_y)
{
  pp.beta0_x = 0.;
  pp.beta1_x = 0.;

  pp.beta0_y = 0.;
  pp.beta1_y = 0.;

  AR1 ar1;

  int err_x = ar1.compute_params(data_x, pp.beta0_x, pp.beta1_x);
#if DEBUG_PP
  LOG_INFO << "beta0_x = " << pp.beta0_x << "\tbeta1_x = " << pp.beta1_x << std::endl << std::endl;
#endif
  if (err_x > 0)
  {
    return err_x;
  }

  int err_y = ar1.compute_params(data_y, pp.beta0_y, pp.beta1_y);
#if DEBUG_PP
  LOG_INFO << "beta0_y = " << pp.beta0_y << "\tbeta1_y = " << pp.beta1_y << std::endl << std::endl;
#endif
  if (err_y > 0)
  {
    return err_y;
  }

  pp.sum_samples_x = 0.;
  pp.sum_samples_y = 0.;

  for (unsigned i = 0; i < data_x.size(); i++)
  {
    pp.sum_samples_x += data_x[i];
    pp.sum_samples_y += data_y[i];
  }

  pp.observation_x = data_x.back();
  pp.observation_y = data_y.back();
  covariance_matrix(pp, data_x, data_y);

  return 0;
}

void PathPredict::ar1_pp(const long double beta0, const long double beta1, long double& pos, long double& observation,
                         long double& sum_samples, long double& mean, long double& cov, long double& stdev,
                         std::vector<long double>& data)
{
  pos = beta0 + beta1 * observation;
  observation = pos;
  data.push_back(pos);

  sum_samples += pos;
  mean = sum_samples / data.size();

  cov = variance(data, mean);
  stdev = standard_deviation(cov);
}

void PathPredict::covariance_matrix(PPLongDouble& pp, std::vector<long double>& data_x,
                                    std::vector<long double>& data_y)
{
  // x
  ar1_pp(pp.beta0_x, pp.beta1_x, pp.pos_x, pp.observation_x, pp.sum_samples_x, pp.mean_x, pp.cov_xx, pp.stdev_x,
         data_x);
  // y
  ar1_pp(pp.beta0_y, pp.beta1_y, pp.pos_y, pp.observation_y, pp.sum_samples_y, pp.mean_y, pp.cov_yy, pp.stdev_y,
         data_y);
  // xy
  pp.cov_xy = covariance(data_x, data_y, pp.mean_x, pp.mean_y);
  pp.corr_xy = correlation(pp.cov_xy, pp.stdev_x, pp.stdev_y);

#if DEBUG_PP
  LOG_INFO << "Position x : " << O_FIX << O_P << pp.pos_x << "\t"
           << "Covariance xx: " << O_FIX << O_P << pp.cov_xx << "\t"
           << "Standard Deviation x: " << O_FIX << O_P << pp.stdev_x << "\t"
           << "Mean x: " << O_FIX << O_P << pp.mean_x << std::endl;
  LOG_INFO << "Position y : " << O_FIX << O_P << pp.pos_y << "\t"
           << "Covariance yy: " << O_FIX << O_P << pp.cov_yy << "\t"
           << "Standard Deviation y: " << O_FIX << O_P << pp.stdev_y << "\t"
           << "Mean y: " << O_FIX << O_P << pp.mean_y << std::endl;
  LOG_INFO << "Covariance xy: " << O_FIX << O_P << pp.cov_xy << "\t"
           << "Correlation xy: " << O_FIX << O_P << pp.corr_xy << std::endl;
  LOG_INFO << std::endl;
#endif
}

int PathPredict::predict(std::size_t max_order_, const std::size_t num_forecasts_, std::vector<long double>& data_x,
                         std::vector<long double>& data_y, std::vector<PPLongDouble>& pps)
{
  if (data_x.size() != data_y.size())
  {
    LOG_INFO << "Error: data_x.size() != data_y.size()" << std::endl;
    return 1;
  }

  if (max_order_ > data_x.size())
  {
    LOG_INFO << "Error: max_order_ > #(input data)" << std::endl;
    return 2;
  }

#if DEBUG_PP
  printf("\nEstimating an AR(%lu) model using %lu samples to forecast %lu steps\n\n",
         static_cast<unsigned long>(max_order_), static_cast<unsigned long>(data_x.size()),
         static_cast<unsigned long>(num_forecasts_));
#endif

  pps.reserve(num_forecasts_);

  for (unsigned i = 0; i < num_forecasts_; i++)
  {
    PPLongDouble pp;
    int err = ar1_params_main(pp, data_x, data_y);

    if (err > 0)
    {
#if DEBUG_PP
      std::cout << "ar1_params_main() error:" << err << std::endl;
#endif
      return err;
    }

    pps.push_back(pp);
  }

  return 0;
}

void PathPredict::confidence_threshold(const unsigned int confidence_lv)
{
  switch (confidence_lv)
  {
    case 1:
      confidence_thr_ = 1.15f;  // 75%
      break;
    case 2:
      confidence_thr_ = 1.96f;  // 95%
      break;
    default:
      confidence_thr_ = 0.675f;  // 50%
  }
}

void PathPredict::confidence_ellipse_main(const std::size_t num_forecasts_, std::vector<long double>& data_x,
                                          std::vector<long double>& data_y, std::vector<PPLongDouble>& pps)
{
  if (show_pp_ >= 1 && show_pp_ <= 3)
  {
    for (unsigned i = 0; i < num_forecasts_; i++)
    {
      confidence_ellipse(pps[i], confidence_thr_);
    }
  }
}

void PathPredict::pp_vertices(PPLongDouble& pps, const msgs::PathPrediction forecast, const int pp_idx,
                              const float abs_speed)
{
  double roll, pitch, yaw;
  geometry_msgs::Quaternion q = tf2::toMsg(pps.q1);
  quaternion_to_rpy(roll, pitch, yaw, q.x, q.y, q.z, q.w);

#if DEBUG_PP
  std::cout << "q.x:" << q.x << " q.y:" << q.y << " q.z:" << q.z << " q.z:" << q.z << " roll:" << roll
            << " pitch:" << pitch << " yaw:" << yaw << " absspeed:" << abs_speed << std::endl;
#endif

  float scale = abs_speed * (pp_idx + 1) / 200.;
  float scale1 = scale;
  float scale2 = scale;

  cv::Mat mag_m(1, 4, CV_32FC1, cv::Scalar(0));
  mag_m.at<float>(0, 0) = scale1;
  mag_m.at<float>(0, 1) = scale2;
  mag_m.at<float>(0, 2) = scale1;
  mag_m.at<float>(0, 3) = scale2;

  cv::Mat ang_rad(1, 4, CV_32FC1, cv::Scalar(0));
  double pi_half = M_PI * 0.5;
  ang_rad.at<float>(0, 0) = yaw;
  ang_rad.at<float>(0, 1) = ang_rad.at<float>(0, 0) + pi_half;
  ang_rad.at<float>(0, 2) = ang_rad.at<float>(0, 1) + pi_half;
  ang_rad.at<float>(0, 3) = ang_rad.at<float>(0, 2) + pi_half;

  cv::Mat x_m(1, 4, CV_32FC1, cv::Scalar(0));
  cv::Mat y_m(1, 4, CV_32FC1, cv::Scalar(0));
  cv::polarToCart(mag_m, ang_rad, x_m, y_m, false);

#if PP_VERTICES_VIA_SPEED == 1
  pps.v1.x = forecast.position.x + x_m.at<float>(0, 0);
  pps.v1.y = forecast.position.y + y_m.at<float>(0, 0);

  pps.v2.x = forecast.position.x + x_m.at<float>(0, 1);
  pps.v2.y = forecast.position.y + y_m.at<float>(0, 1);

  pps.v3.x = forecast.position.x + x_m.at<float>(0, 2);
  pps.v3.y = forecast.position.y + y_m.at<float>(0, 2);

  pps.v4.x = forecast.position.x + x_m.at<float>(0, 3);
  pps.v4.y = forecast.position.y + y_m.at<float>(0, 3);
#endif
}

void PathPredict::main(std::vector<msgs::DetectedObject>& pp_objs_, std::vector<std::vector<PPLongDouble> >& ppss,
                       const unsigned int show_pp, const nav_msgs::OccupancyGrid& wayarea)
{
  show_pp_ = show_pp;

#if PP_WAYAREA == 1
  float wayarea_xlen = (wayarea.info.width - 1) * wayarea.info.resolution;
  float wayarea_ylen = (wayarea.info.height - 1) * wayarea.info.resolution;
#endif

  std::vector<std::vector<PPLongDouble> >().swap(ppss);
  ppss.reserve(pp_objs_.size());

  std::vector<PointLD>().swap(offsets_);
  offsets_.reserve(pp_objs_.size());

  confidence_threshold(0);

  for (unsigned i = 0; i < pp_objs_.size(); i++)
  {
#if DEBUG_COMPACT
    LOG_INFO << "[Track ID] " << pp_objs_[i].track.id << std::endl;
#endif

    std::vector<PPLongDouble> pps;

    if (!pp_objs_[i].track.is_ready_prediction)
    {
      PointLD offset;
      offset.x = 0.;
      offset.y = 0.;
      offset.z = 0.;
      offsets_.push_back(offset);
#if DEBUG_PP
      LOG_INFO << "Require " << num_pp_input_in_use_ << "+ elements in trajectory." << std::endl;
#endif
    }
    else
    {
      std::vector<long double> data_x;
      std::vector<long double> data_y;

      create_pp_input_main(pp_objs_[i].track, data_x, data_y);

#if PP_WAYAREA == 1
      float obj_x_wayarea = data_x.back() - wayarea.info.origin.position.x;
      float obj_y_wayarea = data_y.back() - wayarea.info.origin.position.y;

      if (obj_x_wayarea >= 0. && obj_x_wayarea <= wayarea_xlen && obj_y_wayarea >= 0. && obj_y_wayarea <= wayarea_ylen)
      {
        int px = obj_x_wayarea / wayarea.info.resolution;
        int py = obj_y_wayarea / wayarea.info.resolution;
        int idx = py * wayarea.info.width + px;

        if (wayarea.data[idx] == 100)
        {
          pp_objs_[i].track.is_ready_prediction = false;
          // std::cout << "idx = " << idx << " (PP is filtered by wayarea!)" << std::endl;
        }
        // else
        // {
        //   std::cout << "idx = " << idx << std::endl;
        // }
      }
#endif

      compute_pos_offset(data_x, data_y);
      normalize_pos(data_x, data_y);

#if DEBUG_PP
      LOG_INFO << std::endl;
      LOG_INFO << "== Predict X ==" << std::endl;
      LOG_INFO << "data_x size = " << data_x.size() << std::endl;
      for (unsigned j = 0; j < data_x.size(); j++)
        LOG_INFO << "Data " << j + 1 << ": " << data_x[j] << std::endl;

      LOG_INFO << "== Predict Y ==" << std::endl;
      LOG_INFO << "data_y size = " << data_y.size() << std::endl;
      for (unsigned j = 0; j < data_y.size(); j++)
        LOG_INFO << "Data " << j + 1 << ": " << data_y[j] << std::endl;
#endif

      int err = 0;
      err = predict(max_order_, num_forecasts_, data_x, data_y, pps);
      if (err > 0)
      {
        pp_objs_[i].track.is_ready_prediction = false;
        continue;
      }

      confidence_ellipse_main(num_forecasts_, data_x, data_y, pps);

      for (unsigned j = 0; j < num_forecasts_; j++)
      {
        pp_objs_[i].track.forecasts[j].position.x = pps[j].pos_x - offsets_[i].x;
        pp_objs_[i].track.forecasts[j].position.y = pps[j].pos_y - offsets_[i].y;
        pp_objs_[i].track.forecasts[j].covariance_xx = pps[j].cov_xx;
        pp_objs_[i].track.forecasts[j].covariance_yy = pps[j].cov_yy;
        pp_objs_[i].track.forecasts[j].covariance_xy = pps[j].cov_xy;
        pp_objs_[i].track.forecasts[j].correlation_xy = pps[j].corr_xy;

#if PP_VERTICES_VIA_SPEED == 1
        pp_vertices(pps[j], pp_objs_[i].track.forecasts[j], j, pp_objs_[i].speed_abs);

        unsigned int k = num_forecasts_ + j * 4;

        pp_objs_[i].track.forecasts[k].position.x = pps[j].v1.x;
        pp_objs_[i].track.forecasts[k].position.y = pps[j].v1.y;

        pp_objs_[i].track.forecasts[k + 1].position.x = pps[j].v2.x;
        pp_objs_[i].track.forecasts[k + 1].position.y = pps[j].v2.y;

        pp_objs_[i].track.forecasts[k + 2].position.x = pps[j].v3.x;
        pp_objs_[i].track.forecasts[k + 2].position.y = pps[j].v3.y;

        pp_objs_[i].track.forecasts[k + 3].position.x = pps[j].v4.x;
        pp_objs_[i].track.forecasts[k + 3].position.y = pps[j].v4.y;
#endif
      }

#if DEBUG_PP_TRAJ
      for (unsigned j = 0; j < num_forecasts_; j++)
      {
        LOG_INFO << "Traj " << (j + 1) << " " << pp_objs_[i].track.forecasts[j].position.x << " "
                 << pp_objs_[i].track.forecasts[j].position.y << std::endl;
      }
#endif
    }

    ppss.push_back(pps);
  }
}
}  // namespace tpp
