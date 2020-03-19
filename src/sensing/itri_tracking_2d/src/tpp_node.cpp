#include "tpp_node.h"

namespace tpp
{
boost::shared_ptr<ros::AsyncSpinner> g_spinner;
static double input_fps = 5;    // known callback rate
static double output_fps = 10;  // expected publish rate

static unsigned int num_publishs_per_loop =
    std::max((unsigned int)1, (unsigned int)std::floor(std::floor(output_fps / input_fps)));

bool g_trigger = false;

void signal_handler(int sig)
{
  if (sig == SIGINT)
  {
    LOG_INFO << "END itri_tracking_2d" << std::endl;
    g_spinner->stop();
    ros::shutdown();
  }
}

static bool done_with_profiling()
{
#if ENABLE_PROFILING_MODE
  static int num_loop = 0;
  if (num_loop < 60 * output_fps)
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

void TPPNode::callback_fusion(const msgs::DetectedObjectArray::ConstPtr& input)
{
#if DEBUG_CALLBACK
  LOG_INFO << "callback_fusion() start" << std::endl;
#endif

#if DEBUG_COMPACT
  LOG_INFO << "-----------------------------------------" << std::endl;
#endif

  objs_header_prev_ = objs_header_;
  objs_header_ = input->header;

  double objs_header_stamp_ = objs_header_.stamp.toSec();
  double objs_header_stamp_prev_ = objs_header_prev_.stamp.toSec();

  is_legal_dt_ =
      (objs_header_stamp_prev_ > 0 && vel_.init_time(objs_header_stamp_, objs_header_stamp_prev_) == 0) ? true : false;

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

#if INPUT_ALL_CLASS
    KTs_.objs_.assign(input->objects.begin(), input->objects.end());
#else
    KTs_.objs_.reserve(input->objects.size());
    for (unsigned i = 0; i < input->objects.size(); i++)
    {
      if (input->objects[i].classId >= 1 && input->objects[i].classId <= 3)
      {
        KTs_.objs_.push_back(input->objects[i]);
      }
    }
#endif

    for (auto& obj : KTs_.objs_)
    {
      obj.absSpeed = 0.f;
      obj.relSpeed = 0.f;
    }

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
}

void TPPNode::subscribe_and_advertise_topics()
{
  std::string topic = "PathPredictionOutput";

  if (in_source_ == 1)
  {
    LOG_INFO << "Input Source: /CameraDetection/polygon" << std::endl;
    camera_sub_ = nh_.subscribe("/CameraDetection/polygon", 1, &TPPNode::callback_fusion, this);
  }
  else
  {
    LOG_INFO << "Input Source: /CamObjFrontCenter" << std::endl;
    camera_sub_ = nh_.subscribe("/CamObjFrontCenter", 1, &TPPNode::callback_fusion, this);
  }

  track2d_pub_ = nh_.advertise<msgs::DetectedObjectArray>(topic, 2);

  nh2_.setCallbackQueue(&queue_);
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

    track.box_.absSpeed = track.box_.track.absolute_velocity.speed;  // km/h

    if (std::isnan(track.box_.absSpeed))
    {
      track.box_.absSpeed = 0.f;
    }

    MyPoint32 p_rel;
    track.box_center_.pos.get_point_rel(p_rel);  // m

    Vector3_32 rel_v_rel;
    rel_v_rel.x = track.box_.track.relative_velocity.x;  // km/h
    rel_v_rel.y = track.box_.track.relative_velocity.y;  // km/h
    rel_v_rel.z = track.box_.track.relative_velocity.z;  // km/h

    track.box_.relSpeed = compute_relative_speed_obj2ego(rel_v_rel, p_rel);  // km/h

    if (std::isnan(track.box_.relSpeed))
    {
      track.box_.relSpeed = 0.f;
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
        box.track.is_over_max_length = false;

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

        // set is_over_max_length
        if (track.hist_.len_ >= (unsigned short)track.hist_.max_len_)
        {
          box.track.is_over_max_length = true;
        }

        // set states
        box.track.states.resize(box.track.max_length);

        for (unsigned k = 0; k < box.track.states.size(); k++)
        {
          box.track.states[k] = track.hist_.states_[k];
        }

#if NOT_OUTPUT_SHORT_TERM_TRACK_LOST_BBOX
      }
#endif  // NOT_OUTPUT_SHORT_TERM_TRACK_LOST_BBOX
#if REMOVE_IMPULSE_NOISE
    }
#endif  // REMOVE_IMPULSE_NOISE
  }
}

inline bool test_file_exist(const std::string& name)
{
  ifstream f(name.c_str());
  return f.good();
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
        << "#5-1 input bbox center x (m), "            //
        << "#5-2 input bbox center y (m), "            //
        << "#6-1 kalman-filtered bbox center x (m), "  //
        << "#6-2 kalman-filtered bbox center y (m), "  //
        << "#7 abs vx (km/h), "                        //
        << "#8 abs vy (km/h), "                        //
        << "#9 abs speed (km/h), "                     //
        << "#10 rel vx (km/h), "                       //
        << "#11 rel vy (km/h), "                       //
        << "#12 rel speed (km/h), "                    //
        << "#13 ppx in 5 ticks (m), "                  //
        << "#14 ppy in 5 ticks (m), "                  //
        << "#15 ppx in 10 ticks (m), "                 //
        << "#16 ppy in 10 ticks (m), "                 //
        << "#17 ppx in 15 ticks (m), "                 //
        << "#18 ppy in 15 ticks (m), "                 //
        << "#19 ppx in 20 ticks (m), "                 //
        << "#20 ppy in 20 ticks (m), "                 //
        << "#21 ego x abs (m), "                       //
        << "#22 ego y abs (m), "                       //
        << "#23 ego z abs (m), "                       //
        << "#24 ego heading (rad), "                   //
        << "#25 kf Q1, "                               //
        << "#26 kf Q2, "                               //
        << "#27 kf Q3, "                               //
        << "#28 kf R, "                                //
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
        << obj.lidarInfo.boxCenter.x << ", "                // #5-1 input bbox center x (m)
        << obj.lidarInfo.boxCenter.y << ", "                // #5-2 input bbox center y (m)
        << (obj.bPoint.p0.x + obj.bPoint.p6.x) / 2 << ", "  // #6-1 kalman-filtered bbox center x (m)
        << (obj.bPoint.p0.y + obj.bPoint.p6.y) / 2 << ", "  // #6-2 kalman-filtered bbox center y (m)
        << obj.track.absolute_velocity.x << ", "            // #7 abs vx (km/h)
        << obj.track.absolute_velocity.y << ", "            // #8 abs vy (km/h)
        << obj.absSpeed << ", "                             // #9 abs speed (km/h)
        << obj.track.relative_velocity.x << ", "            // #10 rel vx (km/h)
        << obj.track.relative_velocity.y << ", "            // #11 rel vy (km/h)
        << obj.relSpeed;                                    // #12 rel speed (km/h)

    if (obj.track.is_ready_prediction)
    {
      // #13 ppx in 5 ticks (m)
      // #14 ppy in 5 ticks (m)
      // #15 ppx in 10 ticks (m)
      // #16 ppy in 10 ticks (m)
      // #17 ppx in 15 ticks (m)
      // #18 ppy in 15 ticks (m)
      // #19 ppx in 20 ticks (m)
      // #20 ppy in 20 ticks (m)
      for (unsigned int j = 0; j < num_forecasts_; j = j + 5)
      {
        ofs << ", " << obj.track.forecasts[j].position.x << ", " << obj.track.forecasts[j].position.y;
      }

      ofs << ", "                //
          << ego_x_abs_ << ", "  // #21 ego x abs
          << ego_y_abs_ << ", "  // #22 ego y abs
          << ego_z_abs_ << ", "  // #23 ego z abs
          << ego_heading_;       // #24 ego heading (rad)

      ofs << ", "                   //
          << KTs_.get_Q1() << ", "  // #25 kf Q1
          << KTs_.get_Q2() << ", "  // #26 kf Q2
          << KTs_.get_Q3() << ", "  // #27 kf Q3
          << KTs_.get_R() << ", "   // #28 kf R
          << KTs_.get_P0();         // #29 kf P0
    }

    ofs << "\n";
    std::cout << "[Produced] time = " << obj.header.stamp << ", track_id = " << obj.track.id << std::endl;
  }

  ofs.close();
}

void TPPNode::save_ttc_to_csv(std::vector<msgs::DetectedObject>& objs)
{
  std::ofstream ofs;
  std::stringstream ss;
  ss << "../../../ttc_output.csv";
  std::string fname = ss.str();

  if (objs.empty())
  {
    std::cout << "objs is empty. No output to .csv." << std::endl;
    return;
  }

  if (!test_file_exist(fname))
  {
    ofs.open(fname, std::ios_base::app);

    ofs << "Frame number,"              //
        << "Timestamp,"                 //
        << "dt (sec),"                  //
        << "Track ID,"                  //
        << "Distance of SV & POV (m),"  //
        << "SV abs. speed (km/h),"      //
        << "POV abs. speed (km/h),"     //
        << "POV rel. speed (km/h),"     //
        << "TTC (sec)\n";
  }
  else
  {
    ofs.open(fname, std::ios_base::app);
  }

  ros::Duration dt_s(0, dt_);

  for (const auto& obj : objs)
  {
    float dist_m = closest_distance_of_obj_pivot(obj);  //  Distance of SV & POV (m)
    double ttc_s = (obj.relSpeed < 0) ? (dist_m * 3.6f) / -obj.relSpeed : -1.;

    if (ttc_s != -1.)
    {
      ofs << seq_ << ","                        // Frame number
          << objs_header_.stamp.toSec() << ","  // Timestamp
          << dt_s.toSec() << ", "               // dt (sec)
          << obj.track.id << ","                // Track ID
          << dist_m << ","                      // Distance of SV & POV (m)
          << ego_speed_kmph_ << ","             // SV abs. speed (km/h)
          << obj.absSpeed << ","                // POV abs. speed (km/h)
          << obj.relSpeed << ","                // POV rel. speed (km/h)
          << ttc_s << "\n";                     // TTC (sec)

      if (ttc_s >= 0.)
        LOG_INFO << fixed << setprecision(3)  //
                 << "Seq: " << seq_ << "   Track ID: " << obj.track.id << "   dist = " << dist_m << "m   TTC: " << ttc_s
                 << "s (rel. speed = " << obj.relSpeed << " km/h)" << std::endl;
      else
        LOG_INFO << fixed << setprecision(3)  //
                 << "Seq: " << seq_ << "   Track ID: " << obj.track.id << "   dist = " << dist_m << "m   TTC: ERROR!"
                 << std::endl;
    }
    else
    {
      LOG_INFO << fixed << setprecision(3)  //
               << "Seq: " << seq_ << "   Track ID: " << obj.track.id << "   dist = " << dist_m << "m   TTC: X"
               << std::endl;
    }
  }

  ofs.close();
}
#endif

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

void TPPNode::get_current_ego_data(const tf2_ros::Buffer& tf_buffer, const ros::Time fusion_stamp)
{
  geometry_msgs::TransformStamped tf_stamped;
  bool is_warning = false;

  try
  {
    tf_stamped = tf_buffer.lookupTransform("map", "lidar", fusion_stamp);
  }
  catch (tf2::TransformException& ex)
  {
    ROS_WARN("%s", ex.what());
    try
    {
      tf_stamped = tf_buffer.lookupTransform("map", "lidar", ros::Time(0));
    }
    catch (tf2::TransformException& ex)
    {
      ROS_WARN("%s", ex.what());
      is_warning = true;
    }
  }

  if (!is_warning)
  {
    vel_.set_ego_x_abs(tf_stamped.transform.translation.x);
    vel_.set_ego_y_abs(tf_stamped.transform.translation.y);

    double roll, pitch, yaw;
    quaternion_to_rpy(roll, pitch, yaw, tf_stamped.transform.rotation.x, tf_stamped.transform.rotation.y,
                      tf_stamped.transform.rotation.z, tf_stamped.transform.rotation.w);
    vel_.set_ego_heading(yaw);
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
  std::string domain = "/itri_tracking_2d/";
  nh_.param<int>(domain + "input_source", in_source_, 0);

  nh_.param<double>(domain + "input_fps", input_fps, 10.);
  nh_.param<double>(domain + "output_fps", output_fps, 10.);
  num_publishs_per_loop = std::max((unsigned int)1, (unsigned int)std::floor(std::floor(output_fps / input_fps)));
}

int TPPNode::run()
{
  set_ros_params();

  subscribe_and_advertise_topics();

  LOG_INFO << "itri_tracking_2d is running!" << std::endl;

  signal(SIGINT, signal_handler);

  // Create AsyncSpinner, run it on all available cores and make it process custom callback queue
  g_spinner.reset(new ros::AsyncSpinner(0, &queue_));
  g_spinner->start();

  g_trigger = true;

  tf2_ros::Buffer tf_buffer;
  tf2_ros::TransformListener tf_listener(tf_buffer);

  ros::Rate loop_rate(output_fps);

  while (ros::ok() && !done_with_profiling())
  {
#if DEBUG_CALLBACK
    LOG_INFO << "ROS loop start" << std::endl;
#endif

    if (!is_legal_dt_)
    {
      tf_buffer.clear();
    }

    if (g_trigger && is_legal_dt_)
    {
      get_current_ego_data(tf_buffer, KTs_.header_.stamp);  // sync data

#if DEBUG
      LOG_INFO << "Tracking main process start" << std::endl;
#endif

      // MOT: SORT algorithm
      KTs_.kalman_tracker_main(dt_, ego_x_abs_, ego_y_abs_, ego_z_abs_, ego_heading_);
      compute_velocity_kalman();

      publish_tracking();

      g_trigger = false;
    }

    ros::spinOnce();  // Process callback_fusion()
    loop_rate.sleep();
  }

  return 0;
}
}  // namespace tpp
