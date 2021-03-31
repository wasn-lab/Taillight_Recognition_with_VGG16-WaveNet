/*
 *   File: localization_supervision_node.cpp
 *   Created on: Aug , 2019
 *   Author: Xu, Bo Chun
 *	 Institute: ITRI ICL U300
 */
#include "localization_supervision.h"
#include <std_msgs/Empty.h>
static ros::Publisher marker_pub;
#define RE_BY_LOCAL 1

int state_cnt;
double fq = 40;
double target_fq = 10;
static int state = 0;
int state_gnss_delay = 1;
int state_lidar_delay = 2;
int state_pose_delay = 4;
int pose_unstable = 8;

static bool state_gnss_delay_flag = 0;
static bool state_lidar_delay_flag = 0;
static bool state_pose_delay_flag = 0;
static bool pose_unstable_flag = 1;

msgs::Pose pose_rate_pub_msg;
std_msgs::Int32 state_msg;
// 3 = gnss+lidar, 5 = gnss + pose, 6 = lidar + pose, 7 = gnss + lidar + pose
double vx, vy, vz, ax, ay, az, roll_rate, pitch_rate, yaw_rate;  // velocity and acc of pose
bool is_gnss_new, is_pose_new, is_lidarall_new;

#if RE_BY_LOCAL
int pose_stable_cnt = 0;
#endif

std::chrono::high_resolution_clock::time_point gnss_start_ms, pose_start_ms, lidarall_start_ms;
std::chrono::high_resolution_clock::time_point gnss_check_ms, pose_check_ms, lidarall_check_ms;
std::chrono::high_resolution_clock::time_point while_loop_check_ms;

enum class localization_states
{
  NEED_INIT,
  FIXED,
  ERROR,
  WARNING
};

localization_states states = localization_states::NEED_INIT;
const std::vector<std::string> states_text = { "NEED_INIT", "FIXED", "ERROR", "WARNING" };

bool init_pose_callback = false;

pose pose_rate_calculate(const geometry_msgs::PoseStamped& current_pose, const geometry_msgs::PoseStamped& pre_pose)
{
  pose tmp_current_pose, tmp_pre_pose, tmp_pose_rate;

  double time_delta = current_pose.header.stamp.toSec() - pre_pose.header.stamp.toSec();

  if (time_delta == 0)
  {
    return tmp_current_pose;
  }

  tf::Quaternion current_q(current_pose.pose.orientation.x, current_pose.pose.orientation.y,
                           current_pose.pose.orientation.z, current_pose.pose.orientation.w);
  tf::Matrix3x3 current_m(current_q);
  tmp_current_pose.x = current_pose.pose.position.x;
  tmp_current_pose.y = current_pose.pose.position.y;
  tmp_current_pose.z = current_pose.pose.position.z;
  current_m.getRPY(tmp_current_pose.roll, tmp_current_pose.pitch, tmp_current_pose.yaw);

  tf::Quaternion pre_q(pre_pose.pose.orientation.x, pre_pose.pose.orientation.y, pre_pose.pose.orientation.z,
                       pre_pose.pose.orientation.w);
  tf::Matrix3x3 pre_m(pre_q);
  tmp_pre_pose.x = pre_pose.pose.position.x;
  tmp_pre_pose.y = pre_pose.pose.position.y;
  tmp_pre_pose.z = pre_pose.pose.position.z;
  pre_m.getRPY(tmp_pre_pose.roll, tmp_pre_pose.pitch, tmp_pre_pose.yaw);

  tmp_pose_rate.x = ((tmp_current_pose.x - tmp_pre_pose.x) / time_delta) * 3.6;
  tmp_pose_rate.y = ((tmp_current_pose.y - tmp_pre_pose.y) / time_delta) * 3.6;
  tmp_pose_rate.z = ((tmp_current_pose.z - tmp_pre_pose.z) / time_delta) * 3.6;
  tmp_pose_rate.roll = (tmp_current_pose.roll - tmp_pre_pose.roll) / time_delta;
  tmp_pose_rate.pitch = (tmp_current_pose.pitch - tmp_pre_pose.pitch) / time_delta;
  tmp_pose_rate.yaw = (tmp_current_pose.yaw - tmp_pre_pose.yaw) / time_delta;
  if (((tmp_pose_rate.x) * (tmp_pose_rate.x) + (tmp_pose_rate.y) * (tmp_pose_rate.y)) > 30 * 30)
  {
    pose_unstable_flag = true;
    pose_stable_cnt = 0;
  }

#if RE_BY_LOCAL
  else
  {
    if (pose_stable_cnt > 2147483646)
    {
      pose_stable_cnt = STABLE_CNT;
    }
    pose_stable_cnt++;

    if (pose_stable_cnt > STABLE_CNT)
    {
      pose_unstable_flag = false;
    }
  }
#endif

  return tmp_pose_rate;
}

bool msg_fq_check(const std::chrono::high_resolution_clock::time_point& start,
                  const std::chrono::high_resolution_clock::time_point& end, const double& threshold)
{
  std::chrono::duration<double, std::milli> fq_ms = end - start;

  return (threshold >= fq_ms.count());
}

void lidar_all_callback(const std_msgs::Empty::ConstPtr& msg)
{
  lidarall_start_ms = std::chrono::high_resolution_clock::now();
}

void pose_callback(const geometry_msgs::PoseStamped::ConstPtr& input)
{
  if (!init_pose_callback)
  {
    init_pose_callback = true;
    current_pose_quat = *input;
    return;
  }
  pose tmp_;
  pose_start_ms = std::chrono::high_resolution_clock::now();

  pre_pose_quat = current_pose_quat;
  current_pose_quat = *input;
  tmp_ = pose_rate_calculate(current_pose_quat, pre_pose_quat);
  pose_rate_pub_msg.x = tmp_.x;
  pose_rate_pub_msg.y = tmp_.y;
  pose_rate_pub_msg.z = tmp_.z;
  pose_rate_pub_msg.roll = tmp_.roll;
  pose_rate_pub_msg.pitch = tmp_.pitch;
  pose_rate_pub_msg.yaw = tmp_.yaw;
  pose_rate_pub.publish(pose_rate_pub_msg);

  return;
}

void gnss2local_callback(const geometry_msgs::PoseStamped::ConstPtr& input)
{
  gnss_start_ms = std::chrono::high_resolution_clock::now();
}

void stable_callback(const geometry_msgs::PoseStamped::ConstPtr& input)
{
}

visualization_msgs::Marker marker_generator(const visualization_msgs::Marker& input_marker, const double& cnt_exceed)
{
  visualization_msgs::Marker tmp_marker;
  tmp_marker = input_marker;
  state_cnt++;
  tmp_marker.header.stamp = ros::Time();
  tmp_marker.text = states_text[static_cast<int>(states)];

  return tmp_marker;
}
int main(int argc, char** argv)
{
  ros::init(argc, argv, "caregiver_markers");
  ros::NodeHandle nh;
  double cnt_exceed = fq / target_fq;
  marker_pub = nh.advertise<visualization_msgs::Marker>("caregiver_markers", 1);
  pose_rate_pub = nh.advertise<msgs::Pose>("pose_rate_supervision", 1);
  state_pub = nh.advertise<std_msgs::Int32>("localization_state", 1);

  size_t count = 0;
  visualization_msgs::Marker marker;
  ros::Rate loop_rate(fq);
  ros::Subscriber currentPoseSub = nh.subscribe("current_pose", 1, pose_callback);
  ros::Subscriber lidar_all_sub = nh.subscribe("/LidarAll/replay_to_localization/heartbeat", 1, lidar_all_callback);
  ros::Subscriber gnss2local_sub = nh.subscribe("gnss2local_data", 10, gnss2local_callback);
  ros::Subscriber stable_sub = nh.subscribe("lidar_locaization_fixed", 10, stable_callback);

  marker.ns = "ns";
  marker.action = visualization_msgs::Marker::ADD;
  marker.pose.position.x = 0;
  marker.pose.position.y = 0;
  marker.pose.position.z = 0;
  marker.pose.orientation.x = 0.0;
  marker.pose.orientation.y = 0.0;
  marker.pose.orientation.z = 0.0;
  marker.pose.orientation.w = 1.0;
  marker.frame_locked = true;
  marker.lifetime = ros::Duration(0.1);
  // Draw text
  marker.header.frame_id = "/base_link";
  marker.id = count++;
  marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
  marker.scale.x = 2;
  marker.scale.y = 2;
  marker.scale.z = 5;
  marker.color.a = 1.0;  // alpha
  marker.color.r = 1.0;
  marker.color.g = 0.0;
  marker.color.b = 0.0;

  while (ros::ok())
  {
    ros::spinOnce();  // update
    boost::recursive_mutex::scoped_lock scopedLock(ms_checker);

    while_loop_check_ms = std::chrono::high_resolution_clock::now();
    if (!msg_fq_check(lidarall_start_ms, while_loop_check_ms, LIDAR_POINTCLOUD_INTERVAL_MS))
    {
      // state += state_lidar_delay;
      state_lidar_delay_flag = true;
    }
    else
    {
      state_lidar_delay_flag = false;
    }
    if (!msg_fq_check(pose_start_ms, while_loop_check_ms, POSE_INTERVAL_MS))
    {
      state_pose_delay_flag = true;
    }
    else
    {
      state_pose_delay_flag = false;
    }
    if (!msg_fq_check(gnss_start_ms, while_loop_check_ms, GNSS_INTERVAL_MS))
    {
      state_gnss_delay_flag = true;
    }
    else
    {
      state_gnss_delay_flag = false;
    }
    if (state_lidar_delay_flag)
    {
      state += state_lidar_delay;
    }
    if (state_pose_delay_flag)
    {
      state += state_pose_delay;
    }
    if (state_gnss_delay_flag)
    {
      state += state_gnss_delay;
    }
    if (pose_unstable_flag)
    {
      state += pose_unstable;
    }
    scopedLock.unlock();

    marker = marker_generator(marker, cnt_exceed);
    marker_pub.publish(marker);
    if (state > 0)
    {
      std::cout << "ERROR state: " << state << std::endl;
    }

    state_msg.data = state;
    state_pub.publish(state_msg);

    state = 0;
    loop_rate.sleep();
  }

  return 0;
}
