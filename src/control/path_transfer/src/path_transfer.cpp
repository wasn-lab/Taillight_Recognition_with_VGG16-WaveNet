#include <ros/ros.h>
#include <autoware_planning_msgs/Trajectory.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/Pose.h>
#include <tf/tf.h>
#include <msgs/CurrentTrajInfo.h>
#include <vector>
#include <cmath>
#include <std_msgs/Empty.h>

#define RT_PI 3.14159265358979323846

double angle_diff_setting_in = RT_PI/4;
double angle_diff_setting_out = 0.1;
int LRturn = 0;
double curvature = 0;
double angle_diff_setting_distoturn = 0.3;
double distoturn = 0;

double z_diff_setting_in = 0.8;
double z_diff_setting_out = 0.2;
bool up_hill = false;
double seg_size = 10;
double max_slope = 0;
double slope_setting_distouphill = 0.08;
double distouphill = 0;

ros::Publisher nav_path_pub;
ros::Publisher currenttrajinfo_pub;
ros::Publisher nav_path_heartbeat_pub;

struct Point3D
{
  double x;
  double y;
  double z;
};

void calaulate_distoturn(autoware_planning_msgs::Trajectory traj)
{
  double traj_size = traj.points.size();
  double first_yaw = tf::getYaw(traj.points[0].pose.orientation);
  distoturn = 0;
  int index = 0;
  for (int i = 1; i < traj_size; i++)
  {
    index = i;
    double check_yaw = tf::getYaw(traj.points[i].pose.orientation);
    double angle_diff = abs(check_yaw - first_yaw);
    if (angle_diff > RT_PI)
    {
      angle_diff = angle_diff - 2*RT_PI;
    }
    if (angle_diff <= angle_diff_setting_distoturn)
    {
      double prev_pose_x_ = traj.points[i-1].pose.position.x;
      double prev_pose_y_ = traj.points[i-1].pose.position.y;
      double next_pose_x_ = traj.points[i].pose.position.x;
      double next_pose_y_ = traj.points[i].pose.position.y;
      double dis_waypoints_ = std::sqrt((next_pose_x_-prev_pose_x_)*(next_pose_x_-prev_pose_x_)+(next_pose_y_-prev_pose_y_)*(next_pose_y_-prev_pose_y_));
      distoturn = distoturn + dis_waypoints_;
    }
    else
    {
      break;
    }
  }
  std::cout << "index : " << index << std::endl;
  std::cout << "distoturn : " << distoturn << std::endl;
}

void calculate_LRturn(autoware_planning_msgs::Trajectory traj)
{
  double size = traj.points.size();
  double first_yaw = tf::getYaw(traj.points[0].pose.orientation);
  double last_yaw = tf::getYaw(traj.points[size-1].pose.orientation);
  double angle_diff = last_yaw - first_yaw;
  if (angle_diff < -RT_PI)
  {
    angle_diff = angle_diff + 2*RT_PI;
  }
  else if (angle_diff > RT_PI)
  {
    angle_diff = angle_diff - 2*RT_PI;
  }
  double angle_diff_setting = 0.0;
  if (LRturn == 0)
  {
    angle_diff_setting = angle_diff_setting_in;
  }
  else
  {
    angle_diff_setting = angle_diff_setting_out;
  }
  
  if (angle_diff > angle_diff_setting) //Left turn
  {
    LRturn = 1;
    calaulate_distoturn(traj);
  }
  else if (angle_diff < -angle_diff_setting) //Right turn
  {
    LRturn = 2;
    calaulate_distoturn(traj);
  }
  else //Straight
  {
    LRturn = 0;
    distoturn = 0.0;
  }

  std::cout << "first_yaw : " << first_yaw << std::endl;
  std::cout << "last_yaw : " << last_yaw << std::endl;
  std::cout << "angle_diff_setting_in : " << angle_diff_setting_in << std::endl;
  std::cout << "angle_diff_setting_out : " << angle_diff_setting_out << std::endl;
  std::cout << "angle_diff : " << angle_diff << std::endl;
  std::cout << "LRturn : " << LRturn << std::endl;
}

void FitCenterByLeastSquares(std::vector<Point3D> mapPoint, Point3D &centerP, double &radius) 
{
  double sumX = 0, sumY = 0;
  double sumXX = 0, sumYY = 0, sumXY = 0;
  double sumXXX = 0, sumXXY = 0, sumXYY = 0, sumYYY = 0;

  int pCount = mapPoint.size();
  for (int i = 0; i<pCount; i++)
  { 
    sumX += mapPoint[i].x;
    sumY += mapPoint[i].y;
    sumXX += pow(mapPoint[i].x,2);
    sumYY += pow(mapPoint[i].y,2);
    sumXY += mapPoint[i].x * mapPoint[i].y;
    sumXXX += pow(mapPoint[i].x,3);
    sumXXY += pow(mapPoint[i].x,2) * mapPoint[i].y;
    sumXYY += mapPoint[i].x * pow(mapPoint[i].y,2);
    sumYYY += pow(mapPoint[i].y,3);
  }

  double M1 = pCount * sumXY - sumX * sumY;
  double M2 = pCount * sumXX - sumX * sumX;
  double M3 = pCount * (sumXXX + sumXYY) - sumX * (sumXX + sumYY);
  double M4 = pCount * sumYY - sumY * sumY;
  double M5 = pCount * (sumYYY + sumXXY) - sumY * (sumXX + sumYY);

  double a = (M1 * M5 - M3 * M4) / (M2 * M4 - M1 * M1);
  double b = (M1 * M3 - M2 * M5) / (M2 * M4 - M1 * M1);
  double c = -(a * sumX + b * sumY + sumXX + sumYY) / pCount;

  //Circle center XY and radius
  double xCenter = -0.5 * a;
  double yCenter = -0.5 * b;
  radius = 0.5 * sqrt(a * a + b * b - 4 * c);
  centerP.x = xCenter;
  centerP.y = yCenter;

  //K : + or -
}

void calculate_curvature(autoware_planning_msgs::Trajectory traj)
{
  Point3D trajPoint;
  std::vector<Point3D> trajPoints;
  int traj_size = traj.points.size();
  for (int i = 0; i < traj_size; i++)
  {
    trajPoint.x = traj.points[i].pose.position.x;
    trajPoint.y = traj.points[i].pose.position.y;
    trajPoint.z = traj.points[i].pose.position.z; 
    trajPoints.push_back(trajPoint);
  }
  Point3D centerP;
  double radius;
  FitCenterByLeastSquares(trajPoints,centerP,radius);
  curvature = 1/radius;

  std::cout << "radius : " << radius << std::endl;
  std::cout << "curvature : " << curvature << std::endl;
}

// void calculate_slope(autoware_planning_msgs::Trajectory traj)
// {
//   distouphill = 0;
//   int traj_size = traj.points.size();
//   max_slope = 0;
//   double min_slope = RT_PI;
//   for (int i = 0; i < traj_size/seg_size; i++)
//   {
//     int j = i*seg_size - i;
//     double check_first_z = traj.points[j].pose.position.z;
//     double check_last_z = traj.points[j+seg_size-1].pose.position.z;
//     double z_diff = check_last_z - check_first_z;

//     double prev_pose_x_ = traj.points[j].pose.position.x;
//     double prev_pose_y_ = traj.points[j].pose.position.y;
//     double next_pose_x_ = traj.points[j+seg_size-1].pose.position.x;
//     double next_pose_y_ = traj.points[j+seg_size-1].pose.position.y;
//     double dis_waypoints_ = std::sqrt((next_pose_x_-prev_pose_x_)*(next_pose_x_-prev_pose_x_)+(next_pose_y_-prev_pose_y_)*(next_pose_y_-prev_pose_y_));

//     double check_slope = asin(z_diff/dis_waypoints_);

//     if (check_slope > max_slope)
//     {
//       max_slope = check_slope;
//     }
//     if (check_slope < min_slope)
//     {
//       min_slope = check_slope;
//     }
//     // if (check_slope > )
//     // distouphill = distouphill + dis_waypoints_;
//   }
//   std::cout << "max_slope : " << max_slope << std::endl;
// }

void calculate_slope(autoware_planning_msgs::Trajectory traj)
{
  distouphill = 0;
  int traj_size = traj.points.size();
  max_slope = 0;
  double min_slope = RT_PI;
  double dis_flag = 0;
  double index = 0;
  for (int i = 0; i < traj_size - seg_size; i++)
  {
    double check_first_z = traj.points[i].pose.position.z;
    double check_last_z = traj.points[i+seg_size-1].pose.position.z;
    double z_diff = check_last_z - check_first_z;

    double prev_pose_x_ = traj.points[i].pose.position.x;
    double prev_pose_y_ = traj.points[i].pose.position.y;
    double last_pose_x_ = traj.points[i+seg_size-1].pose.position.x;
    double last_pose_y_ = traj.points[i+seg_size-1].pose.position.y;
    double dis_waypoints_ = std::sqrt((last_pose_x_-prev_pose_x_)*(last_pose_x_-prev_pose_x_)+(last_pose_y_-prev_pose_y_)*(last_pose_y_-prev_pose_y_));

    double check_slope = asin(z_diff/dis_waypoints_);

    if (check_slope > max_slope)
    {
      max_slope = check_slope;
    }
    if (check_slope < min_slope)
    {
      min_slope = check_slope;
    }
    double next_pose_x_ = traj.points[i+1].pose.position.x;
    double next_pose_y_ = traj.points[i+1].pose.position.y;
    double dis_waypoints = std::sqrt((next_pose_x_-prev_pose_x_)*(next_pose_x_-prev_pose_x_)+(next_pose_y_-prev_pose_y_)*(next_pose_y_-prev_pose_y_));
    if (check_slope < slope_setting_distouphill && dis_flag == 0)
    {
      distouphill = distouphill + dis_waypoints;
      index = i;
    }
    else
    {
      dis_flag = 1;
    }
  }
  std::cout << "max_slope : " << max_slope << std::endl;
  std::cout << "distouphill : " << distouphill << std::endl;
  std::cout << "index : " << index << std::endl;
}

void calculate_uphill(autoware_planning_msgs::Trajectory traj)
{
  double size = traj.points.size();
  double first_z = traj.points[0].pose.position.z;
  double last_z = traj.points[size-1].pose.position.z;
  double z_diff = last_z - first_z;

  if (up_hill == false)
  {
    if (z_diff > z_diff_setting_in)
    {
      up_hill = true;
      calculate_slope(traj);
    }
    else
    {
      max_slope = 0;
    }
  }
  else
  {
    if (z_diff < z_diff_setting_out)
    {
      up_hill = false;
      max_slope = 0;
    }
    else
    {
      calculate_slope(traj);
    }
  }

  // std::cout << "first_z : " << first_z << std::endl;
  // std::cout << "last_z : " << last_z << std::endl;
  // std::cout << "z_diff_setting_in : " << z_diff_setting_in << std::endl;
  // std::cout << "z_diff_setting_out : " << z_diff_setting_out << std::endl;
  // std::cout << "z_diff : " << z_diff << std::endl;
  // std::cout << "up_hill : " << up_hill << std::endl;
}

void publish_CurrentTrajInfo()
{
  msgs::CurrentTrajInfo info;
  info.header.frame_id = "rear_wheel";
  info.header.stamp = ros::Time::now();
  info.LRturn = LRturn;
  info.Curvature = curvature;
  info.Distoturn = distoturn;
  info.Uphill = up_hill;
  info.MaxSlope = max_slope;
  info.Distouphill = distouphill;
  currenttrajinfo_pub.publish(info);
}

void transfer_callback(const autoware_planning_msgs::Trajectory& traj)
{
  calculate_LRturn(traj);
  calculate_curvature(traj);
  calculate_uphill(traj);
  publish_CurrentTrajInfo();
  
  nav_msgs::Path current_path;
  geometry_msgs::PoseStamped current_posestamped;

  current_posestamped.header.frame_id = traj.header.frame_id;
  current_posestamped.header.stamp = ros::Time::now();
  current_path.header.frame_id = traj.header.frame_id;
  current_path.header.stamp = ros::Time::now();
 
  int traj_size = traj.points.size();
  for (int i = 0; i < traj_size; i++)
  {
    current_posestamped.pose.position.x = traj.points[i].pose.position.x;
    current_posestamped.pose.position.y = traj.points[i].pose.position.y;
    current_posestamped.pose.position.z = traj.points[i].pose.position.z; 
    current_path.poses.push_back(current_posestamped);
  }
  nav_path_pub.publish(current_path);

  std_msgs::Empty empty_msg;
  nav_path_heartbeat_pub.publish(empty_msg);
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "path_transfer");
  ros::NodeHandle node;

  ros::param::get(ros::this_node::getName()+"/angle_diff_setting_in", angle_diff_setting_in);
  ros::param::get(ros::this_node::getName()+"/angle_diff_setting_out", angle_diff_setting_out);
  ros::param::get(ros::this_node::getName()+"/angle_diff_setting_distoturn", angle_diff_setting_distoturn);

  ros::param::get(ros::this_node::getName()+"/z_diff_setting_in", z_diff_setting_in);
  ros::param::get(ros::this_node::getName()+"/z_diff_setting_out", z_diff_setting_out);
  ros::param::get(ros::this_node::getName()+"/slope_setting_distouphill", slope_setting_distouphill);

  ros::Subscriber safety_waypoints_sub = node.subscribe("/planning/scenario_planning/trajectory", 1, transfer_callback);
  nav_path_pub = node.advertise<nav_msgs::Path>("nav_path_astar_final",1);
  currenttrajinfo_pub = node.advertise<msgs::CurrentTrajInfo>("current_trajectory_info",1);
  nav_path_heartbeat_pub = node.advertise<std_msgs::Empty>("nav_path_astar_final/heartbeat",1);

  ros::spin();
  return 0;
};