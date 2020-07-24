#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <std_msgs/Float64.h>
#include <tf/tf.h>
#include <fstream>
#include <ros/package.h>
#include <cmath>

ros::Publisher goal_publisher;
ros::Publisher checkpoint_publisher;

double seg_x[2000] = {};
double seg_y[2000] = {};
double seg_z[2000] = {};
double ori_x[2000] = {};
double ori_y[2000] = {};
double ori_z[2000] = {};
double ori_w[2000] = {};
int read_index = 0;

template <int size_readtmp>
void read_txt(std::string fpname, double (&SEG_X)[size_readtmp],double (&SEG_Y)[size_readtmp],
double (&SEG_Z)[size_readtmp],double (&ORI_X)[size_readtmp],double (&ORI_Y)[size_readtmp],double (&ORI_Z)[size_readtmp],double (&ORI_W)[size_readtmp])
{
  std::string fname = fpname;

    std::ifstream fin;
    char line[300];
    memset( line, 0, sizeof(line));

    fin.open(fname.c_str(),std::ios::in);
    if(!fin) 
    {
        std::cout << "Fail to import txt" <<std::endl;
        exit(1);
    }

    while(fin.getline(line,sizeof(line),'\n')) 
    {
      std::string nmea_str(line);
      std::stringstream ss(nmea_str);
      std::string token;

      getline(ss,token, ',');
      SEG_X[read_index] = atof(token.c_str());
      getline(ss,token, ',');
      SEG_Y[read_index] = atof(token.c_str());
      getline(ss,token, ',');
      SEG_Z[read_index] = atof(token.c_str());
      getline(ss,token, ',');
      ORI_X[read_index] = atof(token.c_str());
      getline(ss,token, ',');
      ORI_Y[read_index] = atof(token.c_str());
      getline(ss,token, ',');
      ORI_Z[read_index] = atof(token.c_str());
      getline(ss,token, ',');
      ORI_W[read_index] = atof(token.c_str());
      read_index += 1;
    }
}

void Ini_obs_bytxt()
{
  std::string fpname = ros::package::getPath("mission_input");
  std::string fpname_s = fpname + "/data/ITRI_route_01.txt"; // full route

  read_txt(fpname_s, seg_x, seg_y, seg_z, ori_x, ori_y, ori_z, ori_w);
}

void get_goal_point()
{
  geometry_msgs::PoseStamped input_goal;

  input_goal.header.frame_id = "map";
  input_goal.header.stamp = ros::Time::now();

  input_goal.pose.position.x = seg_x[0];
  input_goal.pose.position.y = seg_y[0];
  input_goal.pose.position.z = seg_z[0];
  input_goal.pose.orientation.x = ori_x[0];
  input_goal.pose.orientation.y = ori_y[0];
  input_goal.pose.orientation.z = ori_z[0];
  input_goal.pose.orientation.w = ori_w[0];

  goal_publisher.publish(input_goal);
  std::cout << input_goal.pose.position.x << std::endl;

  ros::Rate loop_rate(1);
  ros::spinOnce();
  loop_rate.sleep();
}

void get_checkpoint_point()
{
  geometry_msgs::PoseStamped input_checkpoint;
 
  for (int i=1;i<read_index;i++)
  {
    input_checkpoint.header.frame_id = "map";
    input_checkpoint.header.stamp = ros::Time::now();
    input_checkpoint.pose.position.x = seg_x[i];
    input_checkpoint.pose.position.y = seg_y[i];
    input_checkpoint.pose.position.z = seg_z[i];
    input_checkpoint.pose.orientation.x = ori_x[i];
    input_checkpoint.pose.orientation.y = ori_y[i];
    input_checkpoint.pose.orientation.z = ori_z[i];
    input_checkpoint.pose.orientation.w = ori_w[i];

    checkpoint_publisher.publish(input_checkpoint);
    std::cout << i << std::endl;
    ros::Rate loop_rate(1);
    ros::spinOnce();
    loop_rate.sleep();
  }
  
}


int main(int argc, char** argv)
{
  ros::init(argc, argv, "mission_input");
  ros::NodeHandle node;

  Ini_obs_bytxt();
  goal_publisher = node.advertise<geometry_msgs::PoseStamped>("/move_base_simple/goal", 10, true);
  checkpoint_publisher = node.advertise<geometry_msgs::PoseStamped>("/checkpoint", 10, true);
  
  
  get_goal_point();
  get_checkpoint_point();
  
	
  //ros::spin();
  return 0;
};
