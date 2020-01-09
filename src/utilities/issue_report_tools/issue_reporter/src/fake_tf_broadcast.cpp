#include <ros/ros.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/TransformStamped.h>



void broadcast(std::string & frame_from, std::string & frame_to){
  static tf2_ros::TransformBroadcaster br;
  geometry_msgs::TransformStamped transformStamped;

  transformStamped.header.stamp = ros::Time::now();
  transformStamped.header.frame_id = frame_from;
  transformStamped.child_frame_id = frame_to;
  transformStamped.transform.translation.x = 0.0;
  transformStamped.transform.translation.y = 0.0;
  transformStamped.transform.translation.z = 0.0;
  transformStamped.transform.rotation.x = 0.0;
  transformStamped.transform.rotation.y = 0.0;
  transformStamped.transform.rotation.z = 0.0;
  transformStamped.transform.rotation.w = 1.0;

  br.sendTransform(transformStamped);
}

int main(int argc, char** argv){
  ros::init(argc, argv, "fake_tf_broadcast");

  std::string frame_from("base");
  std::string frame_to("base_1");
  if (argc >= 3){
      frame_from = std::string(argv[1]);
      frame_to = std::string(argv[2]);
  }

  ros::NodeHandle node;
  ros::Rate loop_rate(10);
  while (ros::ok()){
    broadcast(frame_from, frame_to);
    ros::spinOnce();
    loop_rate.sleep();

  }
  return 0;
};
