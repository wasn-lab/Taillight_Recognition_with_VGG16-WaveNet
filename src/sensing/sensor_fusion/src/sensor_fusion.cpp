#include "sensor_fusion.h"

void MySigintHandler(int sig)
{
  ROS_INFO("****** MySigintHandler ******");
  if (sig == SIGINT)
  {
    ROS_INFO("END SensorFusion");
    ros::shutdown();
  }
}

void callback_lidar(const msgs::DetectedObjectArray::ConstPtr& lidar_obj_array)
{
  lidar_msg.header = lidar_obj_array->header;

  std::vector<msgs::DetectedObject>().swap(lidar_msg.objects);
  lidar_msg.objects.reserve(lidar_obj_array->objects.size());

  for (const auto& obj : lidar_obj_array->objects)
  {
    lidar_msg.objects.push_back(obj);
  }

  fuseDetectedObjects();
}

void callback_camera_main(const msgs::DetectedObjectArray::ConstPtr& camera_obj_array,
                          msgs::DetectedObjectArray& camera_msg)
{
  camera_msg.header = camera_obj_array->header;

  std::vector<msgs::DetectedObject>().swap(camera_msg.objects);
  camera_msg.objects.reserve(camera_obj_array->objects.size());

  for (const auto& obj : camera_obj_array->objects)
  {
    if (obj.distance >= 0)
    {
      camera_msg.objects.push_back(obj);
    }
  }
}

void callback_camera(const msgs::DetectedObjectArray::ConstPtr& camera_obj_array)
{
  callback_camera_main(camera_obj_array, camera_msg);
}

void fuseDetectedObjects()
{
  std::cout << "**************** do_fusion ****************" << std::endl;

  fusion_msg.header.seq = ++seq;
  fusion_msg.header.stamp = lidar_msg.header.stamp;
  fusion_msg.header.frame_id = "lidar";
  std::vector<msgs::DetectedObject>().swap(fusion_msg.objects);

  std::cout << "num_lidar_objs = " << lidar_msg.objects.size() << std::endl;
  std::cout << "num_camera_objs = " << camera_msg.objects.size() << std::endl;

  if (!lidar_msg.objects.empty())
  {
    for (const auto& obj : lidar_msg.objects)
    {
      fusion_msg.objects.push_back(obj);
    }
    // Data association via Hungarian algo
    init_distance_table(fusion_msg.objects, camera_msg.objects);
    associate_data(fusion_msg.objects, camera_msg.objects);
  }

  std::cout << "num_fusion_objs = " << fusion_msg.objects.size() << std::endl;
  fusion_pub.publish(fusion_msg);
}

void get_obj_center(double& obj_cx, double obj_cy, const msgs::DetectedObject& obj)
{
  if (obj.cPoint.lowerAreaPoints.empty())
  {
    obj_cx = (obj.bPoint.p0.x + obj.bPoint.p6.x) / 2.;
    obj_cy = (obj.bPoint.p0.y + obj.bPoint.p6.y) / 2.;
  }
  else
  {
    obj_cx = 0.;
    obj_cy = 0.;

    for (const auto& p : obj.cPoint.lowerAreaPoints)
    {
      obj_cx += p.x;
      obj_cy += p.y;
    }

    obj_cx /= (double)obj.cPoint.lowerAreaPoints.size();
    obj_cy /= (double)obj.cPoint.lowerAreaPoints.size();
  }
}

void init_distance_table(std::vector<msgs::DetectedObject>& objs1, std::vector<msgs::DetectedObject>& objs2)
{
  std::vector<std::vector<double> >().swap(distance_table_);
  distance_table_.resize(objs1.size(), std::vector<double>(objs2.size(), FUSE_INVALID));

  for (size_t i = 0; i < objs1.size(); i++)
  {
    double obj_c1x = 0.;
    double obj_c1y = 0.;
    get_obj_center(obj_c1x, obj_c1y, objs1[i]);

    for (size_t j = 0; j < objs2.size(); j++)
    {
      double obj_c2x = 0.;
      double obj_c2y = 0.;
      get_obj_center(obj_c2x, obj_c2y, objs2[j]);

      double diff_x = obj_c1x - obj_c2x;
      double diff_y = obj_c1y - obj_c2y;
      distance_table_[i][j] = std::pow(diff_x, 2) + std::pow(diff_y, 2);
    }
  }
}

void associate_data(std::vector<msgs::DetectedObject>& objs1, std::vector<msgs::DetectedObject>& objs2)
{
  size_t s1 = objs1.size();
  size_t s2 = objs2.size();

  unsigned cost_mat_size = std::max(s1, s2);
  std::vector<std::vector<double> > cost_mat(cost_mat_size, vector<double>(cost_mat_size, 0.));

  for (size_t i = 0; i < s1; i++)
  {
    for (size_t j = 0; j < s2; j++)
    {
      cost_mat[i][j] = (double)distance_table_[i][j];
    }
  }

#if DEBUG
  std::cout << "cost matrix: " << std::endl;

  for (unsigned i = 0; i < cost_mat.size(); i++)
  {
    for (unsigned j = 0; j < cost_mat.size(); j++)
    {
      std::cout << cost_mat[i][j] << ", ";
    }
    std::cout << std::endl;
  }
#endif

  Hungarian hun;
  std::vector<int> assignment;

#if DEBUG
  double cost = hun.solve(cost_mat, assignment);

  for (unsigned i = 0; i < cost_mat.size(); i++)
  {
    std::cout << i << "," << assignment[i] << "\t";
  }

  std::cout << "\ncost: " << cost << std::endl;
#else
  hun.solve(cost_mat, assignment);
#endif

  std::vector<bool> assigned(s2, false);

  for (size_t i = 0; i < s1; i++)
  {
    if ((size_t)assignment[i] < s2)
    {
      if (cost_mat[i][assignment[i]] < FUSE_RANGE_SED)
      {
        objs1[i].camInfo = objs2[assignment[i]].camInfo;
        assigned[assignment[i]] = true;
      }
    }
  }

  for (size_t i = 0; i < s2; i++)
  {
    if (!assigned[i])
    {
      objs1.push_back(objs2[i]);
    }
  }
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "sensor_fusion");
  ros::NodeHandle nh;

  fusion_pub = nh.advertise<msgs::DetectedObjectArray>("SensorFusion", 1);

  ros::Subscriber lidar_sub = nh.subscribe("LidarDetection", 1, callback_lidar);
  ros::Subscriber camera_sub = nh.subscribe("CameraDetection/polygon", 1, callback_camera);

  signal(SIGINT, MySigintHandler);

  ros::MultiThreadedSpinner spinner(2);
  spinner.spin();
}
