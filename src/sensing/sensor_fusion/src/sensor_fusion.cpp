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

void LidarDetectionCb(const msgs::DetectedObjectArray::ConstPtr& lidar_obj_array)
{
  msgLidarObj.header = lidar_obj_array->header;

  std::vector<msgs::DetectedObject>().swap(msgLidarObj.objects);
  msgLidarObj.objects.reserve(lidar_obj_array->objects.size());

  for (const auto& obj : lidar_obj_array->objects)
  {
    msgLidarObj.objects.push_back(obj);
  }

  fuseDetectedObjects();
}

void callback_camera_main(const msgs::DetectedObjectArray::ConstPtr& cam_obj_array,
                          msgs::DetectedObjectArray& msg_cam_obj)
{
  msg_cam_obj.header = cam_obj_array->header;

  std::vector<msgs::DetectedObject>().swap(msg_cam_obj.objects);
  msg_cam_obj.objects.reserve(cam_obj_array->objects.size());

  for (const auto& obj : cam_obj_array->objects)
  {
    if (obj.distance >= 0)
    {
      msg_cam_obj.objects.push_back(obj);
    }
  }
}

void cam60_0_DetectionCb(const msgs::DetectedObjectArray::ConstPtr& Cam60_0_ObjArray)
{
  callback_camera_main(Cam60_0_ObjArray, msgCam60_0_Obj);
}

void cam60_1_DetectionCb(const msgs::DetectedObjectArray::ConstPtr& Cam60_1_ObjArray)
{
  callback_camera_main(Cam60_1_ObjArray, msgCam60_1_Obj);
}

void cam60_2_DetectionCb(const msgs::DetectedObjectArray::ConstPtr& Cam60_2_ObjArray)
{
  callback_camera_main(Cam60_2_ObjArray, msgCam60_2_Obj);
}

void cam30_0_DetectionCb(const msgs::DetectedObjectArray::ConstPtr& Cam30_0_ObjArray)
{
  callback_camera_main(Cam30_0_ObjArray, msgCam30_0_Obj);
}

void cam30_1_DetectionCb(const msgs::DetectedObjectArray::ConstPtr& Cam30_1_ObjArray)
{
  callback_camera_main(Cam30_1_ObjArray, msgCam30_1_Obj);
}

void cam30_2_DetectionCb(const msgs::DetectedObjectArray::ConstPtr& Cam30_2_ObjArray)
{
  callback_camera_main(Cam30_2_ObjArray, msgCam30_2_Obj);
}

void cam120_0_DetectionCb(const msgs::DetectedObjectArray::ConstPtr& Cam120_0_ObjArray)
{
  callback_camera_main(Cam120_0_ObjArray, msgCam120_0_Obj);
}

void cam120_1_DetectionCb(const msgs::DetectedObjectArray::ConstPtr& Cam120_1_ObjArray)
{
  callback_camera_main(Cam120_1_ObjArray, msgCam120_1_Obj);
}

void cam120_2_DetectionCb(const msgs::DetectedObjectArray::ConstPtr& Cam120_2_ObjArray)
{
  callback_camera_main(Cam120_2_ObjArray, msgCam120_2_Obj);
}

void fuseDetectedObjects()
{
  std::cout << "**************** do_fusion ****************" << std::endl;

  std::vector<msgs::DetectedObject>().swap(vDetectedObjectDF);
  std::vector<msgs::DetectedObject>().swap(vDetectedObjectLID);
  std::vector<msgs::DetectedObject>().swap(vDetectedObjectCAM_60_0);
  std::vector<msgs::DetectedObject>().swap(vDetectedObjectCAM_60_1);
  std::vector<msgs::DetectedObject>().swap(vDetectedObjectCAM_60_2);
  std::vector<msgs::DetectedObject>().swap(vDetectedObjectCAM_30_1);
  std::vector<msgs::DetectedObject>().swap(vDetectedObjectCAM_120_1);

  /************************************************************************/

  std::cout << "num_lidar_objs = " << msgLidarObj.objects.size() << std::endl;

  for (const auto& obj : msgLidarObj.objects)
  {
    vDetectedObjectDF.push_back(obj);
  }

  /************************************************************************/

  // CamObjFrontCenter
  std::cout << "num_cam60_1_objs = " << msgCam60_1_Obj.objects.size() << std::endl;

  init_distance_table(vDetectedObjectDF, msgCam60_1_Obj.objects);
  associate_data(vDetectedObjectDF, msgCam60_1_Obj.objects);

  /************************************************************************/

  // CamObjFrontRight
  std::cout << "num_cam60_0_objs = " << msgCam60_0_Obj.objects.size() << std::endl;

  init_distance_table(vDetectedObjectDF, msgCam60_0_Obj.objects);
  associate_data(vDetectedObjectDF, msgCam60_0_Obj.objects);

  /************************************************************************/

  // CamObjFrontLeft
  std::cout << "num_cam60_2_objs = " << msgCam60_2_Obj.objects.size() << std::endl;

  init_distance_table(vDetectedObjectDF, msgCam60_2_Obj.objects);
  associate_data(vDetectedObjectDF, msgCam60_2_Obj.objects);

  /************************************************************************/

  std::cout << "num_cam30_0_objs = " << msgCam30_1_Obj.objects.size() << std::endl;

  init_distance_table(vDetectedObjectDF, msgCam30_1_Obj.objects);
  associate_data(vDetectedObjectDF, msgCam30_1_Obj.objects);

  /************************************************************************/

  std::cout << "num_cam120_0_objs = " << msgCam120_1_Obj.objects.size() << std::endl;

  init_distance_table(vDetectedObjectDF, msgCam120_1_Obj.objects);
  associate_data(vDetectedObjectDF, msgCam120_1_Obj.objects);

  /************************************************************************/

  std::cout << "num_total_objs = " << vDetectedObjectDF.size() << std::endl;

  msgFusionObj.header.stamp = msgLidarObj.header.stamp;
  msgFusionObj.header.frame_id = "lidar";
  msgFusionObj.header.seq = seq++;
  std::vector<msgs::DetectedObject>().swap(msgFusionObj.objects);
  msgFusionObj.objects.assign(vDetectedObjectDF.begin(), vDetectedObjectDF.end());

  fusion_pub.publish(msgFusionObj);
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

  HungarianAlgorithm HungAlgo;
  std::vector<int> assignment;

#if DEBUG
  double cost = HungAlgo.Solve(cost_mat, assignment);

  for (unsigned i = 0; i < cost_mat.size(); i++)
  {
    std::cout << i << "," << assignment[i] << "\t";
  }

  std::cout << "\ncost: " << cost << std::endl;
#else
  HungAlgo.Solve(cost_mat, assignment);
#endif

  std::vector<bool> assigned(s2, false);

  for (size_t i = 0; i < s1; i++)
  {
    if ((size_t)assignment[i] < s2)
    {
      if (cost_mat[i][assignment[i]] < FUSE_RANGE_SED)
      {
        objs1[i].camInfo.u = objs2[assignment[i]].camInfo.u;
        objs1[i].camInfo.v = objs2[assignment[i]].camInfo.v;
        objs1[i].camInfo.width = objs2[assignment[i]].camInfo.width;
        objs1[i].camInfo.height = objs2[assignment[i]].camInfo.height;
        objs1[i].camInfo.id = objs2[assignment[i]].camInfo.id;
        objs1[i].camInfo.prob = objs2[assignment[i]].camInfo.prob;

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

  ros::Subscriber lidar_det_sub = nh.subscribe("/LidarDetection", 1, LidarDetectionCb);
  // ros::Subscriber cam_F_right_sub = nh.subscribe("/CamObjFrontRight", 1, cam60_0_DetectionCb);
  ros::Subscriber cam_F_center_sub = nh.subscribe("/CamObjFrontCenter", 1, cam60_1_DetectionCb);
  // ros::Subscriber cam_F_left_sub = nh.subscribe("/CamObjFrontLeft", 1, cam60_2_DetectionCb);

  fusion_pub = nh.advertise<msgs::DetectedObjectArray>("SensorFusion", 2);

  signal(SIGINT, MySigintHandler);

  ros::MultiThreadedSpinner spinner(4);
  spinner.spin();
}
