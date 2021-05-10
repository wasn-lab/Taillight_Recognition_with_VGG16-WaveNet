#include "xyz2lla.h"

namespace xyz2lla
{
void XYZ2LLA::testGNSS()
{
  int zone = 51;                  // taiwain 50 or 51
  double out_E_utm = 302331.52;   // 302331.52;
  double out_N_utm = 2741298.25;  // 2741298.25;

  double out_lat_wgs84, out_lon_wgs84;
  gnss_utm_tf_.UTMXYToLatLon(out_E_utm, out_N_utm, zone, false, out_lat_wgs84, out_lon_wgs84);

#if DEBUG == 1
  std::cout << "test : " << out_lon_wgs84 << std::endl;
  std::cout << "test : " << out_lat_wgs84 << std::endl;
#endif
}

void XYZ2LLA::initParam()
{
  double read_tmp[63];
  int read_index = 0;
  std::string fname = ros::package::getPath(root_);
  fname += path_param_xyz_to_enu;
#if DEBUG == 1
  std::cout << fname << std::endl;
#endif

  std::ifstream fin;
  char line[100];
  memset(line, 0, sizeof(line));

  fin.open(fname.c_str(), std::ios::in);
  if (!fin)
  {
    std::cout << "Fail to import txt" << std::endl;
    exit(1);
  }

  while (fin.getline(line, sizeof(line), ','))
  {
    // fin.getline(line,sizeof(line),'\n');
    std::string nmea_str(line);
    std::stringstream ss(nmea_str);
    std::string token;

    getline(ss, token, ',');
    read_tmp[read_index] = atof(token.c_str());
    read_index += 1;
  }

#if DEBUG == 1
  std::cout << read_tmp[10] << std::endl;
#endif

  lon0_ = read_tmp[0];
  lat0_ = read_tmp[1];
  alt0_ = read_tmp[2];

  if (XYZ2ENU_switch_ == false)
  {
    int idx = 3;
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++)
        R_[i][j] = read_tmp[idx++];

    for (int i = 0; i < 3; i++)
      T_[i] = read_tmp[idx++];
  }
  else
  {
    int idx = 3;
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++)
        R1_[i][j] = read_tmp[idx++];

    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++)
        R2_[i][j] = read_tmp[idx++];

    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++)
        R3_[i][j] = read_tmp[idx++];

    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++)
        R4_[i][j] = read_tmp[idx++];

    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++)
        R5_[i][j] = read_tmp[idx++];

    for (int i = 0; i < 3; i++)
      T1_[i] = read_tmp[idx++];

    for (int i = 0; i < 3; i++)
      T2_[i] = read_tmp[idx++];

    for (int i = 0; i < 3; i++)
      T3_[i] = read_tmp[idx++];

    for (int i = 0; i < 3; i++)
      T4_[i] = read_tmp[idx++];

    for (int i = 0; i < 3; i++)
      T5_[i] = read_tmp[idx++];

#if DEBUG == 1
    std::cout << "T5[0] : " << std::setprecision(20) << T5_[0] << std::endl;
#endif
  }

#if DEBUG == 1
  std::cout << "lon0 : " << std::setprecision(20) << lon0_ << std::endl;
  std::cout << "lat0 : " << std::setprecision(20) << lat0_ << std::endl;
  std::cout << "alt0 : " << std::setprecision(20) << alt0_ << std::endl;
#endif
}

void XYZ2LLA::initParamTWD97()
{
  double read_tmp_1[3];
  int read_index_1 = 0;
  std::string fname_1 = ros::package::getPath(root_);
  fname_1 += path_param_xyz_to_twd97;
#if DEBUG == 1
  std::cout << fname_1 << std::endl;
#endif

  std::ifstream fin_1;
  char line_1[100];
  memset(line_1, 0, sizeof(line_1));

  fin_1.open(fname_1.c_str(), std::ios::in);
  if (!fin_1)
  {
    std::cout << "Fail to import txt" << std::endl;
    exit(1);
  }

  while (fin_1.getline(line_1, sizeof(line_1), ','))
  {
    // fin_1.getline(line_1,sizeof(line_1),'\n');
    std::string nmea_str(line_1);
    std::stringstream ss(nmea_str);
    std::string token;

    getline(ss, token, ',');
    read_tmp_1[read_index_1] = atof(token.c_str());
    read_index_1 += 1;
  }
  twd97_shift_x_ = read_tmp_1[0];
  twd97_shift_y_ = read_tmp_1[1];
  twd97_shift_z_ = read_tmp_1[2];

#if DEBUG == 1
  std::cout << "twd97_shift_x_ : " << std::setprecision(20) << twd97_shift_x_ << std::endl;
  std::cout << "twd97_shift_y_ : " << std::setprecision(20) << twd97_shift_y_ << std::endl;
  std::cout << "twd97_shift_z_ : " << std::setprecision(20) << twd97_shift_z_ << std::endl;
#endif
}

void XYZ2LLA::initParamUTM()
{
  double read_tmp_2[3];
  int read_index_2 = 0;
  std::string fname_2 = ros::package::getPath(root_);
  fname_2 += path_param_xyz_to_utm;
#if DEBUG == 1
  std::cout << fname_2 << std::endl;
#endif

  std::ifstream fin_2;
  char line_2[100];
  memset(line_2, 0, sizeof(line_2));

  fin_2.open(fname_2.c_str(), std::ios::in);
  if (!fin_2)
  {
    std::cout << "Fail to import txt" << std::endl;
    exit(1);
  }

  while (fin_2.getline(line_2, sizeof(line_2), ','))
  {
    // fin_2.getline(line_2,sizeof(line_2),'\n');
    std::string nmea_str(line_2);
    std::stringstream ss(nmea_str);
    std::string token;

    getline(ss, token, ',');
    read_tmp_2[read_index_2] = atof(token.c_str());
    read_index_2 += 1;
  }
  utm_shift_x_ = read_tmp_2[0];
  utm_shift_y_ = read_tmp_2[1];
  utm_zone_ = read_tmp_2[2];

#if DEBUG == 1
  std::cout << "utm_shift_x_ : " << std::setprecision(20) << utm_shift_x_ << std::endl;
  std::cout << "utm_shift_y_ : " << std::setprecision(20) << utm_shift_y_ << std::endl;
  std::cout << "utm_zone : " << std::setprecision(20) << utm_zone_ << std::endl;
#endif
}

void XYZ2LLA::convert(double& out_lat_wgs84, double& out_lon_wgs84, double& out_alt_wgs84, double& out_E, double& out_N,
                      double& out_U, const double in_x, const double in_y, const double in_z)
{
  double R_final[3][3], T_final[3];

  if (XYZ2ENU_switch_ == false)
  {
    for (int i = 0; i < 3; i++)
    {
      for (int j = 0; j < 3; j++)
      {
        R_final[i][j] = R_[i][j];
        T_final[i] = T_[i];
      }
    }
  }
  else
  {
    for (int i = 0; i < 3; i++)
    {
      for (int j = 0; j < 3; j++)
      {
        if (in_x <= 0)
        {
          R_final[i][j] = R1_[i][j];
          T_final[i] = T1_[i];
        }
        else if (in_x > 0 && in_x <= 100)
        {
          R_final[i][j] = R2_[i][j];
          T_final[i] = T2_[i];
        }
        else if (in_x > 100 && in_x <= 225)
        {
          R_final[i][j] = R3_[i][j];
          T_final[i] = T3_[i];
        }
        else if (in_x > 225 && in_x <= 350)
        {
          R_final[i][j] = R4_[i][j];
          T_final[i] = T4_[i];
        }
        else
        {
          R_final[i][j] = R5_[i][j];
          T_final[i] = T5_[i];
        }
      }
    }
  }

  // XYZ to ENU
  double d[3] = { in_x, in_y, in_z };
  double d_new[3];
  for (int i = 0; i < 3; i++)
  {
    d_new[i] = R_final[i][0] * d[0] + R_final[i][1] * d[1] + R_final[i][2] * d[2] + T_final[i];
  }

  out_E = d_new[0];
  out_N = d_new[1];
  out_U = d_new[2];
#if DEBUG == 1
  std::cout << "out E : " << std::setprecision(20) << out_E << std::endl;
  std::cout << "out N : " << std::setprecision(20) << out_N << std::endl;
  std::cout << "out U : " << std::setprecision(20) << out_U << std::endl;
#endif

  // initial ecef
  geodetic_converter_.initialiseReference(lat0_, lon0_, alt0_);

  // out_ENU to out_LLA
  double out_lat = 0.;
  double out_lon = 0.;
  double out_alt = 0.;
  geodetic_converter_.enu2Geodetic(out_E, out_N, out_U, &out_lat, &out_lon, &out_alt);
#if DEBUG == 1
  std::cout << "out Lat : " << std::setprecision(20) << out_lat << std::endl;
  std::cout << "out Lon : " << std::setprecision(20) << out_lon << std::endl;
  std::cout << "out Alt : " << std::setprecision(20) << out_alt << std::endl;
#endif

#if TWD97 == 1
  // twd97 to wgs84
  double out_E_twd97 = in_x + twd97_shift_x_;
  double out_N_twd97 = in_y + twd97_shift_y_;
  double out_U_twd97 = in_z + twd97_shift_z_;
  bool pkm = false;
  out_alt_wgs84 = out_U_twd97;
  gnss_tf_.TWD97toWGS84(out_E_twd97, out_N_twd97, &out_lat_wgs84, &out_lon_wgs84, pkm);
#endif

#if UTM == 1
  // utm to wgs84
  int zone = utm_zone_;  // taiwain 50 or 51
  double out_E_utm = in_x + utm_shift_x_;
  double out_N_utm = in_y + utm_shift_y_;
  double out_U_utm = in_z;
  out_alt_wgs84 = out_U_utm;
  gnss_utm_tf_.UTMXYToLatLon(out_E_utm, out_N_utm, zone, false, out_lat_wgs84, out_lon_wgs84);
#endif

#if DEBUG == 1
  std::cout << "out Lat wgs84 : " << std::setprecision(20) << out_lat_wgs84 << std::endl;
  std::cout << "out Lon wgs84 : " << std::setprecision(20) << out_lon_wgs84 << std::endl;
  std::cout << "out Alt wgs84 : " << std::setprecision(20) << out_alt_wgs84 << std::endl;
#endif
}

void XYZ2LLA::centerPointGPS(msgs::DetectedObjectArray& output)
{
  for (auto& obj : output.objects)
  {
    double out_lat = 0.;
    double out_lon = 0.;
    double out_alt = 0.;
    double out_E = 0.;
    double out_N = 0.;
    double out_U = 0.;

    convert(out_lat, out_lon, out_alt, out_E, out_N, out_U, obj.center_point_gps.x, obj.center_point_gps.y,
            obj.center_point_gps.z);

    obj.center_point_gps.x = out_lat;
    obj.center_point_gps.y = out_lon;
    obj.center_point_gps.z = out_alt;

    std::cout << "[ID " << obj.track.id << "]" << std::endl;
    std::cout << "Lat (center_point_gps.x): " << std::setprecision(7) << obj.center_point_gps.x << std::endl;
    std::cout << "Lon (center_point_gps.y): " << std::setprecision(7) << obj.center_point_gps.y << std::endl;
    std::cout << "Alt (center_point_gps.z): " << std::setprecision(7) << obj.center_point_gps.z << std::endl;
    std::cout << "heading_enu: (" << std::setprecision(6) << obj.heading_enu.x << ", " << obj.heading_enu.y << ", "
              << obj.heading_enu.z << ", " << obj.heading_enu.w << ")" << std::endl;
  }
}

void XYZ2LLA::publishMsg(const msgs::DetectedObjectArray& output)
{
  pub_xyz2lla_.publish(output);
#if HEARTBEAT == 1
  std_msgs::Empty output_heartbeat;
  pub_xyz2lla_heartbeat_.publish(output_heartbeat);
#endif
}

int XYZ2LLA::convertClassID(const autoware_perception_msgs::Semantic& semantic)
{
  int res = 0;
  switch (semantic.type)
  {
    case 0:  // semantic.UNKNOWN
      res = sensor_msgs_itri::DetectedObjectClassId::Unknown;
      break;
    case 1:  // semantic.CAR
      res = sensor_msgs_itri::DetectedObjectClassId::Car;
      break;
    case 2:  // semantic.TRUCK
      res = sensor_msgs_itri::DetectedObjectClassId::Truck;
      break;
    case 3:  // semantic.BUS
      res = sensor_msgs_itri::DetectedObjectClassId::Bus;
      break;
    case 4:  // semantic.BICYCLE
      res = sensor_msgs_itri::DetectedObjectClassId::Bicycle;
      break;
    case 5:  // semantic.MOTORBIKE
      res = sensor_msgs_itri::DetectedObjectClassId::Motobike;
      break;
    case 6:  // semantic.PEDESTRIAN
      res = sensor_msgs_itri::DetectedObjectClassId::Person;
      break;
    case 7:  // semantic.ANIMAL
      res = sensor_msgs_itri::DetectedObjectClassId::Unknown;
      break;
    default:
      res = sensor_msgs_itri::DetectedObjectClassId::Unknown;
  }
  return res;
}

void XYZ2LLA::callbackTracking1(const autoware_perception_msgs::DynamicObjectArray::ConstPtr& input)
{
  msgs::DetectedObjectArray output;
  output.header = input->header;
  output.objects.reserve(input->objects.size());

  for (size_t i = 0; i < input->objects.size(); i++)
  {
    msgs::DetectedObject obj;
    obj.classId = convertClassID(input->objects[i].semantic);

    // take the last three uint8 of uuid_msgs/UniqueID to create float id
    obj.track.id = ((unsigned int)input->objects[i].id.uuid[13] << 16) +
                   ((unsigned int)input->objects[i].id.uuid[14] << 8) + (unsigned int)input->objects[i].id.uuid[15];

    obj.center_point_gps.x = input->objects[i].state.pose_covariance.pose.position.x;
    obj.center_point_gps.y = input->objects[i].state.pose_covariance.pose.position.y;
    obj.center_point_gps.z = input->objects[i].state.pose_covariance.pose.position.z;

    obj.heading_enu.x = input->objects[i].state.pose_covariance.pose.orientation.x;
    obj.heading_enu.y = input->objects[i].state.pose_covariance.pose.orientation.y;
    obj.heading_enu.z = input->objects[i].state.pose_covariance.pose.orientation.z;
    obj.heading_enu.w = input->objects[i].state.pose_covariance.pose.orientation.w;

    obj.dimension.length = input->objects[i].shape.dimensions.x;
    obj.dimension.width = input->objects[i].shape.dimensions.y;
    obj.dimension.height = input->objects[i].shape.dimensions.z;

    output.objects.push_back(obj);
  }

  centerPointGPS(output);
  publishMsg(output);
}

void XYZ2LLA::callbackTracking2(const msgs::DetectedObjectArray::ConstPtr& input)
{
  msgs::DetectedObjectArray output;
  output.header = input->header;
  output.objects.assign(input->objects.begin(), input->objects.end());

  // compute center_point_gps
  for (auto& obj : output.objects)
  {
    obj.center_point_gps.x = obj.center_point.x;
    obj.center_point_gps.y = obj.center_point.y;
    obj.center_point_gps.z = obj.center_point.z;

    if (obj.header.frame_id != "map")
    {
      // assign frame_id_source_
      if (obj.header.frame_id == "lidar" || obj.header.frame_id == "base_link")
      {
        frame_id_source_ = "base_link";
      }
      else
      {
        frame_id_source_ = obj.header.frame_id;
      }

      // get tf_stamped: base_link-to-map
      geometry_msgs::TransformStamped tf_stamped;
      try
      {
        tf_stamped = tf_buffer_.lookupTransform(frame_id_target_, frame_id_source_, obj.header.stamp);
      }
      catch (tf2::TransformException& ex1)
      {
        ROS_WARN("%s", ex1.what());
        try
        {
          tf_stamped = tf_buffer_.lookupTransform(frame_id_target_, frame_id_source_, ros::Time(0));
        }
        catch (tf2::TransformException& ex2)
        {
          ROS_WARN("%s", ex2.what());
          return;
        }
      }

      // TF (base_link-to-map) for object pose
      geometry_msgs::Pose pose_in_base_link;
      pose_in_base_link.position.x = obj.center_point.x;
      pose_in_base_link.position.y = obj.center_point.y;
      pose_in_base_link.position.z = obj.center_point.z;
      pose_in_base_link.orientation.x = 0;
      pose_in_base_link.orientation.y = 0;
      pose_in_base_link.orientation.z = 0;
      pose_in_base_link.orientation.w = 1;

      geometry_msgs::Pose pose_in_map;
      tf2::doTransform(pose_in_base_link, pose_in_map, tf_stamped);
      obj.center_point_gps.x = pose_in_map.position.x;
      obj.center_point_gps.y = pose_in_map.position.y;
      obj.center_point_gps.z = pose_in_map.position.z;
    }
  }

  centerPointGPS(output);
  publishMsg(output);
}

int XYZ2LLA::run()
{
  testGNSS();

  initParam();

#if TWD97 == 1
  initParamTWD97();
#endif

#if UTM == 1
  initParamUTM();
#endif

#if INPUT_DYNAMIC_OBJ == 1
  sub_xyz2lla_ = nh_.subscribe(in_topic1_, 1, &XYZ2LLA::callbackTracking1, this);
#else
  sub_xyz2lla_ = nh_.subscribe(in_topic2_, 1, &XYZ2LLA::callbackTracking2, this);
#endif

  pub_xyz2lla_ = nh_.advertise<msgs::DetectedObjectArray>(out_topic_, 1);
#if HEARTBEAT == 1
  pub_xyz2lla_heartbeat_ = nh_.advertise<std_msgs::Empty>(out_topic_ + std::string("/heartbeat"), 1);
#endif

  tf2_ros::TransformListener tf_listener(tf_buffer_);

  ros::Rate loop_rate(10);
  while (ros::ok())
  {
    ros::spinOnce();
    loop_rate.sleep();
  }

  return 0;
}
}  // namespace xyz2lla
