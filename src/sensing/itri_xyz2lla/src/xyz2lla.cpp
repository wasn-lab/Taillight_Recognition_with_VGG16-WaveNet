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
  fname += "/data/ITRI_NEW_XYZ2ENU_sec.txt";
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

  if (XYZ2ENU_siwtch_ == 0)
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
  fname_1 += "/data/ITRI_ShiftXYZ2TWD97.txt";
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
  fname_2 += "/data/ITRI_ShiftXYZ2UTM.txt";
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

  if (XYZ2ENU_siwtch_ == 0)
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

void XYZ2LLA::callbackTracking(const msgs::DetectedObjectArray::ConstPtr& input)
{
  msgs::DetectedObjectArray output;
  output.header = input->header;
  output.objects.assign(input->objects.begin(), input->objects.end());

  // compute center_point_gps
  for (auto& obj : output.objects)
  {
    double out_lat = 0.;
    double out_lon = 0.;
    double out_alt = 0.;
    double out_E = 0.;
    double out_N = 0.;
    double out_U = 0.;
    convert(out_lat, out_lon, out_alt, out_E, out_N, out_U, obj.center_point.x, obj.center_point.y, obj.center_point.z);
    obj.center_point_gps.x = out_lat;
    obj.center_point_gps.y = out_lon;
    obj.center_point_gps.z = out_alt;
  }

  // compute heading_enu
  for (auto& obj : output.objects)
  {
    tf::Quaternion rot(obj.heading.x, obj.heading.y, obj.heading.z, obj.heading.w);
    tf::Vector3 vec(1, 0, 0);
    tf::Vector3 rotated_vec = tf::quatRotate(rot, vec);
    double out_lat = 0.;
    double out_lon = 0.;
    double out_alt = 0.;
    double out_E = 0.;
    double out_N = 0.;
    double out_U = 0.;
    convert(out_lat, out_lon, out_alt, out_E, out_N, out_U, rotated_vec.x(), rotated_vec.y(), rotated_vec.z());
    Eigen::Vector3d A(1., 0., 0.);
    Eigen::Vector3d B(out_E, out_N, out_U);
    Eigen::Quaternion<double> R;
    R.setFromTwoVectors(A, B);
    obj.heading_enu.x = R.x();
    obj.heading_enu.y = R.y();
    obj.heading_enu.z = R.z();
    obj.heading_enu.w = R.w();
  }

  pub_xyz2lla_.publish(output);
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

  sub_xyz2lla_ = nh_.subscribe("Tracking3D", 1, &XYZ2LLA::callbackTracking, this);
  pub_xyz2lla_ = nh_.advertise<msgs::DetectedObjectArray>("Tracking3D/xyz2lla", 1);

  ros::spin();

  return 0;
}
}  // namespace xyz2lla