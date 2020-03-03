#include "drivenet/distance_estimation_b1_v2.h"

DistanceEstimation::~DistanceEstimation()
{
  if (de_mode == 1)
  {
    for (int i = 0; i < img_h; i++)
    {
      delete[] align_FC60[i];
    }
    delete[] align_FC60;
  }

  delete[] arr_params;
  delete[] ShrinkArea;
  delete[] area;

}

void DistanceEstimation::init(int car_id, std::string pkgPath, int mode)
{
  carId = car_id;
  de_mode = mode;

  // ====== Init distance estimation table by alignment ======

  // FC60
  if (de_mode == 1)
  {
    std::string FC60Json = pkgPath;
    FC60Json.append("/data/alignment/out.json");
    align_FC60 = new cv::Point3d*[img_al_h];
    for (int i = 0; i < img_al_h; i++)
    {
      align_FC60[i] = new cv::Point3d[img_al_w];
    }
    ReadDistanceFromJson(FC60Json, align_FC60, img_al_h, img_al_w);
  }

  // // FR60
  // std::string FR60Json = pkgPath;
  // FR60Json.append("/data/alignment/FR60_2.json");
  // align_FR60 = new cv::Point3d*[img_h];
  // for (int i = 0; i < img_h; i++)
  // {
  //   align_FR60[i] = new cv::Point3d[img_w];
  // }
  // ReadDistanceFromJson(FR60Json, align_FR60, img_h, img_w);

  // // FL60
  // std::string FL60Json = pkgPath;
  // FL60Json.append("/data/alignment/FL60_2.json");
  // align_FL60 = new cv::Point3d*[img_h];
  // for (int i = 0; i < img_h; i++)
  // {
  //   align_FL60[i] = new cv::Point3d[img_w];
  // }
  // ReadDistanceFromJson(FL60Json, align_FL60, img_h, img_w);

  arr_params = new DisEstiParams[camera::id::num_ids];
  ShrinkArea = new CheckArea[camera::id::num_ids];
  area = new CheckArea[camera::id::num_ids];
  initParams();
  initShrinkArea();
  initDetectArea();

  Lidar_offset_x = 0;
  Lidar_offset_y = 0;
}

void DistanceEstimation::initParams()
{ 
  // camId: 0 (Front Center)
  arr_params[camera::id::front_bottom_60].regionHeight_x = { 1207, 1181, 1141, 1110,       1086 /*10*/, 1070, 1052, 1039, 1028, 1019, 1009,
                             1003, 996,  991,  985 /*20*/, 960,         946,  934,  926,  919,  914 /*50*/ };
  arr_params[camera::id::front_bottom_60].regionDist_x = { 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 25, 30, 35, 40, 45, 50 };
  arr_params[camera::id::front_bottom_60].regionHeightSlope_y = { 0.12,  0.209, 0.27,   0.337,  0.44,   1.02,  3.87,
                                  -1.53, -0.66, -0.452, -0.333, -0.251, -0.121 };
  arr_params[camera::id::front_bottom_60].regionHeight_y = { -1817, -617, -252, 0, 242, 608, 913, 1220, 1510, 1746, 2016, 2346, 3801 };
  arr_params[camera::id::front_bottom_60].regionDist_y = { 10, 5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5, -10 };

  // camId: 4 (Front Top)
  arr_params[camera::id::front_top_close_120].regionHeight_x = { 1207, 1002, 740, 574, 460, 379, 320, 272, 231, 198, 171,
                              150,  130,  115, 99,  86,  75,  65,  57,  48,  40,  10 };
  arr_params[camera::id::front_top_close_120].regionDist_x = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 25 };
  arr_params[camera::id::front_top_close_120].regionHeight_y = { -1422, -1131, -824, -412, 70,   490,  942,  1292,
                              1732,  2258,  2641, 3030, 3471, 3619, 3709, 3548 };
  arr_params[camera::id::front_top_close_120].regionHeightSlope_y = { 0.603, 0.682,   0.784,   1.012,   1.56,    2.908,   48.28,   -4.4615,
                                   -1.8,  -1.0328, -0.7976, -0.6509, -0.5349, -0.5156, -0.5161, -0.5862 };
  arr_params[camera::id::front_top_close_120].regionDist_y = { 6, 5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5, -6, -7, -8, -9 };

  // camId: 5 (Right Front)
  arr_params[camera::id::right_front_60].regionHeight_x = { 1148, 830, 544, 377, 236, 157, 52 };  // 5 to 10(~1m), 20 to 50m (~5m) //Horizontal line
  arr_params[camera::id::right_front_60].regionDist_x = { 0, 1, 2, 3, 4, 5, 6 };                  // 5 to 10, 20 to 50m (~5m)
  arr_params[camera::id::right_front_60].regionHeight_y = { 2507, 1904, 1498, 1032, 637, 357, -80 };  //-2 to 0 to 2(~1m) //Vertical line
  arr_params[camera::id::right_front_60].regionHeightSlope_y = { -1.2286, -2.4524, -6.000, 21.3529, 4.7308, 3.0297, 1.8171 };
  arr_params[camera::id::right_front_60].regionDist_y = { -6, -5, -4, -3, -2, -1, -0 };  //-2 to 0 to 2 (~1m)

  // camId: 6 (Right Back)
  arr_params[camera::id::right_back_60].regionHeight_x = { 1194, 838, 565, 395, 253, 138, 63, 17 };   // 5 to 10(~1m), 20 to 50m (~5m) //Horizontal
                                                                         // line
  arr_params[camera::id::right_back_60].regionDist_x = { 0, 1, 2, 3, 4, 5, 6, 7 };                    // 5 to 10, 20 to 50m (~5m)
  arr_params[camera::id::right_back_60].regionHeight_y = { 2049, 1688, 1209, 714, 217, -114, -738 };  //-2 to 0 to 2(~1m) //Vertical line
  arr_params[camera::id::right_back_60].regionHeightSlope_y = { -1.7722, -2.1614, -6.4409, 6.9259, 2.1378, 1.6333, 0.9539 };
  arr_params[camera::id::right_back_60].regionDist_y = { -9, -8, -7, -6, -5, -4, -3 };  //-2 to 0 to 2 (~1m)

  // camId: 10 (Back Top)
  arr_params[camera::id::back_top_120].regionHeight_x = { 1207, 836, 650, 532, 435, 367, 316, 270, 240, 210, 182, 161, 143 };
  arr_params[camera::id::back_top_120].regionDist_x = { 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20 };
  arr_params[camera::id::back_top_120].regionHeight_y = { -1566, -1230, -861, -264, 40, 475, 875, 1370, 1704, 2195, 2439, 2808, 3152 };
  arr_params[camera::id::back_top_120].regionHeightSlope_y = { 0.536,  0.612,  0.7197, 1.063,   1.372,  2.624, 14.2,
                                   -2.951, -1.727, -1.167, -0.9098, -0.724, -0.608 };
  arr_params[camera::id::back_top_120].regionDist_y = { 6, 5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5, -6 };
}

void DistanceEstimation::initShrinkArea()
{
  // Area needs to shrink
  // From x 6 - 50 m, y -3 to +3 m.
  ShrinkArea[camera::id::front_bottom_60].LeftLinePoint1 = cv::Point(869, 914);
  ShrinkArea[camera::id::front_bottom_60].LeftLinePoint2 = cv::Point(0, 1207);
  ShrinkArea[camera::id::front_bottom_60].RightLinePoint1 = cv::Point(1097, 914);
  ShrinkArea[camera::id::front_bottom_60].RightLinePoint2 = cv::Point(1746, 1207);

  // From x 0 - 7 m, y -2 to +2 m.
  ShrinkArea[camera::id::front_top_close_120].LeftLinePoint1 = cv::Point(510, 272);
  ShrinkArea[camera::id::front_top_close_120].LeftLinePoint2 = cv::Point(-412, 1207);
  ShrinkArea[camera::id::front_top_close_120].RightLinePoint1 = cv::Point(1351, 272);
  ShrinkArea[camera::id::front_top_close_120].RightLinePoint2 = cv::Point(2258, 1207);

  // From x -8 to -20 m, y -3 to +3 m.
  ShrinkArea[camera::id::back_top_120].LeftLinePoint1 = cv::Point(737, 143);
  ShrinkArea[camera::id::back_top_120].LeftLinePoint2 = cv::Point(-264, 1207);
  ShrinkArea[camera::id::back_top_120].RightLinePoint1 = cv::Point(1182, 143);
  ShrinkArea[camera::id::back_top_120].RightLinePoint2 = cv::Point(2195, 1207);
}

void DistanceEstimation::initDetectArea()
{
  area[camera::id::front_bottom_60].LeftLinePoint1 = cv::Point(636, 914);
  area[camera::id::front_bottom_60].LeftLinePoint2 = cv::Point(-1817, 1207);
  area[camera::id::front_bottom_60].RightLinePoint1 = cv::Point(1371, 914);
  area[camera::id::front_bottom_60].RightLinePoint2 = cv::Point(3801, 1207);

  area[camera::id::front_top_close_120].LeftLinePoint1 = cv::Point(294, 171);
  area[camera::id::front_top_close_120].LeftLinePoint2 = cv::Point(-1422, 1207);
  area[camera::id::front_top_close_120].RightLinePoint1 = cv::Point(1783, 171);
  area[camera::id::front_top_close_120].RightLinePoint2 = cv::Point(3548, 1207);

  area[camera::id::back_top_120].LeftLinePoint1 = cv::Point(422, 143);
  area[camera::id::back_top_120].LeftLinePoint2 = cv::Point(-1566, 1207);
  area[camera::id::back_top_120].RightLinePoint1 = cv::Point(1400, 143);
  area[camera::id::back_top_120].RightLinePoint2 = cv::Point(3152, 1207);
}

int DistanceEstimation::ReadDistanceFromJson(const std::string& filename, cv::Point3d** dist_in_cm, const int rows,
                                             const int cols)
{
  // dist_in_cm should be malloc by caller.
  assert(dist_in_cm);
  for (int i = 0; i < rows; i++)
  {
    assert(dist_in_cm[i]);
  }

  std::ifstream ifs(filename);
  Json::Reader jreader;
  Json::Value jdata;
  jreader.parse(ifs, jdata);
  std::cout << "Reading json file: " << filename << std::endl;

  for (Json::ArrayIndex i = 0; i < jdata.size(); i++)
  {
    auto image_x = jdata[i]["im_x"].asInt();
    auto image_y = jdata[i]["im_y"].asInt();

    if ((image_y < rows) && (image_x < cols))
    {
      dist_in_cm[image_y][image_x].x = jdata[i]["dist_in_cm"][0].asInt();
      dist_in_cm[image_y][image_x].y = jdata[i]["dist_in_cm"][1].asInt();
      dist_in_cm[image_y][image_x].z = jdata[i]["dist_in_cm"][2].asInt();
    }
  }
  return 0;
}

float DistanceEstimation::ComputeObjectXDist(int piexl_loc, std::vector<int> regionHeight,
                                             std::vector<float> regionDist)
{
  float distance = -1;
  float unitLength = 0.0;
  int bias = 0;
  float offset = 0.0;
  for (size_t i = 1; i < regionHeight.size(); i++)
  {
    if (piexl_loc >= regionHeight[i] && piexl_loc <= regionHeight[i - 1])
    {
      int regionpixel = regionHeight[i - 1] - regionHeight[i];
      int regionMeter = regionDist[i] - regionDist[i - 1];
      unitLength = float(regionMeter) / float(regionpixel);
      bias = piexl_loc - regionHeight[i];
      offset = unitLength * float(bias);
      distance = regionDist[i] - offset;
      break;
    }
    else if (piexl_loc <= regionHeight[i] && piexl_loc >= regionHeight[i - 1])
    {
      int regionpixel = regionHeight[i] - regionHeight[i - 1];
      int regionMeter = regionDist[i] - regionDist[i - 1];
      unitLength = float(regionMeter) / float(regionpixel);
      bias = regionHeight[i] - piexl_loc;
      offset = unitLength * float(bias);
      distance = regionDist[i] - offset;
      break;
    }
    else
    {
      if (piexl_loc > regionHeight.front())
      {
        distance = regionDist.front() - 0.2;
      }
      else if (piexl_loc < regionHeight.back())
      {
        distance = 777;
      }
      // distance = regionDist.back() + 0.2;
    }
  }
  return distance;
}

float DistanceEstimation::ComputeObjectXDistWithSlope(int pixel_loc_x, int pixel_loc_y, std::vector<int> regionHeight,
                                                      std::vector<float> regionHeightSlope,
                                                      std::vector<float> regionDist)
{
  float distance = -1;
  float unitLength = 0.0;
  int bias = 0;
  float offset = 0.0;

  std::vector<int> regionHeight_new = regionHeight;

  for (size_t i = 0; i < regionHeight.size(); i++)
  {
    if (regionHeightSlope[i] != 0)
    {
      regionHeight_new[i] = regionHeight[i] - int((regionHeightSlope[i]) * pixel_loc_x);
    }
    else
    {
      regionHeight_new[i] = regionHeight[i];
    }
  }

  for (size_t i = 1; i < regionHeight_new.size(); i++)
  {
    if (pixel_loc_y >= regionHeight_new[i] && pixel_loc_y <= regionHeight_new[i - 1])
    {
      int regionpixel = regionHeight_new[i - 1] - regionHeight_new[i];
      int regionMeter = regionDist[i - 1] - regionDist[i];
      if (regionpixel != 0)
      {
        unitLength = float(regionMeter) / float(regionpixel);
      }
      bias = pixel_loc_y - regionHeight_new[i];
      offset = unitLength * float(bias);
      distance = regionDist[i] - offset;
      break;
    }
    else if (pixel_loc_y <= regionHeight_new[i] && pixel_loc_y >= regionHeight_new[i - 1])
    {
      int regionpixel = regionHeight_new[i] - regionHeight_new[i - 1];
      int regionMeter = regionDist[i] - regionDist[i - 1];
      if (regionpixel != 0)
      {
        unitLength = float(regionMeter) / float(regionpixel);
      }
      bias = regionHeight_new[i] - pixel_loc_y;
      offset = unitLength * float(bias);
      distance = regionDist[i] - offset;
      break;
    }
    else
    {
      if (pixel_loc_y > regionHeight_new.front())
      {
        distance = regionDist.front() + 0.5;
      }
      else if (pixel_loc_y < regionHeight_new.back())
      {
        distance = regionDist.back() - 0.5;
      }
    }
  }

  return distance;
}

float DistanceEstimation::ComputeObjectYDist(int piexl_loc_y, int piexl_loc_x, std::vector<int> regionHeight,
                                             std::vector<float> regionHeightSlope_y, std::vector<float> regionDist,
                                             int img_h)
{
  float distance = -1;
  float unitLength = 0.0;
  int bias = 0;
  float offset = 0.0;

  std::vector<int> regionHeight_new = regionHeight;

  for (size_t i = 0; i < regionHeight.size(); i++)
  {
    int y = img_h - piexl_loc_x;
    if (regionHeightSlope_y[i] != 0)
    {
      regionHeight_new[i] = regionHeight[i] + int((1 / regionHeightSlope_y[i]) * y);
    }
    else
    {
      regionHeight_new[i] = regionHeight[i];
    }
  }

  for (size_t i = 1; i < regionHeight_new.size(); i++)
  {
    if (piexl_loc_y >= regionHeight_new[i] && piexl_loc_y <= regionHeight_new[i - 1])
    {
      int regionpixel = regionHeight_new[i - 1] - regionHeight_new[i];
      int regionMeter = regionDist[i] - regionDist[i - 1];
      if (regionpixel != 0)
      {
        unitLength = float(regionMeter) / float(regionpixel);
      }
      bias = piexl_loc_y - regionHeight_new[i];
      offset = unitLength * float(bias);
      distance = regionDist[i] - offset;
      break;
    }
    else if (piexl_loc_y <= regionHeight_new[i] && piexl_loc_y >= regionHeight_new[i - 1])
    {
      int regionpixel = regionHeight_new[i] - regionHeight_new[i - 1];
      int regionMeter = regionDist[i] - regionDist[i - 1];
      if (regionpixel != 0)
      {
        unitLength = float(regionMeter) / float(regionpixel);
      }
      bias = regionHeight_new[i] - piexl_loc_y;
      offset = unitLength * float(bias);
      distance = regionDist[i] - offset;
      break;
    }
    else
    {
      if (piexl_loc_y < regionHeight_new.front())
      {
        distance = regionDist.front() + 0.5;
      }
      else if (piexl_loc_y > regionHeight_new.back())
      {
        distance = regionDist.back() - 0.5;
      }
    }
  }

  return distance;
}

int DistanceEstimation::CheckPointInArea(CheckArea area, int object_x1, int object_y2)
{
  int point0 = 0;
  int point1 = 1;
  int point2 = 2;
  /// right
  int C1 = (area.RightLinePoint1.x - area.RightLinePoint2.x) * (object_y2 - area.RightLinePoint2.y) -
           (object_x1 - area.RightLinePoint2.x) * (area.RightLinePoint1.y - area.RightLinePoint2.y);
  /// left
  int C2 = (area.LeftLinePoint1.x - area.LeftLinePoint2.x) * (object_y2 - area.LeftLinePoint2.y) -
           (object_x1 - area.LeftLinePoint2.x) * (area.LeftLinePoint1.y - area.LeftLinePoint2.y);

  if (C1 > 0)
  {
    return point2;
  }
  else if (C2 < 0)
  {
    return point1;
  }
  else
  {
    return point0;
  }
}

float DistanceEstimation::RatioDefine(camera::id cam_id, int cls)
{
  if (cam_id == camera::id::front_bottom_60)
  {
    switch (cls)
    {
      // 0:person
      case 0:
        return 1;

      // 1:bicycle, 3:motorbike
      case 1:
        return 0.9;

      // 2:car
      case 2:
        return 0.7;

      case 3:
        return 0.9;

      // 5:bus, 7:truck
      case 5:
        return 0.7;

      case 7:
        return 0.7;

      default:
        return 1;
    }
  }
  else if (cam_id == camera::id::front_top_close_120)
  {
    switch (cls)
    {
      case 0:
        return 1;
      // 0:person

      // 1:bicycle, 3:motorbike
      case 1:
        return 0.9;

      // 2:car
      case 2:
        return 0.7;

      case 3:
        return 0.9;

      // 5:bus, 7:truck
      case 5:
        return 0.5;

      case 7:
        return 0.5;

      default:
        return 1;
    }
  }
  else if (cam_id == camera::id::back_top_120)
  {
    switch (cls)
    {
      case 0:
        // 0:person
        return 1;

      // 1:bicycle, 3:motorbike
      case 1:
        return 0.9;

      // 2:car
      case 2:
        return 0.7;

      case 3:
        return 0.9;

      // 5:bus, 7:truck
      case 5:
        return 0.5;

      case 7:
        return 0.5;

      default:
        return 1;
    }
  }
  return 1;
}

int DistanceEstimation::BoxShrink(camera::id cam_id, std::vector<int> Points_src, std::vector<int>& Points_dst)
{
  // PointsSrc = {class_id, x1, x2, y2};
  // PointsDst = {class_id, x1, x2, y2};

  // int edge_left, edge_right;
  int area_id_R = 1;  // 1: left, 2:right
  int area_id_L = 1;  // 1: left, 2:right

  double shrink_ratio;

  if (cam_id == camera::id::front_bottom_60)
  {
    area_id_L = CheckPointInArea(ShrinkArea[camera::id::front_bottom_60], Points_src[1], Points_src[3]);
    area_id_R = CheckPointInArea(ShrinkArea[camera::id::front_bottom_60], Points_src[2], Points_src[3]);
  }
  else if (cam_id == camera::id::front_top_close_120)
  {
    area_id_L = CheckPointInArea(ShrinkArea[camera::id::front_top_close_120], Points_src[1], Points_src[3]);
    area_id_R = CheckPointInArea(ShrinkArea[camera::id::front_top_close_120], Points_src[2], Points_src[3]);
  }
  else if (cam_id == camera::id::back_top_120)
  {
    area_id_L = CheckPointInArea(ShrinkArea[camera::id::back_top_120], Points_src[1], Points_src[3]);
    area_id_R = CheckPointInArea(ShrinkArea[camera::id::back_top_120], Points_src[2], Points_src[3]);
  }

  shrink_ratio = RatioDefine(cam_id, Points_src[0]);

  // Shrink box when one of x1, x2 is in area and another is not in the area.
  if (area_id_L != area_id_R)
  {
    if (area_id_L == 1 && area_id_R == 0)
    {
      // Keep x1 and shrink right
      Points_dst[1] = Points_src[1];
      Points_dst[2] = Points_src[1] + (Points_src[2] - Points_src[1]) * shrink_ratio;
    }
    if (area_id_L == 0 && area_id_R == 2)
    {
      // Keep x2 and shrink left
      Points_dst[1] = Points_src[2] - (Points_src[2] - Points_src[1]) * shrink_ratio;
      Points_dst[2] = Points_src[2];
    }
  }

  return 0;
}
msgs::BoxPoint DistanceEstimation::Get3dBBox(msgs::PointXYZ p0, msgs::PointXYZ p3, int class_id, camera::id cam_id)
{
  msgs::PointXYZ p1, p2, p4, p5, p6, p7;
  msgs::BoxPoint point8;
  int offset_y = 0;

  /// 3D bounding box
  ///   p5------p6
  ///   /|  2   /|
  /// p1-|----p2 |
  ///  |p4----|-p7
  ///  |/  1  | /
  /// p0-----P3

  /// |----|\
  /// |    | obstacle_l
  /// |----|/
  /// \    /
  /// obstacle_w

  float obstacle_h = 2, obstacle_l = 2 /*, obstacle_w = 2*/;
  if (class_id == 0)
  {
    obstacle_h = 1.8;
    obstacle_l = 0.33; /*obstacle_w = 0.6;*/
  }
  else if (class_id == 1 || class_id == 3)
  {
    obstacle_h = 1.8;
    obstacle_l = 2.5; /*obstacle_w = 0.6;*/
  }
  else if (class_id == 2)
  {
    obstacle_h = 1.5;
    obstacle_l = 5; /*obstacle_w = 2;*/
  }
  else if (class_id == 5 || class_id == 7)
  {
    obstacle_h = 2;
    obstacle_l = 7; /*obstacle_w = 2.5;*/
  }

  /// 1
  if (cam_id == camera::id::front_bottom_60 || cam_id == camera::id::front_top_close_120)
  {
    /// Camera Perspective   ///  Spec view
    ///   p5------p6         ///   p5------p6
    ///   /|  2   /|         ///   /|  2   /|
    /// p1-|----p2 |         /// p1-|----p2 |
    ///  |p4----|-p7    =    ///  |p4----|-p7
    ///  |/  1  | /          ///  |/  1  | /
    /// p0-----P3            /// p0-----P3

    p4 = p0;
    p4.x = p4.x + obstacle_l;
    p7 = p3;
    p7.x = p7.x + obstacle_l;
  }

  else if (cam_id == camera::id::back_top_120)
  {
    /// Camera Perspective   ///  Spec view
    ///   p2------p1         ///   p5------p6
    ///   /|  2   /|         ///   /|  2   /|
    /// p6-|----p5 |         /// p1-|----p2 |
    ///  |p3----|-p0   ->    ///  |p4----|-p7
    ///  |/  1  | /          ///  |/  1  | /
    /// p7-----P4            /// p0-----P3

    msgs::PointXYZ p0_cam, p3_cam, p4_cam, p7_cam;
    offset_y = Lidar_offset_y * (-1);

    p0_cam.x = p0.x * (-1);
    p3_cam.x = p3.x * (-1);
    p0_cam.y = p0.y * (-1) + offset_y;
    p3_cam.y = p3.y * (-1) + offset_y;
    p0_cam.z = p0.z;
    p3_cam.z = p3.z;

    p4_cam = p0_cam;
    p4_cam.x = p4_cam.x - obstacle_l;
    p7_cam = p3_cam;
    p7_cam.x = p7_cam.x - obstacle_l;

    p0 = p7_cam;
    p3 = p4_cam;
    p4 = p3_cam;
    p7 = p0_cam;
  }

  /// 2
  p1 = p0;
  p1.z = p1.z + obstacle_h;

  p2 = p3;
  p2.z = p2.z + obstacle_h;

  p5 = p4;
  p5.z = p5.z + obstacle_h;

  p6 = p7;
  p6.z = p6.z + obstacle_h;

  point8.p0 = p0;
  point8.p1 = p1;
  point8.p2 = p2;
  point8.p3 = p3;
  point8.p4 = p4;
  point8.p5 = p5;
  point8.p6 = p6;
  point8.p7 = p7;

  return point8;
}
msgs::BoxPoint DistanceEstimation::Get3dBBox(int x1, int y1, int x2, int y2, int class_id, camera::id cam_id)
{
  msgs::PointXYZ p0, p1, p2, p3, p4, p5, p6, p7;
  msgs::BoxPoint point8;
  int offset_y = 0;
  /// 3D cube
  ///   p5------p6
  ///   /|  2   /|
  /// p1-|----p2 |
  ///  |p4----|-p7
  ///  |/  1  | /
  /// p0-----P3

  /// birds view
  /// |----|\
  /// |    | obstacle_l
  /// |----|/
  /// \    /
  /// obstacle_w

  /// class id
  /// 0: person
  /// 1: bicycle
  /// 2: car
  /// 3: motorbike
  /// 5: bus
  /// 7: truck
  float obstacle_h = 2, obstacle_l = 2 /*, obstacle_w = 2*/;
  if (class_id == 0)
  {
    obstacle_h = 1.8;
    obstacle_l = 0.33; /*obstacle_w = 0.6;*/
  }
  else if (class_id == 1 || class_id == 3)
  {
    obstacle_h = 1.8;
    obstacle_l = 1.5; /*obstacle_w = 0.6;*/
  }                   // obstacle_l = 2.5
  else if (class_id == 2)
  {
    obstacle_h = 1.5;
    obstacle_l = 2; /*obstacle_w = 2;*/
  }                 // obstacle_l = 5
  else if (class_id == 5 || class_id == 7)
  {
    obstacle_h = 2;
    obstacle_l = 2.5; /*obstacle_w = 2.5;*/
  }                   // obstacle_l = 7

  if (cam_id == camera::id::front_bottom_60 || cam_id == camera::id::front_top_close_120 ||
      cam_id == camera::id::back_top_120)
  {
    std::vector<int> PointsSrc = { class_id, x1, x2, y2 };
    std::vector<int> PointsDst = { class_id, x1, x2, y2 };
    BoxShrink(cam_id, PointsSrc, PointsDst);
    x1 = PointsDst[1];
    x2 = PointsDst[2];
  }

  /// 1
  p0 = GetPointDist(x1, y2, cam_id);
  p3 = GetPointDist(x2, y2, cam_id);

  if (cam_id == camera::id::front_bottom_60 || cam_id == camera::id::front_top_close_120)
  {
    /// Camera Perspective   ///  Spec view
    ///   p5------p6         ///   p5------p6
    ///   /|  2   /|         ///   /|  2   /|
    /// p1-|----p2 |         /// p1-|----p2 |
    ///  |p4----|-p7    =    ///  |p4----|-p7
    ///  |/  1  | /          ///  |/  1  | /
    /// p0-----P3            /// p0-----P3

    p4 = p0;
    if (p4.x != 0)
    {
      p4.x = p4.x + obstacle_l;
    }
    p7 = p3;
    if (p7.x != 0)
    {
      p7.x = p7.x + obstacle_l;
    }
  }

  else if (cam_id == camera::id::back_top_120)
  {
    /// Camera Perspective   ///  Spec view
    ///   p2------p1         ///   p5------p6
    ///   /|  2   /|         ///   /|  2   /|
    /// p6-|----p5 |         /// p1-|----p2 |
    ///  |p3----|-p0   ->    ///  |p4----|-p7
    ///  |/  1  | /          ///  |/  1  | /
    /// p7-----P4            /// p0-----P3

    msgs::PointXYZ p0_cam, p3_cam, p4_cam, p7_cam;
    offset_y = Lidar_offset_y * (-1);

    p0_cam.x = p0.x * (-1);
    p3_cam.x = p3.x * (-1);
    p0_cam.y = p0.y * (-1) + offset_y;
    p3_cam.y = p3.y * (-1) + offset_y;
    p0_cam.z = p0.z;
    p3_cam.z = p3.z;

    p4_cam = p0_cam;
    p4_cam.x = p4_cam.x - obstacle_l;
    p7_cam = p3_cam;
    p7_cam.x = p7_cam.x - obstacle_l;

    p0 = p7_cam;
    p3 = p4_cam;
    p4 = p3_cam;
    p7 = p0_cam;
  }
  /// 2
  p1 = p0;
  p1.z = p1.z + obstacle_h;

  p2 = p3;
  p2.z = p2.z + obstacle_h;

  p5 = p4;
  p5.z = p5.z + obstacle_h;

  p6 = p7;
  p6.z = p6.z + obstacle_h;

  point8.p0 = p0;
  point8.p1 = p1;
  point8.p2 = p2;
  point8.p3 = p3;
  point8.p4 = p4;
  point8.p5 = p5;
  point8.p6 = p6;
  point8.p7 = p7;

  return point8;
}
msgs::PointXYZ DistanceEstimation::GetPointDist(int x, int y, camera::id cam_id)
{
  msgs::PointXYZ p0;
  float x_distMeter = 0, y_distMeter = 0;
  float offset_x = 0;
  int x_loc = y;
  int y_loc = x;
  // int img_h = 1208;

  DisEstiParams parmas_;

  offset_x = Lidar_offset_x;

  parmas_ = arr_params[cam_id];

  if (de_mode == 1)
  {
    if (cam_id == camera::id::front_bottom_60)
    {
      y_loc = (int)((float)y_loc / img_w * img_al_w);
      x_loc = (int)((float)x_loc / img_h * img_al_h);

      p0.x = align_FC60[x_loc][y_loc].x / 100;
      p0.y = align_FC60[x_loc][y_loc].y / 100;
      p0.z = align_FC60[x_loc][y_loc].z / 100;
      return p0;
    }
  }
  else
  {
    if (cam_id == camera::id::front_bottom_60 || cam_id == camera::id::front_top_close_120 || cam_id == camera::id::back_top_120)
    {
      if (!parmas_.regionDist_x.empty())
      {
        x_distMeter = ComputeObjectXDist(x_loc, parmas_.regionHeight_x, parmas_.regionDist_x);
      }
      if (!parmas_.regionDist_y.empty())
      {
        y_distMeter = ComputeObjectYDist(y_loc, x_loc, parmas_.regionHeight_y, parmas_.regionHeightSlope_y,
                                         parmas_.regionDist_y, img_h);
      }
    }
  }
  /*
  if (cam_id == camera::id::right_60 || cam_id == camera::id::left_60)
  {
    if (parmas_.regionDist_x.size() != 0)
    {
      x_distMeter = ComputeObjectYDist(y_loc, x_loc, parmas_.regionHeight_x, parmas_.regionHeightSlope_x,
                                       parmas_.regionDist_x, img_h);
    }
    if (parmas_.regionDist_y.size() != 0)
    {
      y_distMeter = ComputeObjectXDistWithSlope(y_loc, x_loc, parmas_.regionHeight_y, parmas_regionHeightSlope_y,
                                                parmas_.regionDist_y);
    }
  }
  */

  if (x_distMeter == 777)
  {
    p0.x = 0;
    p0.y = 0;
    p0.z = 0;
  }
  p0.x = x_distMeter + offset_x;
  p0.y = y_distMeter;
  p0.z = Lidar_offset_z;

  return p0;
}