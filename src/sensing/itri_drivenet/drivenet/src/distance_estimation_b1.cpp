#include "drivenet/distance_estimation_b1.h"
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

void DistanceEstimation::init(int car_id)
{
    carId = car_id;

    // camId: 0 (Front Right)
    regionHeight_60_FR_x = {1674, 1418, 1174, 895, 664, 470, 230, 20, -132, -340}; 
    regionHeightSlope_60_FR_x = {-1.661, -0.842, -0.603, -0.439, -0.349, -0.297, -0.253, -0.220, -0.203, -0.182}; 
    regionDist_60_FR_x = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    regionHeight_60_FR_y = {1197/*5*/,1171,1149,1121,1108,1098/*10*/,
                            1092,1091,1090,1089,1084/*15*/,
                            1083,1082,1080,1073,1073/*20*/,
                            1067,1072,1070,1066,1063/*25*/,
                            1071,1065,1078,1097,1083}; 
    regionHeightSlope_60_FR_y = {0.002/*5*/,-0.014,-0.034,-0.047,-0.056,-0.064/*10*/,
                                -0.072,-0.082,-0.091,-0.096,-0.098/*15*/,
                                -0.104,-0.107,-0.110,-0.110,-0.113/*20*/,
                                -0.113,-0.119,-0.120,-0.120,-0.121/*25*/,
                                -0.128,-0.127,-0.136,-0.148,-0.142}; 
    regionDist_60_FR_y = {-5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15, -16, -17, -18, -19, -20, -21, -22, -23, -24, -25, -26, -27, -28, -29, -30};

    // camId: 1 (Front Center)
    regionHeight_60_FC_x = { 1207, 1181, 1141, 1110,       1086 /*10*/, 1070, 1052, 1039, 1028, 1019, 1009,
                            1003, 996,  991,  985 /*20*/, 960,         946,  934,  926,  919,  914 /*50*/ };
    regionDist_60_FC_x = { 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 25, 30, 35, 40, 45, 50 };
    regionHeightSlope_60_FC_y = {0.12, 0.209, 0.27, 0.337, 0.44, 1.02, 3.87, -1.53, -0.66, -0.452, -0.333, -0.251, -0.121};
    regionHeight_60_FC_y = { -1817, -617, -252, 0, 242, 608, 913, 1220, 1510, 1746, 2016, 2346, 3801 };
    regionDist_60_FC_y = { 10, 5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5, -10 };

    LeftLinePoint1_60_FC = cv::Point(636, 914);
    LeftLinePoint2_60_FC = cv::Point(-1817, 1181);
    RightLinePoint1_60_FC = cv::Point(1371, 914);
    RightLinePoint2_60_FC = cv::Point(3801, 1181);

    // camId: 4 (Front Top)
    regionHeight_120_FT_x = { 1207, 1002, 740, 574, 460, 379, 320, 272, 231, 198, 171,
                                150,  130,  115, 99,  86,  75,  65,  57,  48,  40,  10 }; 
    regionDist_120_FT_x = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 25 };
    regionHeight_120_FT_y = {
        -1422, -1131, -824, -412, 70, 490, 942, 1292, 1732, 2258, 2641, 3030, 3471, 3619, 3709, 3548
    }; 
    regionHeightSlope_120_FT_y = { 0.603, 0.682,   0.784,   1.012,   1.56,    2.908,   48.28,   -4.4615,
                                    -1.8,  -1.0328, -0.7976, -0.6509, -0.5349, -0.5156, -0.5161, -0.5862 };
    regionDist_120_FT_y = { 6, 5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5, -6, -7, -8, -9 };

    LeftLinePoint1_120_FT = cv::Point(127, 272);
    LeftLinePoint2_120_FT = cv::Point(-1422, 1207);
    RightLinePoint1_120_FT = cv::Point(1904, 272);
    RightLinePoint2_120_FT = cv::Point(3548, 1207);

    // camId: 5 (Right Front)
    regionHeight_120_RF_x = { 1148, 830, 544, 377, 236, 157, 52 };      // 5 to 10(~1m), 20 to 50m (~5m) //Horizontal line
    regionDist_120_RF_x = { 0, 1, 2, 3, 4, 5, 6 };                      // 5 to 10, 20 to 50m (~5m)
    regionHeight_120_RF_y = { 2507, 1904, 1498, 1032, 637, 357, -80 };  //-2 to 0 to 2(~1m) //Vertical line
    regionHeightSlope_120_RF_y = { -1.2286, -2.4524, -6.000, 21.3529, 4.7308, 3.0297, 1.8171 };
    regionDist_120_RF_y = { -6, -5, -4, -3, -2, -1, -0 };  //-2 to 0 to 2 (~1m)

    // camId: 6 (Front Back)
    regionHeight_120_RB_x = { 1194, 838, 565, 395, 253, 138, 63, 17 };  // 5 to 10(~1m), 20 to 50m (~5m) //Horizontal line
    regionDist_120_RB_x = { 0, 1, 2, 3, 4, 5, 6, 7 };                   // 5 to 10, 20 to 50m (~5m)
    regionHeight_120_RB_y = { 2049, 1688, 1209, 714, 217, -114, -738 };  //-2 to 0 to 2(~1m) //Vertical line
    regionHeightSlope_120_RB_y = { -1.7722, -2.1614, -6.4409, 6.9259, 2.1378, 1.6333, 0.9539 };
    regionDist_120_RB_y = { -9, -8, -7, -6, -5, -4, -3 };  //-2 to 0 to 2 (~1m)

    // camId: 10 (Back Top)
    regionHeight_120_BT_x = { 1207, 836, 650, 532, 435, 367, 316, 270, 240, 210, 182, 161, 143 };            
    regionDist_120_BT_x = { 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20 };                             
    regionHeight_120_BT_y = { -1566, -1230, -861, -264, 40, 475, 875, 1370, 1704, 2195, 2439, 2808, 3152 };  
    regionHeightSlope_120_BT_y = { 0.536,  0.612,  0.7197, 1.063,   1.372,  2.624, 14.2,
                                    -2.951, -1.727, -1.167, -0.9098, -0.724, -0.608 };
    regionDist_120_BT_y = { 6, 5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5, -6 }; 

    LeftLinePoint1_120_BT = cv::Point(422, 143);
    LeftLinePoint2_120_BT = cv::Point(-1566, 1207);
    RightLinePoint1_120_BT = cv::Point(1400, 143);
    RightLinePoint2_120_BT = cv::Point(3152, 1207);

    Lidar_offset_x = 0;
    Lidar_offset_y = 0;
}

float DistanceEstimation::ComputeObjectXDist(int piexl_loc, std::vector<int> regionHeight,
                                             std::vector<float> regionDist)
{
  float distance = -1;
  float unitLength = 0.0;
  int bias = 0;
  float offset = 0.0;
  for (uint i = 1; i < regionHeight.size(); i++)
  {
    if ((piexl_loc >= regionHeight[i] && piexl_loc <= regionHeight[i - 1]))
    {
      int regionpixel = regionHeight[i - 1] - regionHeight[i];
      int regionMeter = regionDist[i] - regionDist[i - 1];
      unitLength = float(regionMeter) / float(regionpixel);
      bias = piexl_loc - regionHeight[i];
      offset = unitLength * float(bias);
      distance = regionDist[i] - offset;
      // printf("region[%d~%d][%d~%d], X- distance: %f\n ", i-1, i, regionHeight[i-1], regionHeight[i], distance);
      // printf("piexl_loc: %d,  regionDist: %d, unit: %f, bias: %d, offset: %f\n", piexl_loc, regionDist[i],
      // unitLength, bias, offset);
    }
    else if ((piexl_loc <= regionHeight[i] && piexl_loc >= regionHeight[i - 1]))
    {
      int regionpixel = regionHeight[i] - regionHeight[i - 1];
      int regionMeter = regionDist[i] - regionDist[i - 1];
      unitLength = float(regionMeter) / float(regionpixel);
      bias = regionHeight[i] - piexl_loc;
      offset = unitLength * float(bias);
      distance = regionDist[i] - offset;
      // printf("region[%d~%d][%d~%d], X- distance: %f\n ", i-1, i, regionHeight[i-1], regionHeight[i], distance);
      // printf("piexl_loc: %d,  regionDist: %d, unit: %f, bias: %d, offset: %f\n", piexl_loc, regionDist[i],
      // unitLength, bias, offset);
    }
    else
    {
      if (piexl_loc > regionHeight.front())
        distance = regionHeight.front() - 0.2;
      else if (piexl_loc < regionHeight.back())
        distance = regionHeight.back() + 0.2;
    }
  }
  // printf("piexl_loc: %d,  regionDist: %d, unit: %f, bias: %d, offset: %f\n", piexl_loc, regionDist[i], unitLength,
  // bias, offset);
  int multiplier = pow(10, 2);
  distance = int(distance * multiplier) / (multiplier * 1.0);

  return distance;
}
float DistanceEstimation::ComputeObjectXDistWithSlope(int pixel_loc_x, int pixel_loc_y, std::vector<int> regionHeight,
                                                      std::vector<float> regionHeightSlope_x,
                                                      std::vector<float> regionDist, int img_w)
{
  float distance = -1;
  float unitLength = 0.0;
  int bias = 0;
  float offset = 0.0;

  std::vector<int> regionHeight_new = regionHeight;

  // std::cout << "pixel_loc_x: " << pixel_loc_x <<  ", pixel_loc_y: " << pixel_loc_y << std::endl;
  for (uint i = 0; i < regionHeight.size(); i++)
  {
    // int y = img_h - pixel_loc_x;
    if (regionHeightSlope_x[i] != 0)
      regionHeight_new[i] = regionHeight[i] - int((regionHeightSlope_x[i]) * pixel_loc_x);
    else
      regionHeight_new[i] = regionHeight[i];
    // printf("region[%d], region:%d, new region:%d, \n ", i, regionHeight[i], regionHeight_new[i]);
  }

  for (uint i = 1; i < regionHeight_new.size(); i++)
  {
    if (pixel_loc_y >= regionHeight_new[i] && pixel_loc_y <= regionHeight_new[i - 1])
    {
      int regionpixel = regionHeight_new[i - 1] - regionHeight_new[i];
      int regionMeter = regionDist[i] - regionDist[i - 1];
      if (regionpixel != 0)
        unitLength = float(regionMeter) / float(regionpixel);
      bias = pixel_loc_y - regionHeight_new[i];
      offset = unitLength * float(bias);
      distance = regionDist[i] - offset;
      // printf("region[%d~%d][%d~%d], new region[%d~%d], Y- distance: %f\n ", i-1, i, regionHeight[i-1],
      // regionHeight[i], regionHeight_new[i-1], regionHeight_new[i], distance);
      // printf("piexl_loc: %d,  regionDist: %d, unit: %f, bias: %d, offset: %f\n", pixel_loc_y, regionDist[i],
      // unitLength, bias, offset);
    }
    else if (pixel_loc_y <= regionHeight_new[i] && pixel_loc_y >= regionHeight_new[i - 1])
    {
      int regionpixel = regionHeight_new[i] - regionHeight_new[i - 1];
      int regionMeter = regionDist[i] - regionDist[i - 1];
      if (regionpixel != 0)
        unitLength = float(regionMeter) / float(regionpixel);
      bias = regionHeight_new[i] - pixel_loc_y;
      offset = unitLength * float(bias);
      distance = regionDist[i] - offset;
      // printf("region[%d~%d][%d~%d], new region[%d~%d], Y- distance: %f\n ", i-1, i, regionHeight[i-1],
      // regionHeight[i], regionHeight_new[i-1], regionHeight_new[i], distance);
      // printf("piexl_loc: %d,  regionDist: %d, unit: %f, bias: %d, offset: %f\n", pixel_loc_y, regionDist[i],
      // unitLength, bias, offset);
    }
    else
    {
      if (pixel_loc_y < regionHeight_new[0])
        distance = 8;
      else if (pixel_loc_y > regionHeight_new[regionHeight_new.size() - 1])
        distance = -8;
    }
  }

  int multiplier = pow(10, 2);
  distance = int(distance * multiplier) / (multiplier * 1.0);

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

  // std::cout << "piexl_loc_x: " << piexl_loc_x <<  ", piexl_loc_y: " << piexl_loc_y << std::endl;
  for (uint i = 0; i < regionHeight.size(); i++)
  {
    int y = img_h - piexl_loc_x;
    if (regionHeightSlope_y[i] != 0)
      regionHeight_new[i] = regionHeight[i] + int((1 / regionHeightSlope_y[i]) * y);
    else
      regionHeight_new[i] = regionHeight[i];
    // printf("region[%d], region:%d, new region:%d, \n ", i, regionHeight[i], regionHeight_new[i]);
  }

  for (uint i = 1; i < regionHeight_new.size(); i++)
  {
    if (piexl_loc_y >= regionHeight_new[i] && piexl_loc_y <= regionHeight_new[i - 1])
    {
      int regionpixel = regionHeight_new[i - 1] - regionHeight_new[i];
      int regionMeter = regionDist[i] - regionDist[i - 1];
      if (regionpixel != 0)
        unitLength = float(regionMeter) / float(regionpixel);
      bias = piexl_loc_y - regionHeight_new[i];
      offset = unitLength * float(bias);
      distance = regionDist[i] - offset;
      // printf("region[%d~%d][%d~%d], new region[%d~%d], Y- distance: %f\n ", i-1, i, regionHeight[i-1],
      // regionHeight[i], regionHeight_new[i-1], regionHeight_new[i], distance);
      // printf("piexl_loc: %d,  regionDist: %d, unit: %f, bias: %d, offset: %f\n", piexl_loc_y, regionDist[i],
      // unitLength, bias, offset);
    }
    else if (piexl_loc_y <= regionHeight_new[i] && piexl_loc_y >= regionHeight_new[i - 1])
    {
      int regionpixel = regionHeight_new[i] - regionHeight_new[i - 1];
      int regionMeter = regionDist[i] - regionDist[i - 1];
      if (regionpixel != 0)
        unitLength = float(regionMeter) / float(regionpixel);
      bias = regionHeight_new[i] - piexl_loc_y;
      offset = unitLength * float(bias);
      distance = regionDist[i] - offset;
      // printf("region[%d~%d][%d~%d], new region[%d~%d], Y- distance: %f\n ", i-1, i, regionHeight[i-1],
      // regionHeight[i], regionHeight_new[i-1], regionHeight_new[i], distance);
      // printf("piexl_loc: %d,  regionDist: %d, unit: %f, bias: %d, offset: %f\n", piexl_loc_y, regionDist[i],
      // unitLength, bias, offset);
    }
    else
    {
      if (piexl_loc_y < regionHeight_new.front())
        distance = regionDist.front() + 0.5;
      else if (piexl_loc_y > regionHeight_new.back())
        distance = regionDist.back() - 0.5;
    }
  }

  int multiplier = pow(10, 2);
  distance = int(distance * multiplier) / (multiplier * 1.0);

  return distance;
}

int CheckPointInArea(cv::Point RightLinePoint1, cv::Point RightLinePoint2, cv::Point LeftLinePoint1,
                     cv::Point LeftLinePoint2, int object_x1, int object_y2)
{
    int point0 = 0;
    int point1 = 1;
    int point2 = 2;
    /// right
    int C1 = (RightLinePoint1.x - RightLinePoint2.x) * (object_y2 - RightLinePoint2.y) -
            (object_x1 - RightLinePoint2.x) * (RightLinePoint1.y - RightLinePoint2.y);
    /// left
    int C2 = (LeftLinePoint1.x - LeftLinePoint2.x) * (object_y2 - LeftLinePoint2.y) -
            (object_x1 - LeftLinePoint2.x) * (LeftLinePoint1.y - LeftLinePoint2.y);

    if (C1 > 0)
        return point2;
    else if (C2 < 0)
        return point1;
    else
        return point0;
}
int box_shrink(int cam_id, std::vector<int> Points_src, std::vector<int>& Points_dst)
{
    // PointsSrc = {class_id, x1, x2, y2};
    // PointsDst = {class_id, x1, x2, y2};

    // int edge_left, edge_right;
    int area_id_R = 1;  // 1: left, 2:right
    int area_id_L = 1;  // 1: left, 2:right

    double shrink_ratio;

    // Four points of RoI which need to shrink (usually road lane)
    cv::Point LeftLinePoint1;
    cv::Point LeftLinePoint2;
    cv::Point RightLinePoint1;
    cv::Point RightLinePoint2;

    // int box_center_x = (Points_src[1] + Points_src[2]) / 2; // get x center of objects

    if (cam_id == 1)
    {
        // From x 6 - 50 m, y -3 to +3 m.
        LeftLinePoint1 = cv::Point(869, 914);
        LeftLinePoint2 = cv::Point(0, 1207);
        RightLinePoint1 = cv::Point(1097, 914);
        RightLinePoint2 = cv::Point(1746, 1207);
        area_id_L = CheckPointInArea(RightLinePoint1, RightLinePoint2, LeftLinePoint1, LeftLinePoint2, Points_src[1],
                                    Points_src[3]);
        area_id_R = CheckPointInArea(RightLinePoint1, RightLinePoint2, LeftLinePoint1, LeftLinePoint2, Points_src[2],
                                    Points_src[3]);
        
        switch (Points_src[0])
        {
            case 0:
            {  // 0:person
            shrink_ratio = 1;
            break;
            }
            // 1:bicycle, 3:motorbike
            case 1:
            {
            shrink_ratio = 0.9;
            break;
            }
            case 3:
            {
            shrink_ratio = 0.9;
            break;
            }
            // 2:car
            case 2:
            {
            shrink_ratio = 0.7;
            break;
            }
            // 5:bus, 7:truck
            case 5:
            {
            shrink_ratio = 0.7;
            break;
            }
            case 7:
            {
            shrink_ratio = 0.7;
            break;
            }
            default:
            {
            shrink_ratio = 1;
            break;
            }
        }
    }else if(cam_id == 4)
    {
        // From x 0 - 7 m, y -2 to +2 m.
        LeftLinePoint1 = cv::Point(510, 272);
        LeftLinePoint2 = cv::Point(-412, 1207);
        RightLinePoint1 = cv::Point(1351, 272);
        RightLinePoint2 = cv::Point(2258, 1207);
        area_id_L = CheckPointInArea(RightLinePoint1, RightLinePoint2, LeftLinePoint1, LeftLinePoint2, Points_src[1],
                                    Points_src[3]);
        area_id_R = CheckPointInArea(RightLinePoint1, RightLinePoint2, LeftLinePoint1, LeftLinePoint2, Points_src[2],
                                    Points_src[3]);
        
        switch (Points_src[0])
        {
            case 0:
            {  // 0:person
            shrink_ratio = 1;
            break;
            }
            // 1:bicycle, 3:motorbike
            case 1:
            {
            shrink_ratio = 0.9;
            break;
            }
            case 3:
            {
            shrink_ratio = 0.9;
            break;
            }
            // 2:car
            case 2:
            {
            shrink_ratio = 0.7;
            break;
            }
            // 5:bus, 7:truck
            case 5:
            {
            shrink_ratio = 0.5;
            break;
            }
            case 7:
            {
            shrink_ratio = 0.5;
            break;
            }
            default:
            {
            shrink_ratio = 1;
            break;
            }
        }
    }else if(cam_id == 10)
    {
        // From x -8 to -20 m, y -3 to +3 m.
        LeftLinePoint1 = cv::Point(737, 143);
        LeftLinePoint2 = cv::Point(-264, 1207);
        RightLinePoint1 = cv::Point(1182, 143);
        RightLinePoint2 = cv::Point(2195, 1207);

        area_id_L = CheckPointInArea(RightLinePoint1, RightLinePoint2, LeftLinePoint1, LeftLinePoint2, Points_src[1],
                                    Points_src[3]);
        area_id_R = CheckPointInArea(RightLinePoint1, RightLinePoint2, LeftLinePoint1, LeftLinePoint2, Points_src[2],
                                    Points_src[3]);
        
        switch (Points_src[0])
        {
            case 0:
            {  // 0:person
            shrink_ratio = 1;
            break;
            }
            // 1:bicycle, 3:motorbike
            case 1:
            {
            shrink_ratio = 0.9;
            break;
            }
            case 3:
            {
            shrink_ratio = 0.9;
            break;
            }
            // 2:car
            case 2:
            {
            shrink_ratio = 0.7;
            break;
            }
            // 5:bus, 7:truck
            case 5:
            {
            shrink_ratio = 0.5;
            break;
            }
            case 7:
            {
            shrink_ratio = 0.5;
            break;
            }
            default:
            {
            shrink_ratio = 1;
            break;
            }
        }
    }

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
msgs::BoxPoint DistanceEstimation::Get3dBBox(msgs::PointXYZ p0, msgs::PointXYZ p3, int class_id, int cam_id)
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

  /// |----| \ 
    /// |    |  obstacle_l
  /// |----| /
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
  if (cam_id == 0 || cam_id == 1 || cam_id == 2 || cam_id == 4)
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
  // else if (cam_id == 4)
  // {
  //     /// Camera Perspective   ///  Spec view
  //     ///   p6------p2         ///   p5------p6
  //     ///   /|  2   /|         ///   /|  2   /|
  //     /// p5-|----p7 |         /// p1-|----p2 |
  //     ///  |p7----|-p3   ->    ///  |p4----|-p7
  //     ///  |/  1  | /          ///  |/  1  | /
  //     /// p4-----P0            /// p0-----P3

  //     msgs::PointXYZ p0_cam, p3_cam, p4_cam, p7_cam;
  //     p0_cam = p0;
  //     p3_cam = p3;
  //     p4_cam = p0_cam;
  //     p4_cam.y = p4_cam.y - obstacle_w;
  //     p7_cam = p3_cam;
  //     p7_cam.y = p7_cam.y - obstacle_w;

  //     p0 = p4_cam;
  //     p3 = p0_cam;
  //     p4 = p7_cam;
  //     p7 = p3_cam;
  // }
  // else if (cam_id == 6)
  // {
  //     /// Camera Perspective   ///  Spec view
  //     ///   p1------p5         ///   p5------p6
  //     ///   /|  2   /|         ///   /|  2   /|
  //     /// p2-|----p6 |         /// p1-|----p2 |
  //     ///  |p0----|-p4   ->    ///  |p4----|-p7
  //     ///  |/  1  | /          ///  |/  1  | /
  //     /// p3-----P7            /// p0-----P3

  //     msgs::PointXYZ p0_cam, p3_cam, p4_cam, p7_cam;
  //     p0_cam = p0;
  //     p3_cam = p3;
  //     p4_cam = p0_cam;
  //     p4_cam.y = p4_cam.y + obstacle_w;
  //     p7_cam = p3_cam;
  //     p7_cam.y = p7_cam.y + obstacle_w;

  //     p0 = p3_cam;
  //     p3 = p7_cam;
  //     p4 = p0_cam;
  //     p7 = p4_cam;
  // }
  else if (cam_id == 10)
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
msgs::BoxPoint DistanceEstimation::Get3dBBox(int x1, int y1, int x2, int y2, int class_id, int cam_id)
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
  /// |----| \ 
    /// |    |  obstacle_l
  /// |----| /
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

  if(cam_id == 1 || cam_id == 4 || cam_id == 10 /*|| cam_id == 8 || cam_id == 9*/)
  {
      std::vector<int> PointsSrc = {class_id, x1, x2, y2};
      std::vector<int> PointsDst = {class_id, x1, x2, y2};
      box_shrink(cam_id, PointsSrc, PointsDst);
      x1 = PointsDst[1]; x2 = PointsDst[2];
  }

  /// 1
  p0 = GetPointDist(x1, y2, cam_id);
  p3 = GetPointDist(x2, y2, cam_id);

  /// 1
  if (cam_id == 0 || cam_id == 1 || cam_id == 2 || cam_id == 4)
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
  // else if (cam_id == 4)
  // {
  //     /// Camera Perspective   ///  Spec view
  //     ///   p6------p2         ///   p5------p6
  //     ///   /|  2   /|         ///   /|  2   /|
  //     /// p5-|----p7 |         /// p1-|----p2 |
  //     ///  |p7----|-p3   ->    ///  |p4----|-p7
  //     ///  |/  1  | /          ///  |/  1  | /
  //     /// p4-----P0            /// p0-----P3

  //     msgs::PointXYZ p0_cam, p3_cam, p4_cam, p7_cam;
  //     offset_y = Lidar_offset_y;

  //     p0_cam.x = p0.x *(-1);
  //     p3_cam.x = p3.x *(-1);
  //     p0_cam.z = p0.y + offset_y;
  //     p3_cam.z = p3.y + offset_y;
  //     p0_cam.z = p0.z;
  //     p3_cam.z = p3.z;
  //     p4_cam = p0_cam;
  //     p4_cam.y = p4_cam.y - obstacle_w;
  //     p7_cam = p3_cam;
  //     p7_cam.y = p7_cam.y - obstacle_w;

  //     p0 = p4_cam;
  //     p3 = p0_cam;
  //     p4 = p7_cam;
  //     p7 = p3_cam;
  // }
  // else if (cam_id == 6)
  // {
  //     /// Camera Perspective   ///  Spec view
  //     ///   p1------p5         ///   p5------p6
  //     ///   /|  2   /|         ///   /|  2   /|
  //     /// p2-|----p6 |         /// p1-|----p2 |
  //     ///  |p0----|-p4   ->    ///  |p4----|-p7
  //     ///  |/  1  | /          ///  |/  1  | /
  //     /// p3-----P7            /// p0-----P3

  //     msgs::PointXYZ p0_cam, p3_cam, p4_cam, p7_cam;
  //     offset_y = Lidar_offset_y*(-1);

  //     p0_cam.x = p0.x *(-1);
  //     p3_cam.x = p3.x *(-1);
  //     p0_cam.z = p0.y + offset_y;
  //     p3_cam.z = p3.y + offset_y;
  //     p0_cam.z = p0.z;
  //     p3_cam.z = p3.z;
  //     p4_cam = p0_cam;
  //     p4_cam.y = p4_cam.y + obstacle_w;
  //     p7_cam = p3_cam;
  //     p7_cam.y = p7_cam.y + obstacle_w;

  //     p0 = p3_cam;
  //     p3 = p7_cam;
  //     p4 = p0_cam;
  //     p7 = p4_cam;
  // }
  // else if (cam_id == 7)
  // {
  //     /// Camera Perspective   ///  Spec view
  //     ///   p2------p1         ///   p5------p6
  //     ///   /|  2   /|         ///   /|  2   /|
  //     /// p6-|----p5 |         /// p1-|----p2 |
  //     ///  |p3----|-p0   ->    ///  |p4----|-p7
  //     ///  |/  1  | /          ///  |/  1  | /
  //     /// p7-----P4            /// p0-----P3

  //     msgs::PointXYZ p0_cam, p3_cam, p4_cam, p7_cam;

  //     offset_y = Lidar_offset_y;

  //     p0_cam.x = p0.x *(-1);
  //     p3_cam.x = p3.x *(-1);
  //     p0_cam.y = p0.y + offset_y;
  //     p3_cam.y = p3.y + offset_y;
  //     p0_cam.z = p0.z;
  //     p3_cam.z = p3.z;

  //     p4_cam = p0_cam;
  //     p4_cam.x = p4_cam.x - obstacle_l;
  //     p7_cam = p3_cam;
  //     p7_cam.x = p7_cam.x - obstacle_l;

  //     p0 = p7_cam;
  //     p3 = p4_cam;
  //     p4 = p3_cam;
  //     p7 = p0_cam;
  // }
  else if (cam_id == 10)
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

  // std::cout << "3D: p0(x, y, z) = (" << point8.p0.x  << ", " <<  point8.p0.y <<  ", " << point8.p0.z << ")." <<
  // std::endl;
  // std::cout << "3D: p1(x, y, z) = (" << point8.p1.x  << ", " <<  point8.p1.y <<  ", " << point8.p1.z << ")." <<
  // std::endl;
  // std::cout << "3D: p2(x, y, z) = (" << point8.p2.x  << ", " <<  point8.p2.y <<  ", " << point8.p2.z << ")." <<
  // std::endl;
  // std::cout << "3D: p3(x, y, z) = (" << point8.p3.x  << ", " <<  point8.p3.y <<  ", " << point8.p3.z << ")." <<
  // std::endl;
  // std::cout << "3D: p4(x, y, z) = (" << point8.p4.x  << ", " <<  point8.p4.y <<  ", " << point8.p4.z << ")." <<
  // std::endl;
  // std::cout << "3D: p5(x, y, z) = (" << point8.p5.x  << ", " <<  point8.p5.y <<  ", " << point8.p5.z << ")." <<
  // std::endl;
  // std::cout << "3D: p6(x, y, z) = (" << point8.p6.x  << ", " <<  point8.p6.y <<  ", " << point8.p6.z << ")." <<
  // std::endl;
  // std::cout << "3D: p7(x, y, z) = (" << point8.p7.x  << ", " <<  point8.p7.y <<  ", " << point8.p7.z << ")." <<
  // std::endl;

  return point8;
}
msgs::PointXYZ DistanceEstimation::GetPointDist(int x, int y, int cam_id)
{
  std::vector<int> regionHeight_x;
  std::vector<float> regionDist_x;
  std::vector<int> regionHeight_y;
  std::vector<float> regionDist_y;
  std::vector<float> regionHeightSlope_y;
  std::vector<float> regionHeightSlope_x;

  msgs::PointXYZ p0;
  float x_distMeter = 0, y_distMeter = 0;
  float offset_x = 0;
  int x_loc = y;
  int y_loc = x;
  int img_h = 1208;
  // int img_w = 1920;
  // int mode = 1;

  if (cam_id == 1)
  {
    regionHeight_x = regionHeight_60_FC_x;
    regionDist_x = regionDist_60_FC_x;
    regionHeight_y = regionHeight_60_FC_y;
    // regionHeightSlope_x = regionHeightSlope_60_FC_x;
    regionHeightSlope_y = regionHeightSlope_60_FC_y;
    regionDist_y = regionDist_60_FC_y;
    offset_x = Lidar_offset_x;
  }
  else if (cam_id == 0)
  {
    regionHeight_x = regionHeight_60_FR_x;
    regionDist_x = regionDist_60_FR_x;
    regionHeight_y = regionHeight_60_FR_y;
    regionHeightSlope_x = regionHeightSlope_60_FR_x;    
    regionHeightSlope_y = regionHeightSlope_60_FR_y;
    regionDist_y = regionDist_60_FR_y;
    offset_x = Lidar_offset_x;
  }
//   else if (cam_id == 2)
//   {
//     regionHeight_x = regionHeight_60_FL_x;
//     regionDist_x = regionDist_60_FL_x;
//     regionHeight_y = regionHeight_60_FL_y;
//     regionHeightSlope_x = regionHeightSlope_60_FL_x;        
//     regionHeightSlope_y = regionHeightSlope_60_FL_y;
//     regionDist_y = regionDist_60_FL_y;
//     offset_x = Lidar_offset_x;
//   }
  else if (cam_id == 4)
  {
    regionHeight_x = regionHeight_120_FT_x;
    regionDist_x = regionDist_120_FT_x;
    regionHeight_y = regionHeight_120_FT_y;
    regionHeightSlope_y = regionHeightSlope_120_FT_y;
    regionDist_y = regionDist_120_FT_y;
    offset_x = Lidar_offset_x;
  }
  else if (cam_id == 5)
  {
    regionHeight_x = regionHeight_120_RF_x;
    regionDist_x = regionDist_120_RF_x;
    regionHeight_y = regionHeight_120_RF_y;
    regionHeightSlope_y = regionHeightSlope_120_RF_y;
    regionDist_y = regionDist_120_RF_y;
    offset_x = Lidar_offset_x;
  }
  else if (cam_id == 6)
  {
    regionHeight_x = regionHeight_120_RB_x;
    regionDist_x = regionDist_120_RB_x;
    regionHeight_y = regionHeight_120_RB_y;
    regionHeightSlope_y = regionHeightSlope_120_RB_y;
    regionDist_y = regionDist_120_RB_y;
    offset_x = Lidar_offset_x;
  }
  else if (cam_id == 10)
  {
    regionHeight_x = regionHeight_120_BT_x;
    regionDist_x = regionDist_120_BT_x;
    regionHeight_y = regionHeight_120_BT_y;
    regionHeightSlope_y = regionHeightSlope_120_BT_y;
    regionDist_y = regionDist_120_BT_y;
    offset_x = Lidar_offset_x;
  }
  else
  {
    p0.x = 0;
    p0.y = 0;
    p0.z = 0;
    return p0;
  }

    if(cam_id == 1 || cam_id == 4 || cam_id == 10)
    {
        if (regionDist_x.size() != 0)
            x_distMeter = ComputeObjectXDist(x_loc, regionHeight_x, regionDist_x);
        if (regionDist_y.size() != 0)
            y_distMeter = ComputeObjectYDist(y_loc, x_loc, regionHeight_y, regionHeightSlope_y, regionDist_y, img_h);

        p0.x = x_distMeter + offset_x;
        p0.y = y_distMeter;
        p0.z = Lidar_offset_z;
    }

    // if(cam_id == 0 || cam_id == 2)
    // {
    //     if (regionDist_y.size() != 0)
    //         y_distMeter = ComputeObjectXDist(x_loc, regionHeight_x, regionDist_x);
    //     if (regionDist_x.size() != 0)
    //         x_distMeter = ComputeObjectYDist(y_loc, x_loc, regionHeight_x, regionHeightSlope_x, regionDist_x, img_h);

    //     p0.x = x_distMeter + offset_x;
    //     p0.y = y_distMeter;
    //     p0.z = Lidar_offset_z;
    // }


  return p0;
}
