/*
   CREATER: ICL U300
   DATE: Aug, 2019
*/
#include "DistanceEstimation.h"

DistanceEstimation::DistanceEstimation()
{
  // regionHeight_60_FC_x = {1207, 1119, 1045, 987, 940, 908, 873, 846, 822, 802, 780, 763, 748, 738, 728, 682, 650,
  // 630, 615, 600, 590, 400};
  // regionDist_60_FC_x = {6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 25, 30, 35, 40, 45, 50, 100};
  // regionHeight_60_FC_y = {-1467, -171, 122, 385, 676, 968, 1248, 1553, 1786, 2018, 3008};
  // regionHeightSlope_60_FC_y = {0.316, 0.631, 0.846, 1.22, 2.44, 86.3, -2.66, -1.23, -0.91, -0.70, -0.36};
  // regionDist_60_FC_y = {8, 4, 3, 2, 1, 0, -1, -2, -3, -4, -8};

  regionHeight_60_FC_x = { 1207, 1102, 1033, 975, 928, 890, 857, 827, 803, 783, 765,
                           748,  733,  720,  708, 661, 627, 607, 589, 574, 563, 0 };
  regionDist_60_FC_x = { 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 25, 30, 35, 40, 45, 50, 100 };
  regionHeight_60_FC_y = { -1284, 135, 463, 741, 990, 1262, 1543, 1806, 2930 };
  regionHeightSlope_60_FC_y = { 0.32, 0.869, 1.45, 3.36, 0, -2.45, -1.3, -0.9, -0.36 };
  regionDist_60_FC_y = { 8, 3, 2, 1, 0, -1, -2, -3, -8 };

  regionHeight_30_FC_x = { 1207, 1180, 1126, 1080, 1042, 1004, 972, 942, 916, 816, 748, 698, 662, 634, 608, 0 };
  regionDist_30_FC_x = { 13, 14, 15, 16, 17, 18, 19, 20, 25, 30, 35, 40, 45, 50, 100 };
  regionHeight_30_FC_y = { 135, 463, 741, 990, 1262, 1543, 1806 };
  regionDist_30_FC_y = { 3, 2, 1, 0, -1, -2, -3 };

  regionHeight_120_FC_x = { 1207, 1074, 826, 655, 533, 441, 365, 305, 255, 216, 179 };
  regionDist_120_FC_x = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
  regionHeight_120_FC_y = { 0, 131, 504, 950, 1294, 1646, 1919 };
  regionDist_120_FC_y = { 3, 2, 1, 0, -1, -2, -3 };

  regionHeight_30_R_x = { 1903, 1680, 1530, 1378, 1225, 1112, 1035, 966, 895, 839,
                          800,  758,  707,  661,  531,  450,  387,  353, 320, 297 };
  regionDist_30_R_x = { 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 25, 30, 35, 40, 45, 50 };
  regionHeight_30_R_y = { 873, 525, -170, -1039, -2067 };
  regionHeightSlope_30_R_y = { 4.51, 2.6, 1.29, 0.791, 0.543 };
  regionDist_30_R_y = { 1, 2, 3, 4, 5 };

  Lidar_offset_x = 1;
}

float DistanceEstimation::ComputeObjectXDist(int piexl_loc, std::vector<int> regionHeight,
                                             std::vector<float> regionDist)
{
  float distance = -1;
  float unitLength = 0.0;
  int bias = 0;
  float offset = 0.0;
  for (int i = 1; i < regionHeight.size(); i++)
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
      if (piexl_loc > regionHeight[0])
        distance = 6;
    }
  }
  // printf("piexl_loc: %d,  regionDist: %d, unit: %f, bias: %d, offset: %f\n", piexl_loc, regionDist[i], unitLength,
  // bias, offset);
  int multiplier = pow(10, 2);
  distance = int(distance * multiplier) / (multiplier * 1.0);

  return distance;
}
float DistanceEstimation::ComputeObjectYDist(int piexl_loc_y, int piexl_loc_x, std::vector<int> regionHeight,
                                             std::vector<float> regionHeightSlope_y, std::vector<float> regionDist,
                                             int mode)
{
  float distance = -1;
  float unitLength = 0.0;
  int bias = 0;
  float offset = 0.0;
  int img_h = 1208;

  if (mode == 1)
  {
    for (int i = 1; i < regionHeight.size(); i++)
    {
      if (piexl_loc_y >= regionHeight[i] && piexl_loc_y <= regionHeight[i - 1])
      {
        int regionpixel = regionHeight[i - 1] - regionHeight[i];
        int regionMeter = regionDist[i] - regionDist[i - 1];
        unitLength = float(regionMeter) / float(regionpixel);
        bias = piexl_loc_y - regionHeight[i];
        offset = unitLength * float(bias);
        distance = regionDist[i] - offset;
      }
      else if (piexl_loc_y <= regionHeight[i] && piexl_loc_y >= regionHeight[i - 1])
      {
        int regionpixel = regionHeight[i] - regionHeight[i - 1];
        int regionMeter = regionDist[i] - regionDist[i - 1];
        unitLength = float(regionMeter) / float(regionpixel);
        bias = regionHeight[i] - piexl_loc_y;
        offset = unitLength * float(bias);
        distance = regionDist[i] - offset;
      }
    }
  }
  else if (mode == 2)
  {
    std::vector<int> regionHeight_new = regionHeight;

    for (int i = 1; i < regionHeight.size(); i++)
    {
      int y = img_h - piexl_loc_x;
      if (regionHeightSlope_y[i] != 0)
        regionHeight_new[i] = regionHeight[i] + int((1 / regionHeightSlope_y[i]) * y);
      // std::cout <<  "piexl_loc_x: " << piexl_loc_x << ", y: " << y << "pixel location offset: " <<
      // int((1/regionHeightSlope_y[i])*y) << std::endl;
      // std::cout <<  "regionHeight_new" << i << ": " << regionHeight_new[i] << std::endl;
    }

    for (int i = 1; i < regionHeight_new.size(); i++)
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
        if (piexl_loc_y < regionHeight_new[0])
          distance = 8;
        else if (piexl_loc_y > regionHeight_new[regionHeight_new.size() - 1])
          distance = -8;
      }
    }
  }

  int multiplier = pow(10, 2);
  distance = int(distance * multiplier) / (multiplier * 1.0);

  return distance;
}

msgs::PointXYZ DistanceEstimation::GetPointDist(int x, int y, int cam_id)
{
  std::vector<int> regionHeight_x;
  std::vector<float> regionDist_x;
  std::vector<int> regionHeight_y;
  std::vector<float> regionDist_y;
  std::vector<float> regionHeightSlope_y;

  msgs::PointXYZ p0;
  float x_distMeter = 0, y_distMeter = 0;
  int x_loc = y;
  int y_loc = x;

  int mode = 1;

  if (cam_id == 2)
  {
    regionHeight_x = regionHeight_60_FC_x;
    regionDist_x = regionDist_60_FC_x;
    regionHeight_y = regionHeight_60_FC_y;
    regionHeightSlope_y = regionHeightSlope_60_FC_y;
    regionDist_y = regionDist_60_FC_y;

    offset = Lidar_offset_x;
    mode = 2;
  }
  else if (cam_id == 5)
  {
    regionHeight_x = regionHeight_120_FC_x;
    regionDist_x = regionDist_120_FC_x;
    regionHeight_y = regionHeight_120_FC_y;
    regionDist_y = regionDist_120_FC_y;
    offset = Lidar_offset_x;
  }
  else if (cam_id == 8)
  {
    regionHeight_x = regionHeight_30_FC_x;
    regionDist_x = regionDist_30_FC_x;
    regionHeight_y = regionHeight_30_FC_y;
    regionDist_y = regionDist_30_FC_y;
    offset = Lidar_offset_x;
  }
  else
  {
    p0.x = 0;
    p0.y = 0;
    p0.z = 0;
    return p0;
  }

  if (regionDist_x.size() != 0)
    x_distMeter = ComputeObjectXDist(x_loc, regionHeight_x, regionDist_x);
  if (regionDist_y.size() != 0)
    y_distMeter = ComputeObjectYDist(y_loc, x_loc, regionHeight_y, regionHeightSlope_y, regionDist_y, mode);

  p0.x = x_distMeter + Lidar_offset_x;
  p0.y = y_distMeter;
  p0.z = Lidar_offset_z;

  return p0;
}
