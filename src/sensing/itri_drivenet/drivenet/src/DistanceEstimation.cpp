#include "drivenet/DistanceEstimation.h"
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

void DistanceEstimation::init (int truck_id)
{     
    truckId = truck_id;
    if(truckId == 1)
    {
        regionHeight_60_FC_x = {1207, 1151, 1068, 1006, 947, 908, 873, 846, 822, 802, 780, 763, 748, 738, 728, 682, 650, 630, 615, 600, 590, 0};
        regionDist_60_FC_x = {6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 25, 30, 35, 40, 45, 50, 100}; 
        regionHeight_60_FC_y = {-1296, -169, 122, 385, 676, 968, 1248, 1553, 1786, 2014, 3001};
        regionHeightSlope_60_FC_y = {0.31, 0.63, 0.82, 1.20, 2.43, 86.21, -2.65, -1.22, -0.89, -0.70, -0.35};
        regionDist_60_FC_y = {8, 4, 3, 2, 1, 0, -1, -2, -3, -4, -8};
        
        regionHeight_30_FC_x = {1208, 1151, 1095, 1051, 1009, 981, 950, 920, 894, 868, 849, 828, 811, 794, 779, 765, 751, 740, 722, 677, 632, 597, 584, 0};
        regionDist_30_FC_x = {12.5, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 35, 40, 45, 50, 100}; 
        regionHeight_30_FC_y = {-80, 185, 496, 762, 994, 1266, 1543, 1880, 2211};
        regionHeightSlope_30_FC_y = {0.82, 1.08, 2.04, 4.48, 623, -4.32, -1.85, -0.95, -0.71};
        regionDist_30_FC_y = {4, 3, 2, 1, 0, -1, -2, -3, -4};

        regionHeight_30_SL_x = {1861, 1744, 1635, 1553, 1470, 1410, 1360, 1312, 1266, 1228, 1195, 1160, 1132, 1107, 1080, 1060, 969, 905, 860, 833, 800, 0};
        regionDist_30_SL_x = {10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 30, 35, 40, 45, 50, 100}; 
        regionHeight_30_SL_y = {121, 480, 938, 1507, 1873, 2264};
        regionHeightSlope_30_SL_y = {30.46, -3.41, -1.60, -0.96, -0.76, -0.62};
        regionDist_30_SL_y = {0, 1, 2, 3, 4, 5};
        
        regionHeight_30_SR_x = {1799, 1695, 1585, 1473, 1397, 1349, 1275, 1241, 1205, 1156, 1120, 1088, 1069, 1040, 1017, 999, 983, 957, 943, 922, 907, 843, 799, 753, 726, 0};
        regionDist_30_SR_x = {10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 35, 40, 45, 50, 100}; 
        regionHeight_30_SR_y = {-892, -359, -20, 266, 598};
        regionHeightSlope_30_SR_y = {0.69, 0.90, 1.05, 1.27, 1.67};
        regionDist_30_SR_y = {5, 4, 3, 2, 1};

        regionHeight_120_FC_x = {1207, 1040, 797, 623, 501, 404, 319, 262, 212, 144, 100};
        regionDist_120_FC_x = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10}; 
        regionHeight_120_FC_y = {-709, -332, 0, 194, 540, 919, 1253, 1587, 1841, 2047, 2365};
        regionHeightSlope_120_FC_y = {0.94, 1.22, 1.65, 1.95, 3.49, 30.94, -5.05, -2.40, -1.76, -1.47, -1.18};
        regionDist_120_FC_y = {5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5};

        Lidar_offset_x = 1;
        Lidar_offset_y = 2.490/2;
    }  
    else if(truckId == 2)
    {
        // regionHeight_60_FC_x = {1207, 1120, 1044, 985, 940, 900, 868, 840, 818, 796, 780, 764, 748, 736, 724, 680, 648, 625, 608, 596, 584, 0};
        // regionDist_60_FC_x = {6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 25, 30, 35, 40, 45, 50, 100}; 
        // regionHeight_60_FC_y = {-1284, 135, 463, 741, 990, 1262, 1543, 1806, 2930};
        // regionHeightSlope_60_FC_y = {0.32, 0.869, 1.45, 3.36, 0, -2.45, -1.3, -0.9, -0.36};
        // regionDist_60_FC_y = {8, 3, 2, 1, 0, -1, -2, -3, -8};

        // regionHeight_60_FC_x = {1207, 1102, 1033, 975, 928, 890, 857, 827, 803, 783, 765, 748, 733, 720, 708, 661, 627, 607, 589, 574, 563, 0};
        // regionDist_60_FC_x = {6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 25, 30, 35, 40, 45, 50, 100}; 
        // regionHeight_60_FC_y = {-1141, -217, 40, 305, 588, 876, 1150, 1422, 1700, 1972, 3103};
        // regionHeightSlope_60_FC_y = {0.375, 0.695, 0.914, 1.35, 2.84, -23.667, -2.414, -1.288, -0.862, -0.66, -0.328};
        // regionDist_60_FC_y = {8, 4, 3, 2, 1, 0, -1, -2, -3, -4, -8};

        regionHeight_60_FC_x = {1207, 1102, 1033, 975, 928, 890, 857, 827, 803, 783, 765, 748, 733, 720, 708, 661, 627, 607, 589, 574, 563, 0};
        regionDist_60_FC_x = {6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 25, 30, 35, 40, 45, 50, 100}; 
        regionHeight_60_FC_y = {-1284, 135, 463, 741, 990, 1262, 1543, 1806, 2930};
        regionHeightSlope_60_FC_y = {0.32, 0.869, 1.45, 3.36, 0, -2.45, -1.3, -0.9, -0.36};
        regionDist_60_FC_y = {8, 3, 2, 1, 0, -1, -2, -3, -8};

        // regionHeight_30_FC_x = {1207, 1161, 1106, 1058, 1015, 978, 946, 916, 890, 791, 726, 681, 648, 624, 612, 0}; 
        // regionDist_30_FC_x = {12.5, 13, 14, 15, 16, 17, 18, 19, 20, 25, 30, 35, 40, 45, 50, 100}; 
        // regionHeight_30_FC_y = {104, 409, 705, 993, 1196, 1565, 1854};
        // regionHeightSlope_30_FC_y = {0.84, 1.32, 2.08, 59.9, -8.4, -1.55, -1.01};
        // regionDist_30_FC_y = {3, 2, 1, 0, -1, -2, -3};

        // regionHeight_30_FC_x = {1207, 1153, 1092, 1043, 1002, 965, 933, 900, 870, 772, 700, 650, 610, 580, 555, 0}; 
        // regionDist_30_FC_x = {12.5, 13, 14, 15, 16, 17, 18, 19, 20, 25, 30, 35, 40, 45, 50, 100}; 
        // regionHeight_30_FC_y = {-93, 210, 542, 858, 1172, 1480, 1800};
        // regionHeightSlope_30_FC_y = {0.934, 1.397, 2.958, -44.703, -2.453, -1.319, -0.875};
        // regionDist_30_FC_y = {3, 2, 1, 0, -1, -2, -3};

        regionHeight_30_FC_x = {1207, 1153, 1092, 1043, 1002, 965, 933, 900, 870, 772, 700, 650, 610, 580, 555, 0}; 
        regionDist_30_FC_x = {12.5, 13, 14, 15, 16, 17, 18, 19, 20, 25, 30, 35, 40, 45, 50, 100}; 
        regionHeight_30_FC_y = {-404, 104, 409, 705, 993, 1196, 1565, 1854, 2464};
        regionHeightSlope_30_FC_y = {0.517, 0.84, 1.32, 2.08, 59.9, -8.4, -1.55, -1.01, -0.526};
        regionDist_30_FC_y = {4, 3, 2, 1, 0, -1, -2, -3, -4};

        regionHeight_30_SL_x = {1565, 1463, 1341, 1276, 1210, 1137, 1069, 1001, 951, 920, 885, 760, 676, 620, 581, 537, 514, 0}; 
        regionDist_30_SL_x = {10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 25, 30, 35, 40, 45, 50, 100};
        regionHeight_30_SL_y = {318, 701, 1151, 1592, 1837};
        regionHeightSlope_30_SL_y = {-5.56, -3.09, -1.87, -1.32, -1.18}; 
        regionDist_30_SL_y = {0, 1, 2, 3, 4}; 
        
        // regionHeight_30_SL_x = {1861, 1744, 1635, 1553, 1470, 1410, 1360, 1312, 1266, 1228, 1195, 1160, 1132, 1107, 1080, 1060, 969, 905, 860, 833, 800, 0};
        // regionDist_30_SL_x = {10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 30, 35, 40, 45, 50, 100}; 
        // regionHeight_30_SL_y = {121, 480, 938, 1507, 1873, 2264};
        // regionHeightSlope_30_SL_y = {30.46, -3.41, -1.60, -0.96, -0.76, -0.62};
        // regionDist_30_SL_y = {0, 1, 2, 3, 4, 5};

        regionHeight_30_SR_x = {1803, 1400, 1250, 1128, 1075, 1000, 935, 866, 825, 789, 750, 708, 657, 611, 481, 400, 337, 203, 270, 250};
        // regionHeight_30_SR_x = {1903, 1680, 1530, 1378, 1225, 1112, 1035, 966, 895, 839, 800, 758, 707, 661, 531, 450, 387, 353, 320, 297};
        regionDist_30_SR_x = {7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 25, 30, 35, 40, 45, 50}; 
        regionHeight_30_SR_y = {-2067, -1039, -170, 525, 873};
        regionHeightSlope_30_SR_y = {0.543, 0.791, 1.29, 2.6, 4.51};
        regionDist_30_SR_y = {5, 4, 3, 2, 1};

        // regionHeight_30_SR_x = {1799, 1695, 1585, 1473, 1397, 1349, 1275, 1241, 1205, 1156, 1120, 1088, 1069, 1040, 1017, 999, 983, 957, 943, 922, 907, 843, 799, 753, 726, 0};
        // regionDist_30_SR_x = {10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 35, 40, 45, 50, 100}; 
        // regionHeight_30_SR_y = {-892, -359, -20, 266, 598};
        // regionHeightSlope_30_SR_y = {0.69, 0.90, 1.05, 1.27, 1.67};
        // regionDist_30_SR_y = {5, 4, 3, 2, 1};
        
        regionHeight_120_FC_x = {1207, 1037, 795, 619, 492, 396, 319, 255, 205, 163, 130};
        regionDist_120_FC_x = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10}; 
        regionHeight_120_FC_y = {-425, -104, 331, 660, 950, 1260, 1587, 1955, 2190};
        regionHeightSlope_120_FC_y = {1.06, 1.3, 1.66, 3.08, 0, -3.04, -2.14, -1.48, -1.22};
        regionDist_120_FC_y = {4, 3, 2, 1, 0, -1, -2, -3, -4};

        Lidar_offset_x = 1;
        Lidar_offset_y = 2.490/2;
    }

}

float DistanceEstimation::ComputeObjectXDist(int piexl_loc, std::vector<int> regionHeight, std::vector<float> regionDist)
{
    float distance = -1;
    float unitLength = 0.0;
    int bias = 0;
    float offset = 0.0;
    for (int i = 1; i < regionHeight.size(); i++)
    {
        if ((piexl_loc >= regionHeight[i] && piexl_loc <= regionHeight[i-1]))  
        {
            int regionpixel = regionHeight[i-1] - regionHeight[i];
            int regionMeter = regionDist[i] - regionDist[i-1];
            unitLength = float(regionMeter)/float(regionpixel);
            bias = piexl_loc - regionHeight[i];
            offset = unitLength * float(bias);
            distance = regionDist[i] - offset;
            // printf("region[%d~%d][%d~%d], X- distance: %f\n ", i-1, i, regionHeight[i-1], regionHeight[i], distance);
            // printf("piexl_loc: %d,  regionDist: %d, unit: %f, bias: %d, offset: %f\n", piexl_loc, regionDist[i], unitLength, bias, offset);
        }
        else if ((piexl_loc <= regionHeight[i] && piexl_loc >= regionHeight[i-1]))  
        {
            int regionpixel = regionHeight[i] - regionHeight[i-1];
            int regionMeter = regionDist[i] - regionDist[i-1];
            unitLength = float(regionMeter)/float(regionpixel);
            bias = regionHeight[i] - piexl_loc;
            offset = unitLength * float(bias);
            distance = regionDist[i] - offset;
            // printf("region[%d~%d][%d~%d], X- distance: %f\n ", i-1, i, regionHeight[i-1], regionHeight[i], distance);
            // printf("piexl_loc: %d,  regionDist: %d, unit: %f, bias: %d, offset: %f\n", piexl_loc, regionDist[i], unitLength, bias, offset);
        }
        else
        {
            if (piexl_loc > regionHeight[0])
                distance = 6;
        }
    }
    // printf("piexl_loc: %d,  regionDist: %d, unit: %f, bias: %d, offset: %f\n", piexl_loc, regionDist[i], unitLength, bias, offset);
    int multiplier = pow(10, 2);
    distance = int(distance * multiplier) / (multiplier*1.0); 

    
    return distance;
}

float DistanceEstimation::ComputeObjectYDist(int piexl_loc_y, int piexl_loc_x, std::vector<int> regionHeight, std::vector<float> regionHeightSlope_y, std::vector<float> regionDist, int img_h)
{
    float distance = -1;
    float unitLength = 0.0;
    int bias = 0;
    float offset = 0.0;

    std::vector<int> regionHeight_new = regionHeight;
    
    // std::cout << "piexl_loc_x: " << piexl_loc_x <<  ", piexl_loc_y: " << piexl_loc_y << std::endl;
    for (int i = 0; i < regionHeight.size(); i++)
    {
        int y = img_h - piexl_loc_x;
        if (regionHeightSlope_y[i] !=0)
            regionHeight_new[i] = regionHeight[i] + int((1/regionHeightSlope_y[i])*y);
        else 
            regionHeight_new[i] = regionHeight[i];
        // printf("region[%d], region:%d, new region:%d, \n ", i, regionHeight[i], regionHeight_new[i]);
    }

    for (int i = 1; i < regionHeight_new.size(); i++)
    {
        if (piexl_loc_y >= regionHeight_new[i] && piexl_loc_y <= regionHeight_new[i-1])  
        {
            int regionpixel = regionHeight_new[i-1] - regionHeight_new[i];
            int regionMeter = regionDist[i]  - regionDist[i-1];
            if (regionpixel != 0)
                unitLength = float(regionMeter)/float(regionpixel);
            bias = piexl_loc_y - regionHeight_new[i];
            offset = unitLength * float(bias);
            distance = regionDist[i] - offset;
            // printf("region[%d~%d][%d~%d], new region[%d~%d], Y- distance: %f\n ", i-1, i, regionHeight[i-1], regionHeight[i], regionHeight_new[i-1], regionHeight_new[i], distance);
            // printf("piexl_loc: %d,  regionDist: %d, unit: %f, bias: %d, offset: %f\n", piexl_loc_y, regionDist[i], unitLength, bias, offset);
        }
        else if (piexl_loc_y <= regionHeight_new[i] && piexl_loc_y >= regionHeight_new[i-1])  
        {
            int regionpixel = regionHeight_new[i] - regionHeight_new[i-1];
            int regionMeter = regionDist[i] - regionDist[i-1];
            if (regionpixel != 0)
                unitLength = float(regionMeter)/float(regionpixel);
            bias = regionHeight_new[i] - piexl_loc_y;
            offset = unitLength * float(bias);
            distance = regionDist[i] - offset;
            // printf("region[%d~%d][%d~%d], new region[%d~%d], Y- distance: %f\n ", i-1, i, regionHeight[i-1], regionHeight[i], regionHeight_new[i-1], regionHeight_new[i], distance);
            // printf("piexl_loc: %d,  regionDist: %d, unit: %f, bias: %d, offset: %f\n", piexl_loc_y, regionDist[i], unitLength, bias, offset);
        }
        else
        {
            if (piexl_loc_y < regionHeight_new[0])
                distance = 8;
            else if(piexl_loc_y > regionHeight_new[regionHeight_new.size()-1])
                distance = -8;
        }
    }  
    
    int multiplier = pow(10, 2);
    distance = int(distance * multiplier) / (multiplier*1.0); 
    
    return distance; 
}

float DistanceEstimation::ComputeObjectXDistWithSlope(int piexl_loc_y, int piexl_loc_x, std::vector<int> regionHeight, std::vector<float> regionHeightSlope_y, std::vector<float> regionDist, int img_h)
{
    float distance = -1;
    float unitLength = 0.0;
    int bias = 0;
    float offset = 0.0;

    std::vector<int> regionHeight_new = regionHeight;
    
    // std::cout << "piexl_loc_x: " << piexl_loc_x <<  ", piexl_loc_y: " << piexl_loc_y << std::endl;
    for (int i = 0; i < regionHeight.size(); i++)
    {
        int y = img_h - piexl_loc_x;
        if (regionHeightSlope_y[i] !=0)
            regionHeight_new[i] = regionHeight[i] + int((1/regionHeightSlope_y[i])*y);
        else 
            regionHeight_new[i] = regionHeight[i];
        // printf("region[%d], region:%d, new region:%d, \n ", i, regionHeight[i], regionHeight_new[i]);
    }

    for (int i = 1; i < regionHeight_new.size(); i++)
    {
        if (piexl_loc_y >= regionHeight_new[i] && piexl_loc_y <= regionHeight_new[i-1])  
        {
            int regionpixel = regionHeight_new[i-1] - regionHeight_new[i];
            int regionMeter = regionDist[i]  - regionDist[i-1];
            if (regionpixel != 0)
                unitLength = float(regionMeter)/float(regionpixel);
            bias = piexl_loc_y - regionHeight_new[i];
            offset = unitLength * float(bias);
            distance = regionDist[i] - offset;
            // printf("region[%d~%d][%d~%d], new region[%d~%d], Y- distance: %f\n ", i-1, i, regionHeight[i-1], regionHeight[i], regionHeight_new[i-1], regionHeight_new[i], distance);
            // printf("piexl_loc: %d,  regionDist: %d, unit: %f, bias: %d, offset: %f\n", piexl_loc_y, regionDist[i], unitLength, bias, offset);
        }
        else if (piexl_loc_y <= regionHeight_new[i] && piexl_loc_y >= regionHeight_new[i-1])  
        {
            int regionpixel = regionHeight_new[i] - regionHeight_new[i-1];
            int regionMeter = regionDist[i] - regionDist[i-1];
            if (regionpixel != 0)
                unitLength = float(regionMeter)/float(regionpixel);
            bias = regionHeight_new[i] - piexl_loc_y;
            offset = unitLength * float(bias);
            distance = regionDist[i] - offset;
            // printf("region[%d~%d][%d~%d], new region[%d~%d], Y- distance: %f\n ", i-1, i, regionHeight[i-1], regionHeight[i], regionHeight_new[i-1], regionHeight_new[i], distance);
            // printf("piexl_loc: %d,  regionDist: %d, unit: %f, bias: %d, offset: %f\n", piexl_loc_y, regionDist[i], unitLength, bias, offset);
        }
        else
        {
            if (piexl_loc_y < regionHeight_new[0])
                distance = 8;
            else if(piexl_loc_y > regionHeight_new[regionHeight_new.size()-1])
                distance = -8;
        }
    }  
    
    int multiplier = pow(10, 2);
    distance = int(distance * multiplier) / (multiplier*1.0); 
    
    return distance; 
}

int CheckPointInArea(cv::Point RightLinePoint1, cv::Point RightLinePoint2, cv::Point LeftLinePoint1, cv::Point LeftLinePoint2, int object_x1, int object_y2)
{
    int point0 = 0;
    int point1 = 1;
    int point2 = 2;
    ///right
    int C1 = (RightLinePoint1.x - RightLinePoint2.x)*(object_y2 - RightLinePoint2.y) - (object_x1 - RightLinePoint2.x)*(RightLinePoint1.y - RightLinePoint2.y);
    ///left
    int C2 = (LeftLinePoint1.x - LeftLinePoint2.x)*(object_y2 - LeftLinePoint2.y) - (object_x1 - LeftLinePoint2.x)*(LeftLinePoint1.y - LeftLinePoint2.y);

    if (C1 > 0)
        return point2;
    else if (C2 < 0)
        return point1; 
    else
        return point0;  
}
int box_shrink(int cam_id, std::vector<int> Points_src, std::vector<int> &Points_dst)
{
    int edge_left, edge_right;
    int area_id = 1; //1: left, 2:right

    cv::Point LeftLinePoint1;
    cv::Point LeftLinePoint2;
    cv::Point RightLinePoint1;
    cv::Point RightLinePoint2; 

    int box_center_x = (Points_src[1] + Points_src[2]) / 2; // get x center of objects
    if (cam_id == 2)
    {
        LeftLinePoint1 = cv::Point(891, 584);
        LeftLinePoint2 = cv::Point(360, 1207);
        RightLinePoint1 = cv::Point(1020, 584);
        RightLinePoint2 = cv::Point(1470, 1207);        
        area_id = CheckPointInArea(RightLinePoint1, RightLinePoint2, LeftLinePoint1, LeftLinePoint2, box_center_x, Points_src[3]);
    }

    double shrink_ratio_left;
    double shrink_ratio_right;
    double shrink_ratio_center;

    switch(Points_src[0])
    {
        case 0:{ //0:person
            shrink_ratio_left = 1;
            shrink_ratio_right = 1;
            shrink_ratio_center = 0.8;
            break;
        }
        //1:bicycle, 3:motorbike
        case 1:{
            shrink_ratio_left = 1;
            shrink_ratio_right = 1;
            shrink_ratio_center = 0.9;
            break;
        } 
        case 3:{
            shrink_ratio_left = 1;
            shrink_ratio_right = 1;
            shrink_ratio_center = 0.9;
            break;
        }
        //2:car
        case 2:{ 
            shrink_ratio_left = 1;
            shrink_ratio_right = 1;
            shrink_ratio_center = 0.5;
            break;
        }
        //5:bus, 7:truck
        case 5:{ 
            shrink_ratio_left = 1;
            shrink_ratio_right = 1;
            shrink_ratio_center = 0.7;            
            break;
        }       
        case 7:{ 
            shrink_ratio_left = 1;
            shrink_ratio_right = 1;
            shrink_ratio_center = 0.7;            
            break;
        }       
        default:{
            shrink_ratio_left = 0.1;
            shrink_ratio_right = 0.1;
            shrink_ratio_center = 0.1;
            break;
        }
    }
    //std::cout << "class id: " << Points_src[0] << ", shrink_ratio_left: " << shrink_ratio_left << ", shrink_ratio_right: " << shrink_ratio_right << std::endl;

    // Over left edge
    if(area_id == 0) //over right edge
    {
        // shrink x
        Points_dst[1] = (box_center_x - Points_src[1])*shrink_ratio_center + Points_src[1];
        Points_dst[2] = Points_src[2] - (Points_src[2] - box_center_x)*shrink_ratio_center;
    }
    else if(area_id == 1)
    {   
        // shrink x
        Points_dst[1] = (box_center_x - Points_src[1])*(shrink_ratio_left-0.3) + Points_src[1];
        Points_dst[2] = Points_src[2] - (Points_src[2] - box_center_x)*shrink_ratio_left;
    }
    else if(area_id == 2) //over right edge
    {
        // shrink x
        Points_dst[1] = (box_center_x - Points_src[1])*shrink_ratio_right + Points_src[1];
        Points_dst[2] = Points_src[2] - (Points_src[2] - box_center_x)*(shrink_ratio_right-0.3);
    }
    return 0;
}
msgs::BoxPoint DistanceEstimation::Get3dBBox(msgs::PointXYZ p0, msgs::PointXYZ p3, int class_id, int cam_id)
{
    msgs::PointXYZ p1, p2, p4, p5, p6, p7;
    msgs::BoxPoint point8;
    ///3D bounding box
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
    ///obstacle_w

    float obstacle_h = 2, obstacle_l = 2 , obstacle_w = 2;
    if(class_id == 0) { obstacle_h = 1.8; obstacle_l = 0.33; obstacle_w = 0.6;}
    else if(class_id == 1 || class_id == 3) { obstacle_h = 1.8; obstacle_l = 2.5; obstacle_w = 0.6;}
    else if(class_id == 2) { obstacle_h = 1.5; obstacle_l = 5; obstacle_w = 2;}
    else if(class_id == 5 || class_id == 7) {obstacle_h = 2; obstacle_l = 7; obstacle_w = 2.5;} 

    ///1
    if(cam_id == 1 || cam_id == 2 || cam_id == 3 || cam_id == 5 || cam_id == 8 )
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
    else if (cam_id == 4)
    {
        /// Camera Perspective   ///  Spec view
        ///   p6------p2         ///   p5------p6
        ///   /|  2   /|         ///   /|  2   /|
        /// p5-|----p7 |         /// p1-|----p2 |
        ///  |p7----|-p3   ->    ///  |p4----|-p7
        ///  |/  1  | /          ///  |/  1  | /
        /// p4-----P0            /// p0-----P3

        msgs::PointXYZ p0_cam, p3_cam, p4_cam, p7_cam;
        p0_cam = p0;
        p3_cam = p3;
        p4_cam = p0_cam;
        p4_cam.y = p4_cam.y - obstacle_w;
        p7_cam = p3_cam;
        p7_cam.y = p7_cam.y - obstacle_w;

        p0 = p4_cam;
        p3 = p0_cam;
        p4 = p7_cam;
        p7 = p3_cam;
    }
    else if (cam_id == 6)
    {
        /// Camera Perspective   ///  Spec view
        ///   p1------p5         ///   p5------p6
        ///   /|  2   /|         ///   /|  2   /|
        /// p2-|----p6 |         /// p1-|----p2 |
        ///  |p0----|-p4   ->    ///  |p4----|-p7
        ///  |/  1  | /          ///  |/  1  | /
        /// p3-----P7            /// p0-----P3

        msgs::PointXYZ p0_cam, p3_cam, p4_cam, p7_cam;
        p0_cam = p0;
        p3_cam = p3;
        p4_cam = p0_cam;
        p4_cam.y = p4_cam.y + obstacle_w;
        p7_cam = p3_cam;
        p7_cam.y = p7_cam.y + obstacle_w;

        p0 = p3_cam;
        p3 = p7_cam;
        p4 = p0_cam;
        p7 = p4_cam;
    }
    else if (cam_id == 7 || cam_id == 9)
    {
        /// Camera Perspective   ///  Spec view
        ///   p2------p1         ///   p5------p6
        ///   /|  2   /|         ///   /|  2   /|
        /// p6-|----p5 |         /// p1-|----p2 |
        ///  |p3----|-p0   ->    ///  |p4----|-p7
        ///  |/  1  | /          ///  |/  1  | /
        /// p7-----P4            /// p0-----P3

        msgs::PointXYZ p0_cam, p3_cam, p4_cam, p7_cam;

        p0_cam = p0;
        p3_cam = p3;
        p4_cam = p0_cam;
        p4_cam.x = p4_cam.x - obstacle_l;
        p7_cam = p3_cam;
        p7_cam.x = p7_cam.x - obstacle_l;

        p0 = p7_cam;
        p3 = p4_cam;
        p4 = p3_cam;
        p7 = p0_cam;
    }

    ///2
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
    ///obstacle_w
    
    /// class id
    /// 0: person
    /// 1: bicycle
    /// 2: car
    /// 3: motorbike
    /// 5: bus
    /// 7: truck
    float obstacle_h = 2, obstacle_l = 2 , obstacle_w = 2;
    if(class_id == 0) { obstacle_h = 1.8; obstacle_l = 0.33; obstacle_w = 0.6;}
    else if(class_id == 1 || class_id == 3) { obstacle_h = 1.8; obstacle_l = 2.5; obstacle_w = 0.6;}
    else if(class_id == 2) { obstacle_h = 1.5; obstacle_l = 5; obstacle_w = 2;}
    else if(class_id == 5 || class_id == 7) {obstacle_h = 2; obstacle_l = 7; obstacle_w = 2.5;} 

    // if(cam_id == 2 || cam_id == 5 || cam_id == 7 || cam_id == 8 || cam_id == 9)
    // {
    //     std::vector<int> PointsSrc = {class_id, x1, x2, y2};
    //     std::vector<int> PointsDst = {class_id, x1, x2, y2};
    //     box_shrink(cam_id, PointsSrc, PointsDst);
    //     x1 = PointsDst[1]; x2 = PointsDst[2];
    // }
    ///1
    p0 = GetPointDist(x1, y2, cam_id);
    p3 = GetPointDist(x2, y2, cam_id);

    ///1
    if(cam_id == 1 || cam_id == 2 || cam_id == 3 || cam_id == 5 || cam_id == 8 )
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
    else if (cam_id == 4)
    {
        /// Camera Perspective   ///  Spec view
        ///   p6------p2         ///   p5------p6
        ///   /|  2   /|         ///   /|  2   /|
        /// p5-|----p7 |         /// p1-|----p2 |
        ///  |p7----|-p3   ->    ///  |p4----|-p7
        ///  |/  1  | /          ///  |/  1  | /
        /// p4-----P0            /// p0-----P3

        msgs::PointXYZ p0_cam, p3_cam, p4_cam, p7_cam;
        offset_y = Lidar_offset_y;

        p0_cam.x = p0.x *(-1);
        p3_cam.x = p3.x *(-1);
        p0_cam.z = p0.y + offset_y;
        p3_cam.z = p3.y + offset_y;
        p0_cam.z = p0.z;
        p3_cam.z = p3.z;
        p4_cam = p0_cam;
        p4_cam.y = p4_cam.y - obstacle_w;
        p7_cam = p3_cam;
        p7_cam.y = p7_cam.y - obstacle_w;

        p0 = p4_cam;
        p3 = p0_cam;
        p4 = p7_cam;
        p7 = p3_cam;
    }
    else if (cam_id == 6)
    {
        /// Camera Perspective   ///  Spec view
        ///   p1------p5         ///   p5------p6
        ///   /|  2   /|         ///   /|  2   /|
        /// p2-|----p6 |         /// p1-|----p2 |
        ///  |p0----|-p4   ->    ///  |p4----|-p7
        ///  |/  1  | /          ///  |/  1  | /
        /// p3-----P7            /// p0-----P3

        msgs::PointXYZ p0_cam, p3_cam, p4_cam, p7_cam;
        offset_y = Lidar_offset_y*(-1);

        p0_cam.x = p0.x *(-1);
        p3_cam.x = p3.x *(-1);
        p0_cam.z = p0.y + offset_y;
        p3_cam.z = p3.y + offset_y;
        p0_cam.z = p0.z;
        p3_cam.z = p3.z;
        p4_cam = p0_cam;
        p4_cam.y = p4_cam.y + obstacle_w;
        p7_cam = p3_cam;
        p7_cam.y = p7_cam.y + obstacle_w;

        p0 = p3_cam;
        p3 = p7_cam;
        p4 = p0_cam;
        p7 = p4_cam;
    }
    else if (cam_id == 7)
    {
        /// Camera Perspective   ///  Spec view
        ///   p2------p1         ///   p5------p6
        ///   /|  2   /|         ///   /|  2   /|
        /// p6-|----p5 |         /// p1-|----p2 |
        ///  |p3----|-p0   ->    ///  |p4----|-p7
        ///  |/  1  | /          ///  |/  1  | /
        /// p7-----P4            /// p0-----P3

        msgs::PointXYZ p0_cam, p3_cam, p4_cam, p7_cam;
        
        offset_y = Lidar_offset_y;

        p0_cam.x = p0.x *(-1);
        p3_cam.x = p3.x *(-1);
        p0_cam.y = p0.y + offset_y;
        p3_cam.y = p3.y + offset_y;
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
    else if (cam_id == 9)
    {
        /// Camera Perspective   ///  Spec view
        ///   p2------p1         ///   p5------p6
        ///   /|  2   /|         ///   /|  2   /|
        /// p6-|----p5 |         /// p1-|----p2 |
        ///  |p3----|-p0   ->    ///  |p4----|-p7
        ///  |/  1  | /          ///  |/  1  | /
        /// p7-----P4            /// p0-----P3

        msgs::PointXYZ p0_cam, p3_cam, p4_cam, p7_cam;
        offset_y = Lidar_offset_y*(-1);
        
        p0_cam.x = p0.x *(-1);
        p3_cam.x = p3.x *(-1);
        p0_cam.y = p0.y *(-1) + offset_y;
        p3_cam.y = p3.y *(-1) + offset_y;
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
    ///2
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

    // std::cout << "3D: p0(x, y, z) = (" << point8.p0.x  << ", " <<  point8.p0.y <<  ", " << point8.p0.z << ")." << std::endl;
    // std::cout << "3D: p1(x, y, z) = (" << point8.p1.x  << ", " <<  point8.p1.y <<  ", " << point8.p1.z << ")." << std::endl;
    // std::cout << "3D: p2(x, y, z) = (" << point8.p2.x  << ", " <<  point8.p2.y <<  ", " << point8.p2.z << ")." << std::endl;
    // std::cout << "3D: p3(x, y, z) = (" << point8.p3.x  << ", " <<  point8.p3.y <<  ", " << point8.p3.z << ")." << std::endl;
    // std::cout << "3D: p4(x, y, z) = (" << point8.p4.x  << ", " <<  point8.p4.y <<  ", " << point8.p4.z << ")." << std::endl;
    // std::cout << "3D: p5(x, y, z) = (" << point8.p5.x  << ", " <<  point8.p5.y <<  ", " << point8.p5.z << ")." << std::endl;
    // std::cout << "3D: p6(x, y, z) = (" << point8.p6.x  << ", " <<  point8.p6.y <<  ", " << point8.p6.z << ")." << std::endl;
    // std::cout << "3D: p7(x, y, z) = (" << point8.p7.x  << ", " <<  point8.p7.y <<  ", " << point8.p7.z << ")." << std::endl;

    return point8; 
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
    float offset_x = 0;
    int x_loc = y;
    int y_loc = x;
    int img_h = 1208;
    int mode = 1;


    if (cam_id == 2)
    {
        regionHeight_x = regionHeight_60_FC_x;
        regionDist_x = regionDist_60_FC_x;
        regionHeight_y = regionHeight_60_FC_y;
        regionHeightSlope_y = regionHeightSlope_60_FC_y;
        regionDist_y = regionDist_60_FC_y;
        offset_x = Lidar_offset_x;
    }
    else if (cam_id == 5)
    {
        regionHeight_x = regionHeight_120_FC_x;
        regionDist_x = regionDist_120_FC_x;
        regionHeight_y = regionHeight_120_FC_y;
        regionHeightSlope_y = regionHeightSlope_120_FC_y;
        regionDist_y = regionDist_120_FC_y;
        offset_x = Lidar_offset_x;
    }
    else if (cam_id == 7)
    {
        regionHeight_x = regionHeight_30_SL_x;
        regionDist_x = regionDist_30_SL_x;
        regionHeight_y = regionHeight_30_SL_y;
        regionHeightSlope_y = regionHeightSlope_30_SL_y;
        regionDist_y = regionDist_30_SL_y;
        offset_x = Lidar_offset_x;
        img_h = 1920;
    }
    else if (cam_id == 8)
    {
        regionHeight_x = regionHeight_30_FC_x;
        regionDist_x = regionDist_30_FC_x;
        regionHeight_y = regionHeight_30_FC_y;
        regionHeightSlope_y = regionHeightSlope_30_FC_y;
        regionDist_y = regionDist_30_FC_y;
        offset_x = Lidar_offset_x;
    }
    else if (cam_id == 9)
    {
        regionHeight_x = regionHeight_30_SR_x;
        regionDist_x = regionDist_30_SR_x;
        regionHeight_y = regionHeight_30_SR_y;
        regionHeightSlope_y = regionHeightSlope_30_SR_y;
        regionDist_y = regionDist_30_SR_y;
        offset_x = Lidar_offset_x;
        img_h = 1920;
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
        y_distMeter = ComputeObjectYDist(y_loc, x_loc, regionHeight_y, regionHeightSlope_y, regionDist_y, img_h);

    p0.x = x_distMeter + offset_x;
    p0.y = y_distMeter;
    p0.z = Lidar_offset_z;

    return p0;
}
