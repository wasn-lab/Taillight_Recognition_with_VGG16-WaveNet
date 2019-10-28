#ifndef DISTANCEESTIMATION_H_
#define DISTANCEESTIMATION_H_

//ROS message
#include <msgs/BoxPoint.h>
#include <msgs/PointXYZ.h>

class DistanceEstimation{
private:

    std::vector<int> regionHeight_60_FC_x;
    std::vector<float> regionDist_60_FC_x; 
    std::vector<int> regionHeight_60_FC_y;
    std::vector<float> regionDist_60_FC_y;
    std::vector<float> regionHeightSlope_60_FC_y;

    std::vector<int> regionHeight_30_FC_x;
    std::vector<float> regionDist_30_FC_x; 
    std::vector<int>regionHeight_30_FC_y;
    std::vector<float> regionDist_30_FC_y;
    std::vector<float> regionHeightSlope_30_FC_y;
    
    std::vector<int> regionHeight_120_FC_x;
    std::vector<float> regionDist_120_FC_x; 
    std::vector<int>regionHeight_120_FC_y;
    std::vector<float> regionDist_120_FC_y;
    std::vector<float> regionHeightSlope_120_FC_y;

    std::vector<int> regionHeight_30_R_x;
    std::vector<float> regionDist_30_R_x; 
    std::vector<int>regionHeight_30_R_y;
    std::vector<float> regionHeightSlope_30_R_y;
    std::vector<float> regionDist_30_R_y;

    float offset;
    float Lidar_offset_x = 0;
    float Lidar_offset_z = -3;

    float ComputeObjectXDist(int piexl_loc, std::vector<int> regionHeight, std::vector<float> regionDist);
    float ComputeObjectYDist(int piexl_loc_y, int piexl_loc_x, std::vector<int> regionHeight, std::vector<float> regionHeightSlope_y, std::vector<float> regionDist, int mode);
    

public:
    DistanceEstimation();
msgs::PointXYZ GetPointDist(int x, int y, int cam_id);
    // msgs::BoxPoint Get3dBBox(int x1, int y1, int x2, int y2, int class_id, int cam_id);
    // msgs::BoxPoint Get3dBBox(msgs::PointXYZ p0, msgs::PointXYZ p3, int class_id, int cam_id);
};

#endif /*DISTANCEESTIMATION_H_*/
