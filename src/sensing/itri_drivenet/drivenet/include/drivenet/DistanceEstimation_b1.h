#ifndef DISTANCEESTIMATION_H_
#define DISTANCEESTIMATION_H_

//ROS message
#include <msgs/BoxPoint.h>
#include <msgs/PointXYZ.h>

class DistanceEstimation{
private:

    /// camId: 1
    std::vector<int> regionHeight_60_FR_x;
    std::vector<float> regionDist_60_FR_x; 
    std::vector<int> regionHeight_60_FR_y;
    std::vector<float> regionDist_60_FR_y;
    std::vector<float> regionHeightSlope_60_FR_y;

    /// camId: 2
    std::vector<int> regionHeight_60_FC_x;
    std::vector<float> regionDist_60_FC_x; 
    std::vector<int> regionHeight_60_FC_y;
    std::vector<float> regionDist_60_FC_y;
    std::vector<float> regionHeightSlope_60_FC_y;

    /// camId: 3
    std::vector<int> regionHeight_60_FL_x;
    std::vector<float> regionDist_60_FL_x; 
    std::vector<int> regionHeight_60_FL_y;
    std::vector<float> regionDist_60_FL_y;
    std::vector<float> regionHeightSlope_60_FL_y;

    /// camId: 4   
    std::vector<int> regionHeight_120_FT_x;
    std::vector<float> regionDist_120_FT_x; 
    std::vector<int> regionHeight_120_FT_y;
    std::vector<float> regionDist_120_FT_y;
    std::vector<float> regionHeightSlope_120_FT_y;

    /// camId: 5 
    std::vector<int> regionHeight_120_RF_x;
    std::vector<float> regionDist_120_RF_x; 
    std::vector<int>regionHeight_120_RF_y;
    std::vector<float> regionDist_120_RF_y;
    std::vector<float> regionHeightSlope_120_RF_y;
    
    /// camId: 6
    std::vector<int> regionHeight_120_RB_x;
    std::vector<float> regionDist_120_RB_x; 
    std::vector<int> regionHeight_120_RB_y;
    std::vector<float> regionDist_120_RB_y;
    std::vector<float> regionHeightSlope_120_RB_y;

    /// camId: 7
    std::vector<int> regionHeight_120_LF_x;
    std::vector<float> regionDist_120_LF_x; 
    std::vector<int> regionHeight_120_LF_y;
    std::vector<float> regionDist_120_LF_y;
    std::vector<float> regionHeightSlope_120_LF_y;

    /// camId: 8    
    std::vector<int> regionHeight_120_LR_x;
    std::vector<float> regionDist_120_LR_x; 
    std::vector<int> regionHeight_120_LR_y;
    std::vector<float> regionDist_120_LR_y;
    std::vector<float> regionHeightSlope_120_LR_y;

    /// camId: 9   
    std::vector<int> regionHeight_120_BT_x;
    std::vector<float> regionDist_120_BT_x; 
    std::vector<int> regionHeight_120_BT_y;
    std::vector<float> regionDist_120_BT_y;
    std::vector<float> regionHeightSlope_120_BT_y;

    float Lidar_offset_x = 0;
    float Lidar_offset_y = 0;
    float Lidar_offset_z = -3;
    int carId = 1;

    float ComputeObjectXDist(int piexl_loc, std::vector<int> regionHeight, std::vector<float> regionDist);
    float ComputeObjectXDistWithSlope(int piexl_loc_y, int piexl_loc_x, std::vector<int> regionHeight, std::vector<float> regionHeightSlope_x, std::vector<float> regionDist, int img_w);   
    float ComputeObjectYDist(int piexl_loc_y, int piexl_loc_x, std::vector<int> regionHeight, std::vector<float> regionHeightSlope_y, std::vector<float> regionDist, int img_h);
    msgs::PointXYZ GetPointDist(int x, int y, int cam_id);

public:
    void init(int carId);
    msgs::BoxPoint Get3dBBox(int x1, int y1, int x2, int y2, int class_id, int cam_id);
    msgs::BoxPoint Get3dBBox(msgs::PointXYZ p0, msgs::PointXYZ p3, int class_id, int cam_id);
};

#endif /*DISTANCEESTIMATION_H_*/
