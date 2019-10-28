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

    std::vector<int> regionHeight_30_SL_x;
    std::vector<float> regionDist_30_SL_x; 
    std::vector<int>regionHeight_30_SL_y;
    std::vector<float> regionDist_30_SL_y;
    std::vector<float> regionHeightSlope_30_SL_y;

    std::vector<int> regionHeight_30_SR_x;
    std::vector<float> regionDist_30_SR_x; 
    std::vector<int>regionHeight_30_SR_y;
    std::vector<float> regionDist_30_SR_y;
    std::vector<float> regionHeightSlope_30_SR_y;
    
    std::vector<int> regionHeight_120_FC_x;
    std::vector<float> regionDist_120_FC_x; 
    std::vector<int> regionHeight_120_FC_y;
    std::vector<float> regionDist_120_FC_y;
    std::vector<float> regionHeightSlope_120_FC_y;

    float Lidar_offset_x = 0;
    float Lidar_offset_y = 0;
    float Lidar_offset_z = -3;
    int truckId = 1;

    float ComputeObjectXDist(int piexl_loc, std::vector<int> regionHeight, std::vector<float> regionDist);
    float ComputeObjectYDist(int piexl_loc_y, int piexl_loc_x, std::vector<int> regionHeight, std::vector<float> regionHeightSlope_y, std::vector<float> regionDist, int img_h);
    msgs::PointXYZ GetPointDist(int x, int y, int cam_id);

public:
    void init(int truck_id);
    msgs::BoxPoint Get3dBBox(int x1, int y1, int x2, int y2, int class_id, int cam_id);
    msgs::BoxPoint Get3dBBox(msgs::PointXYZ p0, msgs::PointXYZ p3, int class_id, int cam_id);
};

#endif /*DISTANCEESTIMATION_H_*/
