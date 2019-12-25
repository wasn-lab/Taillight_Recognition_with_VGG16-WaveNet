/*
 * all_header.h
 *
 *  Created on: May 24, 2019
 *      Author: root, BensonHuang@itri.com.tw
 */

#ifndef ALL_HEADER_H_
#define ALL_HEADER_H_


// ROS libraries
//--------------------------------------------//
#include <ros/ros.h>
#include <ros/package.h>
// tf2
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <geometry_msgs/TransformStamped.h>

// MSG: (ROS std_msgs)
#include <std_msgs/Bool.h>
#include <std_msgs/Int32.h>
#include <std_msgs/String.h>

// MSG: GeoPoseStamped
#include <geometry_msgs/PoseStamped.h>

// MSG: Image
// #include "opencv2/opencv.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv/cv.hpp>
//
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>

// MSG: ITRIPointCloud
#include <msgs/PointCloud.h>
// MSG: ITRI3DBoundingBox
#include <msgs/LidRoi.h>
// MSG: ITRICamObj
#include <msgs/CamObj.h>
// MSG: ITRIDetectedObjectArray
#include <msgs/DetectedObjectArray.h>

// MSG: ITRICarInfoCarA
#include <msgs/TaichungVehInfo.h>
// MSG: ITRICarInfo
#include <msgs/VehInfo.h>
// MSG: ITRIDynamicPath
#include <msgs/DynamicPath.h>
// MSG: ITRIFlagInfo
#include <msgs/Flag_Info.h>
// MSG: ITRITransObj
#include <msgs/TransfObj.h>

// MSG: GUI2_op
#include <opengl_test/GUI2_op.h>
//--------------------------------------------//

// Core libraries
#include <string>
#include <iostream>
#include <vector>
#include <utility> // std::pair, std::make_pair
// #include <mutex>   // mutex lock
#include <thread>
#include <chrono>

// Others usefull libraries
#include <sstream>
#include <iomanip> // For adjusting the precision
#include <ctime>
#include <time.h>
#include <algorithm>
using namespace cv;
// The SPSC non-blocking buffer
// #include <async_buffer.hpp>
#include <async_buffer_v2.hpp>


// Boost
#include <boost/bind.hpp>
#include <boost/any.hpp> // any type, Note: in c++17, this is also implemented in std::any


// PCL
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>








// #include <stdio.h>
// #include <stdlib.h>
// #include <stdint.h>
// #include <stdexcept>
// #include <fstream>
// #include <functional>
// #include <errno.h>
// #include <limits.h>
// #include <math.h>
// #include <omp.h>
// #include <unistd.h> //sleep


// #include <rosbag/bag.h>
// #include <geometry_msgs/PoseStamped.h>
// #include <sensor_msgs/PointCloud2.h>



// #include <pcl/point_cloud.h>
// #include <pcl/point_types.h>

// #include <pcl/io/pcd_io.h>
// #include <pcl/io/boost.h>

// #include <pcl/common/geometry.h>
// #include <pcl/common/common.h> //getMinMax3D
// #include <pcl/common/common_headers.h>
// #include <pcl/common/transforms.h> //transform
// #include <pcl/common/centroid.h>
// #include <pcl/common/time.h>



// #include <boost/asio.hpp>
// #include <boost/property_tree/ptree.hpp>
// #include <boost/property_tree/xml_parser.hpp>
// #include <boost/property_tree/ini_parser.hpp>



// #include <Eigen/Core>

using std::vector;
using std::string;
/*
using namespace std;
using namespace pcl;
using namespace Eigen;
*/

// user define

#define D2R (M_PI/180.0)
#define PATH_RAW_DATA "../../raw_data/"

// user define

struct POSE_MAP
{
    double tx;
    double ty;
    double tz;
    double rx;
    double ry;
    double rz;
    double qx;
    double qy;
    double qz;
    double qw;
};

/*
struct CLUSTER_INFO
{
    // Cluster
    PointCloud<PointXYZ> cloud;
    float hull_vol;
    float GRSD21[21];

    PointXYZ min;
    PointXYZ max;
    PointXYZ center;
    float dis_max_min;       // size of vehicle
    float dis_center_origin;
    float angle_from_x_axis;  // when z = 0
    float dx;
    float dy;
    float dz;

    Eigen::Matrix3f covariance;
    vector<PointXYZ> obb_vertex;  //oriented bounding box
    PointXYZ obb_center;
    float obb_dx;
    float obb_dy;
    float obb_dz;
    float obb_orient;  // when z = 0

    // Tracking
    bool found_num;
    int tracking_id;
    PointXYZ track_last_center;
    PointXYZ predict_next_center;
    PointXYZ velocity;  // use point to represent velocity x,y,z

    // Classification
    int cluster_tag;  // tag =0 (not key object) tag =1 (unknown object) tag =2 (pedestrian) tag =3 (motorcycle)  tag =4 (car)  tag =5 (bus)
    int confidence;

    // Output
    PointXY to_2d_PointWL[2];  // to_2d_PointWL[0] --> 2d image coordinate x,y     to_2d_PointWL[1] --> x = width y = height
    PointXY to_2d_points[4];   // to_2d_points[0]~[4] --> 2d image coordinate x,y

};
typedef struct CLUSTER_INFO CLUSTER_INFO;
*/

// to_string_with_precision
namespace all_header{
    template <typename T>
    std::string to_string_p(const T a_value, const int n = 6)
    {
        std::ostringstream out;
        out.precision(n);
        out << std::fixed << a_value;
        return out.str();
    }
}


#endif /* ALL_HEADER_H_ */
