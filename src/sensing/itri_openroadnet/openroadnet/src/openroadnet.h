#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <cstring>
#include <string>
#include <ros/ros.h>
#include <ros/package.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include "std_msgs/Header.h"
#include <time.h>

#include "DistanceEstimation.h"

// ROS msgs
#include <msgs/FreeSpaceResult.h>
#include <msgs/FreeSpace.h>
#include <msgs/Boundary.h>

///Opencv
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// C API
#include "tf_utils.hpp"
#include "npp_wrapper.h"

// #include <jsoncpp/json/json.h>

class openroadnet
{
    public:
        openroadnet();
        ~openroadnet();

        void init(std::string pb_path);
        msgs::FreeSpaceResult run(const Npp8u* rawCUDA);
        msgs::FreeSpaceResult run(const cv::Mat mat60);
        std::vector<msgs::PointXYZ> calBoundary(std::vector<uint8_t> res,  std::vector<std::vector <int> > &dis_free, int type);
        void release();
        void display_result(cv::Mat &video_ptr, std::vector<std::vector <int>> dis_free_);
        // int read_distance_from_json(const std::string& filename, cv::Point3d** dist_in_cm_, const int rows, const int cols);
        int read_distance_from_json(const std::string& filename, int arr_x[], int arr_y[], int arr_z[], const int rows, const int cols);
        std::vector<std::vector <int>> dis_free_;

    private:
        TF_Output input_op;
        TF_Output out_op;
        TF_Status *status;
        TF_Session *sess;

        clock_t t1, t2;
        
        Npp8u* rawCUDA;
        
        int count = 0;

        std::vector<msgs::PointXYZ> OpenRoadNet_output;
        int input_w = 1920;
        int input_h = 1208;

        int infer_w = 769;
        int infer_h = 769;

        int cam_ = 2;
        
        int arr_x_[1208 * 1920]; 
        int arr_y_[1208 * 1920]; 
        int arr_z_[1208 * 1920]; 

        msgs::FreeSpaceResult OpenRoadNet_output_pub;
        msgs::FreeSpace OpenRoadNet_Free;
        msgs::Boundary OpenRoadNet_Bound;
};

