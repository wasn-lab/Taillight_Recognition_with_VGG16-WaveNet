/*
   this program with Normal Distributions Transform
   thanks to
   Yuki KITSUKAWA
 */

/*
 *   File: localization_main.cpp
 *   Created on: March , 2018
 *	 Institute: ITRI ICL U300
 */

#define LOG 0
#define LOG_ALIGNMENT 0
#define CUDA 1
#define SBG 0
#define TRIMBLE 1
#define MAPPING 0

#include "localization_can_class.h"
#include "pcl_conversions.h"
#include "localization/ErrorCode.h"
#include "localization/LocalizationToVeh.h"
#include "localization/VehInfo.h"


#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <pthread.h>
#include <sstream>
#include <string>
#include <std_msgs/Float64.h>

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <std_msgs/Bool.h>
#include <std_msgs/Float32.h>
#include <std_msgs/String.h>
#include <tf/tf.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_listener.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <geometry_msgs/TwistStamped.h>

#include <pcl/common/angles.h>
#include <pcl/console/time.h>
#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/ndt.h>
#include <pcl_conversions/pcl_conversions.h>

#ifdef MAPPING
#include <dynamic_reconfigure/server.h>
#include <localization/localizationConfig.h>
#endif

#if CUDA
#include <cuda_downsample/cuda_downsample_class.h>
#include <ndt_gpu/NormalDistributionsTransform.h>
#endif

#define PREDICT_POSE_THRESHOLD 0.5

#define Wa 0.4
#define Wb 0.3
#define Wc 0.3

#if LOG
std::ofstream myfile("log.csv");
int cnt_log = 0;
#endif

struct pose
{
        double x;
        double y;
        double z;
        double roll;
        double pitch;
        double yaw;
};

enum init_status {
        NOT_INITIALIZED,
        XY_FOUND,
        Z_FOUND,
        HEADING_FOUND,
        INITIALIZED
};

static pose initial_pose, predict_pose, previous_pose, ndt_pose, current_pose,
            localizer_pose, current_pose_sbg, sbg_raw_pose,
            sbg_local_pose, sbg_vm_pose, current_pose_2vm, initial_pose_key, current_gnss2local_pose, initial_input_pose, rviz_input_pose;
#if MAPPING
static pose added_pose;
#endif

static double offset_x, offset_y, offset_z, offset_yaw; // current_pos - previous_pose
static double offset_sbg_x, offset_sbg_y, offset_sbg_z, offset_sbg_roll, offset_sbg_pitch, offset_sbg_yaw;
static pcl::PointCloud<pcl::PointXYZI> lidar_map_north, lidar_map_south;

// If the map is loaded, map_loaded will be true.
static bool map_loaded = false;
static bool north_map_loaded = false;
static bool south_map_loaded = false;

static bool is_pose_init = 0;
static int init_key_pose_flag = 0;
static double map_mean_value = 0;
#if CUDA
static std::shared_ptr<gpu::GNormalDistributionsTransform> north_gpu_ndt_ptr = std::make_shared<gpu::GNormalDistributionsTransform>();
static std::shared_ptr<gpu::GNormalDistributionsTransform> south_gpu_ndt_ptr = std::make_shared<gpu::GNormalDistributionsTransform>();
#endif

static pcl::VoxelGrid<pcl::PointXYZI> voxel_grid_filter;
pcl::PointCloud<pcl::PointXYZI>::Ptr LidFrontTop_cloudPtr(new pcl::PointCloud<pcl::PointXYZI>);
pcl::PointCloud<pcl::PointXYZI> LidFrontTop;

#if MAPPING
pcl::PointCloud<pcl::PointXYZI>::Ptr mapping_pointsCloudPtr (new pcl::PointCloud<pcl::PointXYZI>);
static std::shared_ptr<gpu::GNormalDistributionsTransform> north_gpu_ndt_mapping_ptr = std::make_shared<gpu::GNormalDistributionsTransform>();
static std::shared_ptr<gpu::GNormalDistributionsTransform> south_gpu_ndt_mapping_ptr = std::make_shared<gpu::GNormalDistributionsTransform>();
static pcl::PointCloud<pcl::PointXYZI> new_sub_map;
static int go_into_mapping_cnt = 0;
static bool save_pcd = false;
static std::string pcd_name = "sub_map.pcd";
#endif

// Default values localization mapping
static int max_iter = 30;      // Maximum iterations 30 30
static float ndt_res = 1.4;     // Resolution MAPPING 1.0 2.0
static double step_size = 0.1;  // Step size 0.1  0.2
static double trans_eps = 0.01; // Transformation epsilon  0.01 0.02

static ros::Publisher ndt_pose_pub;
static geometry_msgs::PoseStamped ndt_pose_msg;

static ros::Publisher current_pose_pub;
static geometry_msgs::PoseStamped current_pose_msg;

static ros::Publisher current_pose_2vm_pub;
static geometry_msgs::PoseStamped current_pose_2vm_msg;

static ros::Publisher sbg_raw_pose_pub;
static geometry_msgs::PoseStamped sbg_raw_pose_msg;

static ros::Publisher sbg_vm_pose_pub;
static geometry_msgs::PoseStamped sbg_vm_pose_msg;

static ros::Publisher sbg_local_pose_pub;
static geometry_msgs::PoseStamped sbg_local_pose_msg;

static ros::Publisher localizer_pose_pub;
static geometry_msgs::PoseStamped localizer_pose_msg;

static ros::Publisher predict_pose_pub;
static geometry_msgs::PoseStamped predict_pose_msg;

static ros::Publisher estimate_twist_pub;
static geometry_msgs::TwistStamped estimate_twist_msg;

static ros::Publisher localization_to_veh_pub;
static localization::LocalizationToVeh localization_to_veh_pub_msg;

static std_msgs::Float32 ndt_reliability;
static std_msgs::Float32 time_ndt_matching;

static ros::Time current_scan_time;
static ros::Time previous_scan_time;
static ros::Duration scan_duration;

static double exe_time = 0.0;
static bool has_converged;
static int iteration = 0;
static double fitness_score = 0.0;
static double trans_probability = 0.0; // Get the registration alignment probability.


static double diff = 0.0;
static double diff_x = 0.0, diff_y = 0.0, diff_z = 0.0, diff_yaw = 0.0;

static double current_velocity = 0.0, previous_velocity = 0.0, previous_previous_velocity = 0.0; // [m/s]
static double current_velocity_x = 0.0, previous_velocity_x = 0.0;
static double current_velocity_y = 0.0, previous_velocity_y = 0.0;
static double current_velocity_z = 0.0, previous_velocity_z = 0.0;
static double current_velocity_smooth = 0.0;

static double current_accel = 0.0, previous_accel = 0.0; // [m/s^2]
static double current_accel_x = 0.0;
static double current_accel_y = 0.0;
static double current_accel_z = 0.0;
static double angular_velocity = 0.0;

static int use_predict_pose = 0;


static double predict_pose_error = 0.0;

static double tf_x_, tf_y_, tf_z_, tf_roll_, tf_pitch_, tf_yaw_;
static Eigen::Matrix4f tf_btol;

static std::string _offset = "linear"; // linear, zero, quadratic

static ros::Publisher pointCloudPublisher;
static sensor_msgs::PointCloud2 current_points;

static ros::Publisher subMapPointCloudPublisher;
static sensor_msgs::PointCloud2 subMapPointCloud;

static bool use_ndt_gpu_ = true;
static bool use_gps_ = false;
static bool use_trimble_ = true;
static bool use_rviz_ = false;
static bool got_rviz_pose = false;

static bool use_local_transform_ = false;
static bool is_gps_new = false;
static bool is_trimble_new = false;
static bool is_got_z_ = false;
static bool is_heading_stable_ = true;

static tf::StampedTransform local_transform;

static double find_z_range = 10.0;
static double traveling_distance = 0.0;

static int points_map_num = 0;
static double voxel_leaf_size = 0.6;
static localization::VehInfo sbg;

static float ego_speed = 0;
static init_status init_status;
Eigen::Matrix4f vector_to_map;

static std::chrono::time_point<std::chrono::system_clock> matching_start, matching_end;


#if LOG_ALIGNMENT
static double log_distance = 1623;
pcl::PointCloud<pcl::PointXYZI>::Ptr log_gnss_raw_cloudPtr(new pcl::PointCloud<pcl::PointXYZI>);
pcl::PointCloud<pcl::PointXYZI>::Ptr log_slam_cloudPtr(new pcl::PointCloud<pcl::PointXYZI>);

pcl::PointXYZI log_gnss_raw_point;
pcl::PointXYZI log_slam_point;
#endif

#if CUDA
static CudaDownSample CDS;
#endif

#if MAPPING
void cfg_callback(localization::localizationConfig &config, uint32_t level) {


        if(save_pcd)
        {
                std::cout << "SAVING PCD .........................." << std::endl;
                pcl::PointCloud<pcl::PointXYZI>::Ptr save_pcd_mapPtr(new pcl::PointCloud<pcl::PointXYZI>);
                *save_pcd_mapPtr += new_sub_map;

                if (save_pcd_mapPtr->size() > 2)
                {
                        if(pcl::io::savePCDFileBinary(pcd_name, *save_pcd_mapPtr) == -1) {
                                std::cout << "Failed saving " << "total_map.pcd" << "." << std::endl;
                        }
                        std::cout << "Saved " << "sub_map.pcd" << " (" << save_pcd_mapPtr->size() << " points)" << std::endl;
                        save_pcd = false;
                }
                else
                {
                        std::cout << "no points to savePCD" << std::endl;
                        save_pcd = false;
                }
        }
        save_pcd = config.save_pcd;
        pcd_name = config.pcd_name;
        std::cout << "CFG CALL BACK 1. save_pcd "  << save_pcd << " 2. name " << pcd_name << std::endl;

}
#endif

pthread_mutex_t mutex;

bool inRange(float low, float high, float input)
{
        return ((input-high)*(input-low) <= 0);
}

bool heading_rate(float low, float high, float input)
{
        return ((input-high)*(input-low) <= 0);
}

float find_z(const pcl::PointCloud<pcl::PointXYZI> &input, const pcl::PointXYZI center_p, double max_range)
{
        std::cout << " finding z value ... " << std::endl;

        pcl::PointCloud<pcl::PointXYZI> ranged_scan;
        pcl::PointXYZI p;
        float min_value = 999;
        int cnt_ = 0;
        double square_max_range = max_range * max_range;
        double diff_;

        for(int i = 0; i < input.points.size(); i++)
        {
                p.x = input.points[i].x;
                p.y = input.points[i].y;
                p.z = input.points[i].z;
                p.intensity = 200;
                diff_ = (p.x-center_p.x)*(p.x-center_p.x) + (p.y-center_p.y)*(p.y-center_p.y);
                if(diff_ <= square_max_range)
                {
                        ranged_scan.points.push_back(p);
                        cnt_++;
                }

        }

        if (ranged_scan.points.size()< 2)
        {
                std::cout << " Poor info of Z ... " << std::endl;
                return min_value;
        }
        else
        {
                for(int i = 0; i < ranged_scan.points.size(); i++)
                {
                        if (ranged_scan.points[i].z < min_value)
                        {
                                min_value = ranged_scan.points[i].z;
                        }
                }
                std::cout << " find local Z = " << min_value << std::endl;
                is_got_z_ = true;
        }

        return min_value;
}

void transformPoint(const struct pose &input_pose, struct pose &pose_output, const Eigen::Matrix4f &transform)
{
        Eigen::Matrix<float, 3, 1> pt(input_pose.x, input_pose.y, input_pose.z);
        pose_output.x = static_cast<float>(transform(0, 0) * pt.coeffRef(0) + transform(0, 1) * pt.coeffRef(1) + transform(0, 2) * pt.coeffRef(2) + transform(0, 3));
        pose_output.y = static_cast<float>(transform(1, 0) * pt.coeffRef(0) + transform(1, 1) * pt.coeffRef(1) + transform(1, 2) * pt.coeffRef(2) + transform(1, 3));
        pose_output.z = static_cast<float>( transform(2, 0) * pt.coeffRef(0) + transform(2, 1) * pt.coeffRef(1) + transform(2, 2) * pt.coeffRef(2) + transform(2, 3));
}


static void initializePose(const struct pose &init_pose)
{
        current_velocity = 0;
        current_velocity_x = 0;
        current_velocity_y = 0;
        current_velocity_z = 0;
        angular_velocity = 0;

        transformPoint(init_pose, sbg_local_pose, vector_to_map);
        sbg_local_pose.yaw = init_pose.yaw;

        initial_pose.x = init_pose.x;
        initial_pose.y = init_pose.y;
        initial_pose.z = init_pose.z;
        initial_pose.roll = init_pose.roll;
        initial_pose.pitch = init_pose.pitch;
        initial_pose.yaw = init_pose.yaw;

        // Setting position and posture for the first time.
        localizer_pose.x = initial_pose.x;
        localizer_pose.y = initial_pose.y;
        localizer_pose.z = initial_pose.z;
        localizer_pose.roll = initial_pose.roll;
        localizer_pose.pitch = initial_pose.pitch;
        localizer_pose.yaw = initial_pose.yaw;

        previous_pose.x = initial_pose.x;
        previous_pose.y = initial_pose.y;
        previous_pose.z = initial_pose.z;
        previous_pose.roll = initial_pose.roll;
        previous_pose.pitch = initial_pose.pitch;
        previous_pose.yaw = initial_pose.yaw;

        current_pose.x = initial_pose.x;
        current_pose.y = initial_pose.y;
        current_pose.z = initial_pose.z;
        current_pose.roll = initial_pose.roll;
        current_pose.pitch = initial_pose.pitch;
        current_pose.yaw = initial_pose.yaw;

        offset_x = 0.0;
        offset_y = 0.0;
        offset_z = 0.0;
        offset_yaw = 0.0;

        std::cout << " initial pose loaded " << std::endl;
        std::cout << " initial_pose.x : " << initial_pose.x << std::endl;
        std::cout << " initial_pose.y : " << initial_pose.y << std::endl;
        std::cout << " initial_pose.z : " << initial_pose.z << std::endl;
        std::cout << " initial_pose.roll : " << initial_pose.roll << std::endl;
        std::cout << " initial_pose.pitch : " << initial_pose.pitch << std::endl;
        std::cout << " initial_pose.yaw : " << initial_pose.yaw << std::endl;
}

void map_mean_value_callback(const std_msgs::Float64 &initial_input) {
        map_mean_value = static_cast<double>(initial_input.data);
        std::cout << " map_mean_value : " << map_mean_value << std::endl;

}

void rviz_initialpose_callback(const geometry_msgs::PoseWithCovarianceStamped::ConstPtr& initial_input) {


        rviz_input_pose.x = initial_input->pose.pose.position.x;
        rviz_input_pose.y = initial_input->pose.pose.position.y;

        pcl::PointXYZI p_;
        p_.x =rviz_input_pose.x;
        p_.y =rviz_input_pose.y;
        if (p_.x < map_mean_value)
        {
                p_.z = 0;
                p_.intensity = 1;
                p_.z = find_z(lidar_map_south, p_, find_z_range);
        }
        else if (p_.x > map_mean_value)
        {
                p_.z = 0;
                p_.intensity = 1;
                p_.z = find_z(lidar_map_north, p_, find_z_range);
        }
        rviz_input_pose.z = p_.z + 2.7;   //lidar height

        // rviz_input_pose.z = 0;
        rviz_input_pose.roll = 0;
        rviz_input_pose.pitch = 0;
        rviz_input_pose.yaw = tf::getYaw(initial_input->pose.pose.orientation);

        std::cout << "pose init by rviz " << std::endl;
        std::cout << "got a new start x:" << rviz_input_pose.x << " y:" << rviz_input_pose.y << " z:" << rviz_input_pose.z << " yaw:" << rviz_input_pose.yaw << " from rviz." << std::endl;
        got_rviz_pose = true;



}

static void north_map_callback(const sensor_msgs::PointCloud2::ConstPtr &input)
{
        std::cout << " north points map callback " << std::endl;

        if (points_map_num != input->width)
        {
                std::cout << " north points map loading ... " << std::endl;

                points_map_num = input->width;

                pcl::fromROSMsg(*input, lidar_map_north);

                pcl::PointCloud<pcl::PointXYZI>::Ptr map_ptr( new pcl::PointCloud<pcl::PointXYZI>(lidar_map_north));
                std::cout << " north_map size  " << map_ptr->size() << std::endl;

                if (use_ndt_gpu_ == true) {
                        std::shared_ptr<gpu::GNormalDistributionsTransform> north_gpu_ndt_ptr_ = std::make_shared<gpu::GNormalDistributionsTransform>();
                        north_gpu_ndt_ptr_->setResolution(ndt_res);
                        north_gpu_ndt_ptr_->setStepSize(step_size);
                        north_gpu_ndt_ptr_->setTransformationEpsilon(trans_eps);
                        north_gpu_ndt_ptr_->setMaximumIterations(max_iter);
                        pcl::PointCloud<pcl::PointXYZI>::Ptr dummy_scan_ptr( new pcl::PointCloud<pcl::PointXYZI>());
                        pcl::PointXYZI dummy_point;
                        dummy_scan_ptr->push_back(dummy_point);

                        north_gpu_ndt_ptr_->setInputTarget(map_ptr);

                        north_gpu_ndt_ptr_->setInputSource(dummy_scan_ptr);

                        north_gpu_ndt_ptr_->align(Eigen::Matrix4f::Identity());

                        pthread_mutex_lock(&mutex);
                        north_gpu_ndt_ptr = north_gpu_ndt_ptr_;
                        pthread_mutex_unlock(&mutex);

#if MAPPING
                        north_gpu_ndt_mapping_ptr = north_gpu_ndt_ptr_;
#endif
                }

                north_map_loaded = true;
                std::cout << " north_map loaded " << std::endl;

        }
}

static void
south_map_callback(const sensor_msgs::PointCloud2::ConstPtr &input)
{
        std::cout << " south points map callback " << std::endl;

        if (points_map_num != input->width)
        {
                std::cout << " south points map loading ... " << std::endl;

                points_map_num = input->width;

                pcl::fromROSMsg(*input, lidar_map_south);

                pcl::PointCloud<pcl::PointXYZI>::Ptr map_ptr(new pcl::PointCloud<pcl::PointXYZI>(lidar_map_south));
                std::cout << " south_map_ptr size  " << map_ptr->size() << std::endl;

                if (use_ndt_gpu_ == true) {
                        std::shared_ptr<gpu::GNormalDistributionsTransform> south_gpu_ndt_ptr_ = std::make_shared<gpu::GNormalDistributionsTransform>();
                        south_gpu_ndt_ptr_->setResolution(ndt_res);
                        south_gpu_ndt_ptr_->setStepSize(step_size);
                        south_gpu_ndt_ptr_->setTransformationEpsilon(trans_eps);
                        south_gpu_ndt_ptr_->setMaximumIterations(max_iter);
                        pcl::PointCloud<pcl::PointXYZI>::Ptr dummy_scan_ptr(new pcl::PointCloud<pcl::PointXYZI>());
                        pcl::PointXYZI dummy_point;
                        dummy_scan_ptr->push_back(dummy_point);
                        south_gpu_ndt_ptr_->setInputTarget(map_ptr);

                        south_gpu_ndt_ptr_->setInputSource(dummy_scan_ptr);
                        south_gpu_ndt_ptr_->align(Eigen::Matrix4f::Identity());
                        pthread_mutex_lock(&mutex);
                        south_gpu_ndt_ptr = south_gpu_ndt_ptr_;
                        pthread_mutex_unlock(&mutex);
#if MAPPING
                        south_gpu_ndt_mapping_ptr = south_gpu_ndt_ptr_;
#endif
                }

                south_map_loaded = true;
                std::cout << " south_map is loaded " << std::endl;
        }
}
#if SBG
static void sbg_callback(const localization::VehInfo::ConstPtr &input)
{
        static double previous_sbg_x = input->ego_x;
        static double previous_sbg_y = input->ego_y;
        static double previous_sbg_z = input->ego_z;
        static double previous_sbg_heading = pcl::deg2rad(input->ego_heading);
        static double previous_sbg_speed = input->ego_speed;
        ego_speed = input->ego_speed;
        static double previous_offset_sbg_x = 0;
        static double previous_offset_sbg_y = 0;
        static double previous_offset_sbg_z = 0;
        static double previous_offset_sbg_heading = pcl::deg2rad(sbg.ego_heading);
        static double previous_offset_sbg_speed = sbg.ego_speed;

        sbg_raw_pose.x = input->ego_x;
        sbg_raw_pose.y = input->ego_y;
        sbg_raw_pose.z = input->ego_z;
        sbg_raw_pose.roll = 0;
        sbg_raw_pose.pitch = 0;
        sbg_raw_pose.yaw = pcl::deg2rad(input->ego_heading);

        sbg_vm_pose.x = input->ego_x;
        sbg_vm_pose.y = input->ego_y;
        sbg_vm_pose.z = input->ego_z;
        sbg_vm_pose.roll = 0;
        sbg_vm_pose.pitch = 0;
        sbg_vm_pose.yaw = pcl::deg2rad(input->ego_heading);

        if (input->gps_fault_flag)
        {
                std::cout << "gps_fault_flag " << std::endl;
                return;
        }
        if (!is_pose_init) {
                initializePose(sbg_vm_pose);
                is_pose_init = true;
                std::cout << "pose_init by SBG " << std::endl;

        }
        is_gps_new = true;
}
#endif
#if TRIMBLE
static void gnss2local_callback(const geometry_msgs::PoseStamped::ConstPtr &input)
{


        tf::Quaternion gnss_q(input->pose.orientation.x, input->pose.orientation.y, input->pose.orientation.z, input->pose.orientation.w);
        tf::Matrix3x3 gnss_m(gnss_q);

        pose current_gnss_pose;
        current_gnss_pose.x = input->pose.position.x;
        current_gnss_pose.y = input->pose.position.y;
        current_gnss_pose.z = input->pose.position.z;
        gnss_m.getRPY(current_gnss_pose.roll, current_gnss_pose.pitch, current_gnss_pose.yaw);

        current_gnss_pose.yaw *= -1;
        current_gnss_pose.yaw -= M_PI*0.83;

        static pose previous_gnss_pose = current_gnss_pose;
        current_gnss2local_pose = current_gnss_pose;


        previous_gnss_pose.x = current_gnss_pose.x;
        previous_gnss_pose.y = current_gnss_pose.y;
        previous_gnss_pose.z = current_gnss_pose.z;
        previous_gnss_pose.roll = current_gnss_pose.roll;
        previous_gnss_pose.pitch = current_gnss_pose.pitch;
        previous_gnss_pose.yaw = current_gnss_pose.yaw;

        is_trimble_new = true;

}
#endif

static void
callbackLidFrontTop(const sensor_msgs::PointCloud2::ConstPtr &input)
{

        if (init_key_pose_flag == 1)
        {
                initializePose(initial_pose_key);
                is_pose_init = true;
                init_key_pose_flag = 0;
                std::cout << "init_pose by KEY " << std::endl;

        }

        if (use_rviz_ && got_rviz_pose)
        {
                initializePose(rviz_input_pose);
                is_pose_init = true;
                got_rviz_pose = false;
                std::cout << "init_pose by RVIZ " << std::endl;

        }
#if SBG
        if (!is_pose_init&& use_gps_ && is_gps_new)
        {
                initializePose(sbg_vm_pose);
                is_pose_init = true;
                std::cout << "is_init_pose by SBG " << std::endl;

        }
#endif

#if TRIMBLE

        if (!is_pose_init && use_trimble_ && is_trimble_new  )
        {
                initial_input_pose = current_gnss2local_pose;
                pcl::PointXYZI p_;
                p_.x =current_gnss2local_pose.x;
                p_.y =current_gnss2local_pose.y;
                if (p_.x < map_mean_value)
                {
                        p_.z = 0;
                        p_.intensity = 1;
                        p_.z = find_z(lidar_map_south, p_, find_z_range);
                }
                else if (p_.x > map_mean_value)
                {
                        p_.z = 0;
                        p_.intensity = 1;
                        p_.z = find_z(lidar_map_north, p_, find_z_range);
                }

                if (is_got_z_)
                {
                        initial_input_pose.z = p_.z + 2.7; //lidar height
                        initializePose(initial_input_pose);
                        is_pose_init = true;
                        std::cout << "init_pose by TRIMBLE " << std::endl;
                }
        }
#endif
        if(!is_pose_init)
        {
                ros::Rate ros_rate(1);
                std::cout <<"NO initial input " << std::endl;
                ros_rate.sleep();
                return;
        }


        pcl::console::TicToc tt_;
        if (north_map_loaded && south_map_loaded  && is_pose_init)
        {
                std::cout << "-----------------------------------------------------------------" << std::endl;
                tt_.tic();
                matching_start = std::chrono::system_clock::now();

                static tf::TransformBroadcaster br;
                tf::Transform transform;
                tf::Quaternion predict_q, ndt_q, current_q, localizer_q, sbg_raw_q, sbg_vm_q, sbg_local_q, current_pose_2vm_q;

                current_scan_time = input->header.stamp;
                pcl::fromROSMsg(*input, *LidFrontTop_cloudPtr);

                pcl::PointCloud<pcl::PointXYZI>::Ptr filtered_scan_ptr(new pcl::PointCloud<pcl::PointXYZI>);
                pcl::PointCloud<pcl::PointXYZI>::Ptr transformed_filtered_scan_ptr(new pcl::PointCloud<pcl::PointXYZI>);
                pcl::PointCloud<pcl::PointXYZI>::Ptr mapping_trg_ptr(new pcl::PointCloud<pcl::PointXYZI>);
                pose predict_pose_for_ndt;

#if CUDA
                if (!CDS.downsampling(*LidFrontTop_cloudPtr, voxel_leaf_size))
                {
                        cudaDeviceReset();
                        std::cout << "cudaDownSample.downsampling NOT SUCCESFULL " << std::endl;
                }
                else
                {
                        *filtered_scan_ptr = *LidFrontTop_cloudPtr;
                }
#else
                voxel_grid_filter.setInputCloud(LidFrontTop_cloudPtr);
                voxel_grid_filter.filter(*filtered_scan_ptr);
#endif

                int scan_points_num = filtered_scan_ptr->size();

                Eigen::Matrix4f t(Eigen::Matrix4f::Identity()); // base_link
                Eigen::Matrix4f t2(Eigen::Matrix4f::Identity()); // localizer

                std::chrono::time_point<std::chrono::system_clock> align_start, align_end;
                static double align_time = 0.0;


#if MAPPING
                if (go_into_mapping_cnt > 10)
                {
                        pthread_mutex_lock(&mutex);
                        pcl::PointXYZI p;
                        pcl::PointCloud<pcl::PointXYZI> scan;
                        double r;

                        for (pcl::PointCloud<pcl::PointXYZI>::const_iterator item = filtered_scan_ptr->begin(); item != filtered_scan_ptr->end(); item++)
                        {
                                p.x = (double)item->x;
                                p.y = (double)item->y;
                                p.z = (double)item->z;
                                p.intensity = (double)item->intensity;

                                r = sqrt(pow(p.x, 2.0) + pow(p.y, 2.0));
                                if (5 < r && r < 120)
                                {
                                        scan.push_back(p);
                                }
                        }
                        pcl::PointCloud<pcl::PointXYZI>::Ptr scan_ptr(new pcl::PointCloud<pcl::PointXYZI>(scan));

                        if (use_ndt_gpu_ == true && current_pose.x > map_mean_value)
                        {
                                north_gpu_ndt_mapping_ptr->setInputSource(scan_ptr);

                        }

                        if (use_ndt_gpu_ == true && current_pose.x < map_mean_value)
                        {
                                south_gpu_ndt_mapping_ptr->setInputSource(scan_ptr);

                        }

// Guess the initial gross estimation of the transformation
                        predict_pose.x = previous_pose.x + offset_x;
                        predict_pose.y = previous_pose.y + offset_y;
                        predict_pose.z = previous_pose.z + offset_z;
                        predict_pose.roll = previous_pose.roll;
                        predict_pose.pitch = previous_pose.pitch;
                        predict_pose.yaw = previous_pose.yaw + offset_yaw;

//[]predict_pose_for_ndt

                        predict_pose_for_ndt = predict_pose;

                        Eigen::Translation3f init_translation(predict_pose_for_ndt.x, predict_pose_for_ndt.y, predict_pose_for_ndt.z);
                        Eigen::AngleAxisf init_rotation_x(predict_pose_for_ndt.roll, Eigen::Vector3f::UnitX());
                        Eigen::AngleAxisf init_rotation_y(predict_pose_for_ndt.pitch, Eigen::Vector3f::UnitY());
                        Eigen::AngleAxisf init_rotation_z(predict_pose_for_ndt.yaw, Eigen::Vector3f::UnitZ());
                        Eigen::Matrix4f init_guess = (init_translation * init_rotation_z *init_rotation_y * init_rotation_x) *tf_btol;

                        if (use_ndt_gpu_ == true && current_pose.x > map_mean_value)
                        {
                                align_start = std::chrono::system_clock::now();
                                north_gpu_ndt_mapping_ptr->align(init_guess);
                                align_end = std::chrono::system_clock::now();
                                has_converged = north_gpu_ndt_mapping_ptr->hasConverged();
                                std::cout << "[Time] : North align  " << tt_.toc() << "ms" << std::endl;

                                t = north_gpu_ndt_mapping_ptr->getFinalTransformation();
                                iteration = north_gpu_ndt_mapping_ptr->getFinalNumIteration();
                                // fitness_score = north_gpu_ndt_ptr->getFitnessScore();
                                trans_probability = north_gpu_ndt_mapping_ptr->getTransformationProbability();
                                pcl::transformPointCloud(*scan_ptr, *transformed_filtered_scan_ptr, t);

                        }

                        if (use_ndt_gpu_ == true && current_pose.x < map_mean_value)
                        {
                                align_start = std::chrono::system_clock::now();
                                south_gpu_ndt_mapping_ptr->align(init_guess);
                                align_end = std::chrono::system_clock::now();
                                has_converged = south_gpu_ndt_mapping_ptr->hasConverged();
                                std::cout << "[Time] : South align  " << tt_.toc() << "ms"<< std::endl;

                                t = south_gpu_ndt_mapping_ptr->getFinalTransformation();
                                iteration = south_gpu_ndt_mapping_ptr->getFinalNumIteration();
                                // fitness_score = south_gpu_ndt_ptr->getFitnessScore();
                                trans_probability = south_gpu_ndt_mapping_ptr->getTransformationProbability();

                                pcl::transformPointCloud(*scan_ptr, *transformed_filtered_scan_ptr, t);



                        }

                        align_time = std::chrono::duration_cast<std::chrono::microseconds>(align_end - align_start).count() /1000.0;

                        t2 = t * tf_btol.inverse();

                        if (pointCloudPublisher.getNumSubscribers() != 0) {
                                pcl::toROSMsg(*transformed_filtered_scan_ptr, current_points);
                                current_points.header.stamp = input->header.stamp;
                                current_points.header.seq = input->header.seq;
                                current_points.header.frame_id = "/map";
                                pointCloudPublisher.publish(current_points);
                        }

                        if (subMapPointCloudPublisher.getNumSubscribers() != 0) {
                                pcl::toROSMsg(new_sub_map, subMapPointCloud);
                                subMapPointCloud.header.stamp = input->header.stamp;
                                subMapPointCloud.header.seq = input->header.seq;
                                subMapPointCloud.header.frame_id = "/map";
                                subMapPointCloudPublisher.publish(subMapPointCloud);
                        }


                        pthread_mutex_unlock(&mutex);
                }

                else
                {
#endif
                pthread_mutex_lock(&mutex);
                if (use_ndt_gpu_ == true && current_pose.x > map_mean_value)
                {
                        north_gpu_ndt_ptr->setInputSource(filtered_scan_ptr);

                }

                if (use_ndt_gpu_ == true && current_pose.x < map_mean_value)
                {
                        south_gpu_ndt_ptr->setInputSource(filtered_scan_ptr);

                }

// Guess the initial gross estimation of the transformation
                predict_pose.x = previous_pose.x + offset_x;
                predict_pose.y = previous_pose.y + offset_y;
                predict_pose.z = previous_pose.z + offset_z;
                predict_pose.roll = previous_pose.roll;
                predict_pose.pitch = previous_pose.pitch;
                predict_pose.yaw = previous_pose.yaw + offset_yaw;

//[]predict_pose_for_ndt

                predict_pose_for_ndt = predict_pose;

                Eigen::Translation3f init_translation(predict_pose_for_ndt.x, predict_pose_for_ndt.y, predict_pose_for_ndt.z);
                Eigen::AngleAxisf init_rotation_x(predict_pose_for_ndt.roll, Eigen::Vector3f::UnitX());
                Eigen::AngleAxisf init_rotation_y(predict_pose_for_ndt.pitch, Eigen::Vector3f::UnitY());
                Eigen::AngleAxisf init_rotation_z(predict_pose_for_ndt.yaw, Eigen::Vector3f::UnitZ());
                Eigen::Matrix4f init_guess = (init_translation * init_rotation_z *init_rotation_y * init_rotation_x) *tf_btol;


                if (use_ndt_gpu_ == true && current_pose.x > map_mean_value)
                {
                        align_start = std::chrono::system_clock::now();
                        north_gpu_ndt_ptr->align(init_guess);
                        align_end = std::chrono::system_clock::now();
                        has_converged = north_gpu_ndt_ptr->hasConverged();
                        std::cout << "[Time] : North align  " << tt_.toc() << "ms" << std::endl;

                        t = north_gpu_ndt_ptr->getFinalTransformation();
                        iteration = north_gpu_ndt_ptr->getFinalNumIteration();
                        // fitness_score = north_gpu_ndt_ptr->getFitnessScore();
                        trans_probability = north_gpu_ndt_ptr->getTransformationProbability();
                }

                if (use_ndt_gpu_ == true && current_pose.x < map_mean_value)
                {
                        align_start = std::chrono::system_clock::now();
                        south_gpu_ndt_ptr->align(init_guess);
                        align_end = std::chrono::system_clock::now();
                        has_converged = south_gpu_ndt_ptr->hasConverged();
                        std::cout << "[Time] : South align  " << tt_.toc() << "ms"<< std::endl;

                        t = south_gpu_ndt_ptr->getFinalTransformation();
                        iteration = south_gpu_ndt_ptr->getFinalNumIteration();
                        // fitness_score = south_gpu_ndt_ptr->getFitnessScore();
                        trans_probability = south_gpu_ndt_ptr->getTransformationProbability();
                }

                align_time = std::chrono::duration_cast<std::chrono::microseconds>(align_end - align_start).count() /1000.0;

                t2 = t * tf_btol.inverse();

                pcl::transformPointCloud(*filtered_scan_ptr, *transformed_filtered_scan_ptr, t);

                if (pointCloudPublisher.getNumSubscribers() != 0) {
                        pcl::toROSMsg(*transformed_filtered_scan_ptr, current_points);
                        current_points.header.stamp = input->header.stamp;
                        current_points.header.seq = input->header.seq;
                        current_points.header.frame_id = "/map";
                        pointCloudPublisher.publish(current_points);
                }

                pthread_mutex_unlock(&mutex);


#if MAPPING
        }
        go_into_mapping_cnt++;
        std::cout << "-------go_into_mapping_cnt---------" << go_into_mapping_cnt << "--------------" << std::endl;

#endif



                tf::Matrix3x3 mat_l; // localizer
                mat_l.setValue(static_cast<double>(t(0, 0)), static_cast<double>(t(0, 1)),
                               static_cast<double>(t(0, 2)), static_cast<double>(t(1, 0)),
                               static_cast<double>(t(1, 1)), static_cast<double>(t(1, 2)),
                               static_cast<double>(t(2, 0)), static_cast<double>(t(2, 1)),
                               static_cast<double>(t(2, 2)));

// Update localizer_pose
                localizer_pose.x = t(0, 3);
                localizer_pose.y = t(1, 3);
                localizer_pose.z = t(2, 3);
                mat_l.getRPY(localizer_pose.roll, localizer_pose.pitch, localizer_pose.yaw,1);

                tf::Matrix3x3 mat_b; // base_link
                mat_b.setValue(static_cast<double>(t2(0, 0)), static_cast<double>(t2(0, 1)),
                               static_cast<double>(t2(0, 2)), static_cast<double>(t2(1, 0)),
                               static_cast<double>(t2(1, 1)), static_cast<double>(t2(1, 2)),
                               static_cast<double>(t2(2, 0)), static_cast<double>(t2(2, 1)),
                               static_cast<double>(t2(2, 2)));

// Update ndt_pose
                ndt_pose.x = t2(0, 3);
                ndt_pose.y = t2(1, 3);
                ndt_pose.z = t2(2, 3);
                mat_b.getRPY(ndt_pose.roll, ndt_pose.pitch, ndt_pose.yaw, 1);
// current_pose_2vm = ndt_pose;
                transformPoint(ndt_pose, current_pose_2vm, vector_to_map.inverse());

#if LOG_ALIGNMENT
                log_slam_point.x = ndt_pose.x;
                log_slam_point.y = ndt_pose.y;
                log_slam_point.z = ndt_pose.z;
                log_slam_point.intensity = 50;
                log_slam_cloudPtr->points.push_back(log_slam_point);

                log_gnss_raw_point.x = sbg_raw_pose.x;
                log_gnss_raw_point.y = sbg_raw_pose.y;
                log_gnss_raw_point.z = sbg_raw_pose.z;
                log_gnss_raw_point.intensity = 100;
                log_gnss_raw_cloudPtr->points.push_back(log_gnss_raw_point);
#endif

                current_pose_2vm.roll = 0;
                current_pose_2vm.pitch = 0;
                current_pose_2vm.yaw = ndt_pose.yaw;

                if (current_pose_2vm.yaw > 2 * M_PI)
                {
                        current_pose_2vm.yaw -= 2 * M_PI; // 540+360
                }
                else if (current_pose_2vm.yaw < 0)
                {
                        current_pose_2vm.yaw += 2 * M_PI;
                }

// Calculate the difference between ndt_pose and predict_pose
                predict_pose_error = sqrt((ndt_pose.x - predict_pose_for_ndt.x) *
                                          (ndt_pose.x - predict_pose_for_ndt.x) +
                                          (ndt_pose.y - predict_pose_for_ndt.y) *
                                          (ndt_pose.y - predict_pose_for_ndt.y) +
                                          (ndt_pose.z - predict_pose_for_ndt.z) *
                                          (ndt_pose.z - predict_pose_for_ndt.z));

                if (predict_pose_error <= PREDICT_POSE_THRESHOLD)
                {
                        use_predict_pose = 0;
                }
                else
                {
                        use_predict_pose = 1;
                }
                use_predict_pose = 0;

                if (use_predict_pose == 0) {
                        current_pose.x = ndt_pose.x;
                        current_pose.y = ndt_pose.y;
                        current_pose.z = ndt_pose.z;
                        current_pose.roll = ndt_pose.roll;
                        current_pose.pitch = ndt_pose.pitch;
                        current_pose.yaw = ndt_pose.yaw;
                } else {
                        current_pose.x = predict_pose_for_ndt.x;
                        current_pose.y = predict_pose_for_ndt.y;
                        current_pose.z = predict_pose_for_ndt.z;
                        current_pose.roll = predict_pose_for_ndt.roll;
                        current_pose.pitch = predict_pose_for_ndt.pitch;
                        current_pose.yaw = predict_pose_for_ndt.yaw;
                }



#if MAPPING

                double shift = sqrt(pow(current_pose.x - added_pose.x, 2.0) + pow(current_pose.y - added_pose.y, 2.0));
                std::cout << "--------///shift////"<<  shift << "///shift///-------" << std::endl;

                if (shift >= 2.0)
                {
                        added_pose.x = current_pose.x;
                        added_pose.y = current_pose.y;
                        added_pose.z = current_pose.z;
                        added_pose.roll = current_pose.roll;
                        added_pose.pitch = current_pose.pitch;
                        added_pose.yaw = current_pose.yaw;

                        if (use_ndt_gpu_ == true && current_pose.x > map_mean_value)
                        {
                                // *mapping_trg_ptr += *transformed_filtered_scan_ptr;
                                *mapping_trg_ptr += new_sub_map;
                                *mapping_trg_ptr += lidar_map_north;

                                north_gpu_ndt_mapping_ptr->setInputTarget(mapping_trg_ptr);
                                new_sub_map += *transformed_filtered_scan_ptr;

                        }

                        if (use_ndt_gpu_ == true && current_pose.x < map_mean_value)
                        {
                                // *mapping_trg_ptr += *transformed_filtered_scan_ptr;
                                *mapping_trg_ptr += new_sub_map;
                                *mapping_trg_ptr += lidar_map_south;
                                south_gpu_ndt_mapping_ptr->setInputTarget(mapping_trg_ptr);

                                new_sub_map += *transformed_filtered_scan_ptr;


                        }



                }
#endif
// Compute the velocity and acceleration
                scan_duration = current_scan_time - previous_scan_time;
                double secs = scan_duration.toSec();
                diff_x = current_pose.x - previous_pose.x;
                diff_y = current_pose.y - previous_pose.y;
                diff_z = current_pose.z - previous_pose.z;
                diff_yaw = current_pose.yaw - previous_pose.yaw;
                diff = sqrt(diff_x * diff_x + diff_y * diff_y + diff_z * diff_z);

                current_velocity = diff / secs;
                current_velocity_x = diff_x / secs;
                current_velocity_y = diff_y / secs;
                current_velocity_z = diff_z / secs;
                angular_velocity = diff_yaw / secs;

                current_velocity_smooth =(current_velocity + previous_velocity + previous_previous_velocity) /3.0;
                if (current_velocity_smooth < 0.2)
                {
                        current_velocity_smooth = 0.0;
                }

                current_accel = (current_velocity - previous_velocity) / secs;
                current_accel_x = (current_velocity_x - previous_velocity_x) / secs;
                current_accel_y = (current_velocity_y - previous_velocity_y) / secs;
                current_accel_z = (current_velocity_z - previous_velocity_z) / secs;

                ndt_q.setRPY(ndt_pose.roll, ndt_pose.pitch, ndt_pose.yaw);

                if (ndt_pose_pub.getNumSubscribers() != 0)
                {
                        if (use_local_transform_ == true)
                        {
                                tf::Vector3 v(ndt_pose.x, ndt_pose.y, ndt_pose.z);
                                tf::Transform transform(ndt_q, v);
                                ndt_pose_msg.header.frame_id = "/map";
                                ndt_pose_msg.header.stamp = current_scan_time;
                                ndt_pose_msg.pose.position.x = (local_transform * transform).getOrigin().getX();
                                ndt_pose_msg.pose.position.y = (local_transform * transform).getOrigin().getY();
                                ndt_pose_msg.pose.position.z = (local_transform * transform).getOrigin().getZ();
                                ndt_pose_msg.pose.orientation.x = (local_transform * transform).getRotation().x();
                                ndt_pose_msg.pose.orientation.y = (local_transform * transform).getRotation().y();
                                ndt_pose_msg.pose.orientation.z = (local_transform * transform).getRotation().z();
                                ndt_pose_msg.pose.orientation.w = (local_transform * transform).getRotation().w();
                        } else
                        {
                                ndt_pose_msg.header.frame_id = "/map";
                                ndt_pose_msg.header.stamp = current_scan_time;
                                ndt_pose_msg.pose.position.x = ndt_pose.x;
                                ndt_pose_msg.pose.position.y = ndt_pose.y;
                                ndt_pose_msg.pose.position.z = ndt_pose.z;
                                ndt_pose_msg.pose.orientation.x = ndt_q.x();
                                ndt_pose_msg.pose.orientation.y = ndt_q.y();
                                ndt_pose_msg.pose.orientation.z = ndt_q.z();
                                ndt_pose_msg.pose.orientation.w = ndt_q.w();
                        }
                        ndt_pose_pub.publish(ndt_pose_msg);
                }
// current_pose is published by vel_pose_mux

                current_q.setRPY(current_pose.roll, current_pose.pitch, current_pose.yaw);
                current_pose_msg.header.frame_id = "/map";
                current_pose_msg.header.stamp = current_scan_time;
                current_pose_msg.pose.position.x = current_pose.x;
                current_pose_msg.pose.position.y = current_pose.y;
                current_pose_msg.pose.position.z = current_pose.z;
                current_pose_msg.pose.orientation.x = current_q.x();
                current_pose_msg.pose.orientation.y = current_q.y();
                current_pose_msg.pose.orientation.z = current_q.z();
                current_pose_msg.pose.orientation.w = current_q.w();
                current_pose_pub.publish(current_pose_msg);

                sbg_raw_q.setRPY(sbg_raw_pose.roll, sbg_raw_pose.pitch, sbg_raw_pose.yaw);

                if (sbg_raw_pose_pub.getNumSubscribers() != 0)
                {
                        sbg_raw_pose_msg.header.frame_id = "/map";
                        sbg_raw_pose_msg.header.stamp = current_scan_time;
                        sbg_raw_pose_msg.pose.position.x = sbg_raw_pose.x;
                        sbg_raw_pose_msg.pose.position.y = sbg_raw_pose.y;
                        sbg_raw_pose_msg.pose.position.z = sbg_raw_pose.z;
                        sbg_raw_pose_msg.pose.orientation.x = sbg_raw_q.x();
                        sbg_raw_pose_msg.pose.orientation.y = sbg_raw_q.y();
                        sbg_raw_pose_msg.pose.orientation.z = sbg_raw_q.z();
                        sbg_raw_pose_msg.pose.orientation.w = sbg_raw_q.w();
                        sbg_raw_pose_pub.publish(sbg_raw_pose_msg);
                }
                sbg_vm_q.setRPY(sbg_vm_pose.roll, sbg_vm_pose.pitch, sbg_vm_pose.yaw);

                if (sbg_vm_pose_pub.getNumSubscribers() != 0) {
                        sbg_vm_pose_msg.header.frame_id = "/map";
                        sbg_vm_pose_msg.header.stamp = current_scan_time;
                        sbg_vm_pose_msg.pose.position.x = sbg_vm_pose.x;
                        sbg_vm_pose_msg.pose.position.y = sbg_vm_pose.y;
                        sbg_vm_pose_msg.pose.position.z = sbg_vm_pose.z;
                        sbg_vm_pose_msg.pose.orientation.x = sbg_vm_q.x();
                        sbg_vm_pose_msg.pose.orientation.y = sbg_vm_q.y();
                        sbg_vm_pose_msg.pose.orientation.z = sbg_vm_q.z();
                        sbg_vm_pose_msg.pose.orientation.w = sbg_vm_q.w();
                        sbg_vm_pose_pub.publish(sbg_vm_pose_msg);
                }

                transformPoint(sbg_vm_pose, sbg_local_pose, vector_to_map);

                sbg_local_q.setRPY(sbg_local_pose.roll, sbg_local_pose.pitch, sbg_local_pose.yaw);

                if (sbg_local_pose_pub.getNumSubscribers() != 0) {

                        sbg_local_pose_msg.header.frame_id = "/map";
                        sbg_local_pose_msg.header.stamp = current_scan_time;
                        sbg_local_pose_msg.pose.position.x = sbg_local_pose.x;
                        sbg_local_pose_msg.pose.position.y = sbg_local_pose.y;
                        sbg_local_pose_msg.pose.position.z = sbg_local_pose.z;
                        sbg_local_pose_msg.pose.orientation.x = sbg_local_q.x();
                        sbg_local_pose_msg.pose.orientation.y = sbg_local_q.y();
                        sbg_local_pose_msg.pose.orientation.z = sbg_local_q.z();
                        sbg_local_pose_msg.pose.orientation.w = sbg_local_q.w();
                        sbg_local_pose_pub.publish(sbg_local_pose_msg);
                }

                current_pose_2vm_q.setRPY(current_pose_2vm.roll, current_pose_2vm.pitch, current_pose_2vm.yaw);

                if (current_pose_2vm_pub.getNumSubscribers() != 0) {
                        current_pose_2vm_msg.header.frame_id = "/map";
                        current_pose_2vm_msg.header.stamp = current_scan_time;
                        current_pose_2vm_msg.pose.position.x = current_pose_2vm.x;
                        current_pose_2vm_msg.pose.position.y = current_pose_2vm.y;
                        current_pose_2vm_msg.pose.position.z = current_pose_2vm.z;
                        current_pose_2vm_msg.pose.orientation.x = current_pose_2vm_q.x();
                        current_pose_2vm_msg.pose.orientation.y = current_pose_2vm_q.y();
                        current_pose_2vm_msg.pose.orientation.z = current_pose_2vm_q.z();
                        current_pose_2vm_msg.pose.orientation.w = current_pose_2vm_q.w();
                        current_pose_2vm_pub.publish(current_pose_2vm_msg);
                }

                predict_q.setRPY(predict_pose.roll, predict_pose.pitch, predict_pose.yaw);

                if (predict_pose_pub.getNumSubscribers() != 0) {
                        predict_pose_msg.header.frame_id = "/map";
                        predict_pose_msg.header.stamp = current_scan_time;
                        predict_pose_msg.pose.position.x = predict_pose.x;
                        predict_pose_msg.pose.position.y = predict_pose.y;
                        predict_pose_msg.pose.position.z = predict_pose.z;
                        predict_pose_msg.pose.orientation.x = predict_q.x();
                        predict_pose_msg.pose.orientation.y = predict_q.y();
                        predict_pose_msg.pose.orientation.z = predict_q.z();
                        predict_pose_msg.pose.orientation.w = predict_q.w();
                        predict_pose_pub.publish(predict_pose_msg);

                }

                localizer_q.setRPY(localizer_pose.roll, localizer_pose.pitch,localizer_pose.yaw);
                if (localizer_pose_pub.getNumSubscribers() != 0) {

                        if (use_local_transform_ == true)
                        {
                                tf::Vector3 v(localizer_pose.x, localizer_pose.y, localizer_pose.z);
                                tf::Transform transform(localizer_q, v);
                                localizer_pose_msg.header.frame_id = "/map";
                                localizer_pose_msg.header.stamp = current_scan_time;
                                localizer_pose_msg.pose.position.x = (local_transform * transform).getOrigin().getX();
                                localizer_pose_msg.pose.position.y = (local_transform * transform).getOrigin().getY();
                                localizer_pose_msg.pose.position.z = (local_transform * transform).getOrigin().getZ();
                                localizer_pose_msg.pose.orientation.x = (local_transform * transform).getRotation().x();
                                localizer_pose_msg.pose.orientation.y = (local_transform * transform).getRotation().y();
                                localizer_pose_msg.pose.orientation.z = (local_transform * transform).getRotation().z();
                                localizer_pose_msg.pose.orientation.w = (local_transform * transform).getRotation().w();
                        } else {
                                localizer_pose_msg.header.frame_id = "/map";
                                localizer_pose_msg.header.stamp = current_scan_time;
                                localizer_pose_msg.pose.position.x = localizer_pose.x;
                                localizer_pose_msg.pose.position.y = localizer_pose.y;
                                localizer_pose_msg.pose.position.z = localizer_pose.z;
                                localizer_pose_msg.pose.orientation.x = localizer_q.x();
                                localizer_pose_msg.pose.orientation.y = localizer_q.y();
                                localizer_pose_msg.pose.orientation.z = localizer_q.z();
                                localizer_pose_msg.pose.orientation.w = localizer_q.w();
                        }
                        localizer_pose_pub.publish(localizer_pose_msg);
                }



// Send TF "/base_link" to "/map"
                transform.setOrigin(tf::Vector3(current_pose.x, current_pose.y, current_pose.z));
                transform.setRotation(current_q);

                if (use_local_transform_ == true)
                {
                        br.sendTransform(tf::StampedTransform(local_transform * transform,current_scan_time, "/map","/base_link"));
                } else
                {
                        br.sendTransform(tf::StampedTransform(transform, current_scan_time,"/map", "/base_link"));
                }

                matching_end = std::chrono::system_clock::now();
                exe_time = std::chrono::duration_cast<std::chrono::microseconds>(matching_end - matching_start).count() /1000.0;
                time_ndt_matching.data = exe_time;

// Set values for /estimate_twist
                estimate_twist_msg.header.stamp = current_scan_time;
                estimate_twist_msg.header.frame_id = "/base_link";
                estimate_twist_msg.twist.linear.x = current_velocity;
                estimate_twist_msg.twist.linear.y = 0.0;
                estimate_twist_msg.twist.linear.z = 0.0;
                estimate_twist_msg.twist.angular.x = 0.0;
                estimate_twist_msg.twist.angular.y = 0.0;
                estimate_twist_msg.twist.angular.z = angular_velocity;

                tt_.toc();
/* Compute NDT_Reliability */
                ndt_reliability.data = Wa * (exe_time / 100.0) * 100.0 +
                                       Wb * (iteration / 10.0) * 100.0 +
                                       Wc * ((2.0 - trans_probability) / 2.0) * 100.0;

                std::cout << "(x,y,z,roll,pitch,yaw): " << std::endl;
                std::cout << "(" << current_pose.x << ", " << current_pose.y << ", "
                          << current_pose.z << ", " << current_pose.roll << ", "
                          << current_pose.pitch << ", " << current_pose.yaw << ")"
                          << std::endl;

#if SBG
                std::cout << "(sbg_raw_pose x,y,z,roll,pitch,yaw): " << std::endl;
                std::cout << "(" << sbg_raw_pose.x << ", " << sbg_raw_pose.y << ", "
                          << sbg_raw_pose.z << ", " << sbg_raw_pose.roll << ", "
                          << sbg_raw_pose.pitch << ", " << sbg_raw_pose.yaw << ")"
                          << std::endl;
#endif


#if TRIMBLE

                std::cout << "(current_gnss2local_pose x,y,z,roll,pitch,yaw): " << std::endl;
                std::cout << "(" << current_gnss2local_pose.x << ", " << current_gnss2local_pose.y << ", "
                          << current_gnss2local_pose.z << ", " << current_gnss2local_pose.roll << ", "
                          << current_gnss2local_pose.pitch << ", " << current_gnss2local_pose.yaw << ")"
                          << std::endl;
#endif


                std::cout << "Align time: " << align_time << std::endl;
                std::cout << "[Time] : END  " << tt_.toc() << "ms" << std::endl;
                std::cout
                << "-----------------------------------------------------------------"
                << std::endl;

#if LOG
                myfile << current_pose.x << "," << current_pose.y << "," << current_pose.z << "," << std::endl;
                cnt_log++;
#endif
#if LOG_ALIGNMENT
                if (traveling_distance > log_distance)
                {
                        std::cout << "SAVE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"<< std::endl;
                        std::string pcd_filename = "log_gnss_raw_cloudPtr.pcd";
                        if (pcl::io::savePCDFileBinary(pcd_filename, *log_gnss_raw_cloudPtr) ==-1)
                        {
                                std::cout << "Failed saving log_gnss_raw_cloud" << pcd_filename << "."<< std::endl;
                        }

                        pcd_filename = "log_slam_cloudPtr.pcd";
                        if (pcl::io::savePCDFileBinary(pcd_filename, *log_slam_cloudPtr) == -1)
                        {
                                std::cout << "Failed saving log_slam_cloud" << pcd_filename << "."<< std::endl;
                        }

                        log_distance += log_distance;
                }
#endif
// Update offset
                if (_offset == "linear")
                {
                        offset_x = diff_x;
                        offset_y = diff_y;
                        offset_z = diff_z;
                        offset_yaw = diff_yaw;
                } else if (_offset == "quadratic")
                {
                        offset_x = (current_velocity_x + current_accel_x * secs) * secs;
                        offset_y = (current_velocity_y + current_accel_y * secs) * secs;
                        offset_z = diff_z;
                        offset_yaw = diff_yaw;
                } else if (_offset == "zero")
                {
                        offset_x = 0.0;
                        offset_y = 0.0;
                        offset_z = 0.0;
                        offset_yaw = 0.0;
                }

// Update previous_***
                previous_pose.x = current_pose.x;
                previous_pose.y = current_pose.y;
                previous_pose.z = current_pose.z;
                previous_pose.roll = current_pose.roll;
                previous_pose.pitch = current_pose.pitch;
                previous_pose.yaw = current_pose.yaw;

                previous_scan_time.sec = current_scan_time.sec;
                previous_scan_time.nsec = current_scan_time.nsec;

                previous_previous_velocity = previous_velocity;
                previous_velocity = current_velocity;
                previous_velocity_x = current_velocity_x;
                previous_velocity_y = current_velocity_y;
                previous_velocity_z = current_velocity_z;
                previous_accel = current_accel;

// Set values for /ndt_stat
// localization_info_pub_msg.header.stamp = current_scan_time;
// localization_info_pub_msg.exe_time = time_ndt_matching.data;
// localization_info_pub_msg.iteration = iteration;
// localization_info_pub_msg.score = fitness_score;
// localization_info_pub_msg.velocity = current_velocity;
// localization_info_pub_msg.acceleration = current_accel;
// localization_info_pub_msg.use_predict_pose = 0;
//
// localization_info_pub.publish(localization_info_pub_msg);



        }

        localization_to_veh_pub_msg.x = current_pose_2vm.x;
        localization_to_veh_pub_msg.y = current_pose_2vm.y;
        localization_to_veh_pub_msg.heading = current_pose_2vm.yaw;
        localization_to_veh_pub_msg.fitness_score = fitness_score;
        localization_to_veh_pub_msg.z = current_pose_2vm.z;
        localization_to_veh_pub_msg.ndt_reliability = ndt_reliability.data;
        localization_to_veh_pub.publish(localization_to_veh_pub_msg);

}

void *thread_func(void *args) {
        ros::NodeHandle nh_map;
        ros::CallbackQueue map_callback_queue;
        nh_map.setCallbackQueue(&map_callback_queue);

        ros::Subscriber north_map_sub = nh_map.subscribe("points_map_north", 10, north_map_callback);
        ros::Subscriber south_map_sub = nh_map.subscribe("points_map_south", 10, south_map_callback);

        ros::Rate ros_rate(1);
        while (nh_map.ok())
        {
                map_callback_queue.callAvailable(ros::WallDuration());
                ros_rate.sleep();
        }
        return nullptr;
}

int main(int argc, char **argv) {


        CDS.warmUpGPU();
        ros::init(argc, argv, "localization");
        pthread_mutex_init(&mutex, NULL);

        ros::NodeHandle nh;
        ros::NodeHandle private_nh("~");
        private_nh.getParam("use_rviz_", use_rviz_);
        std::cout << "use_rviz_: " << use_rviz_ << std::endl;

#if MAPPING

        private_nh.getParam("save_pcd", save_pcd);
        private_nh.getParam("pcd_name", pcd_name);
        dynamic_reconfigure::Server<localization::localizationConfig> cfg_server;
        dynamic_reconfigure::Server<localization::localizationConfig>::CallbackType cfg_tmp;
        cfg_tmp = boost::bind(&cfg_callback, _1, _2);
        cfg_server.setCallback(cfg_tmp);
#endif
        init_status = NOT_INITIALIZED;
        tf_x_ = 0.20;
        tf_y_ = 0;
        tf_z_ = 0;

        vector_to_map << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1;

        std::cout<< "-----------------------------------------------------------------"<< std::endl;
        std::cout << "init_key_pose_flag: " << init_key_pose_flag << std::endl;
        std::cout<< "-----------------------------------------------------------------"<< std::endl;

        Eigen::Translation3f tl_btol(tf_x_, tf_y_, tf_z_); // tl: translation
        Eigen::AngleAxisf rot_x_btol(tf_roll_, Eigen::Vector3f::UnitX()); // rot: rotation
        Eigen::AngleAxisf rot_y_btol(tf_pitch_, Eigen::Vector3f::UnitY());
        Eigen::AngleAxisf rot_z_btol(tf_yaw_, Eigen::Vector3f::UnitZ());
        tf_btol = (tl_btol * rot_z_btol * rot_y_btol * rot_x_btol).matrix();

        // Updated in initialpose_callback or gnss_callback
        initial_pose.x = 0.0;
        initial_pose.y = 0.0;
        initial_pose.z = 0.0;
        initial_pose.roll = 0.0;
        initial_pose.pitch = 0.0;
        initial_pose.yaw = 0.0;
#if MAPPING
        added_pose.x = 0.0;
        added_pose.y = 0.0;
        added_pose.z = 0.0;
        added_pose.roll = 0.0;
        added_pose.pitch = 0.0;
        added_pose.yaw = 0.0;
#endif
        // initial_pose_key.x = 574.30;
        // initial_pose_key.y = -153.03;
        // initial_pose_key.z = -23.45;
        // initial_pose_key.roll = -0.0237587304412;
        // initial_pose_key.pitch = 0.00238022293266;
        // initial_pose_key.yaw = 3.01196590467;

        initial_pose_key.x = 460.222;
        initial_pose_key.y = -254.724655151;
        initial_pose_key.z = -23.6271781921;
        initial_pose_key.roll = 0.0196261;
        initial_pose_key.pitch = 0.0681919;
        initial_pose_key.yaw = -1.6733683;

        voxel_grid_filter.setLeafSize(voxel_leaf_size, voxel_leaf_size, voxel_leaf_size);

        // Publishers
        ndt_pose_pub = nh.advertise<geometry_msgs::PoseStamped>("/ndt_pose", 1000);
        current_pose_pub =nh.advertise<geometry_msgs::PoseStamped>("/current_pose", 1000);
        current_pose_2vm_pub =nh.advertise<geometry_msgs::PoseStamped>("/current_pose_2vm", 1000);
        localizer_pose_pub =nh.advertise<geometry_msgs::PoseStamped>("/localizer_pose", 1000);
        pointCloudPublisher =nh.advertise<sensor_msgs::PointCloud2>("current_points", 1, true);
        sbg_raw_pose_pub =nh.advertise<geometry_msgs::PoseStamped>("/sbg_raw_pose", 1000);
        sbg_vm_pose_pub =nh.advertise<geometry_msgs::PoseStamped>("/sbg_vm_pose", 1000);
        sbg_local_pose_pub =nh.advertise<geometry_msgs::PoseStamped>("/sbg_local_pose", 1000);
        predict_pose_pub =nh.advertise<geometry_msgs::PoseStamped>("/predict_pose", 1000);

        localization_to_veh_pub = nh.advertise<localization::LocalizationToVeh>("/localization_to_veh", 1000);
#if MAPPING
        subMapPointCloudPublisher =nh.advertise<sensor_msgs::PointCloud2>("sub_map_", 1, true);
#endif

        // Subscribers
        ros::Subscriber LidFrontTopSub =nh.subscribe("LidarFrontTop", 1, callbackLidFrontTop);
        ros::Subscriber subRvizPose = nh.subscribe("/initialpose", 1, rviz_initialpose_callback);
        ros::Subscriber mapMeaValueSub = nh.subscribe("map_mean_value", 1, map_mean_value_callback);

        // ros::Subscriber LidFrontTopSub =nh.subscribe("LidarFrontLeft", 1, callbackLidFrontTop);

#if SBG
        ros::Subscriber sbg_sub = nh.subscribe("veh_info", 10, sbg_callback);
        #endif
#if TRIMBLE
        ros::Subscriber gnss2local_sub = nh.subscribe("gnss2local_data", 10, gnss2local_callback);
        #endif
        pthread_t thread;
        pthread_create(&thread, NULL, thread_func, NULL);
        ros::MultiThreadedSpinner s(3);
        ros::spin(s);

        return 0;
}
