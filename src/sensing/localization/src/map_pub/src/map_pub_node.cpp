/*
 *   File: map_pub_node.cpp
 *   Created on: March , 2018
 *   Author: Xu, Bo Chun
 *	 Institute: ITRI ICL U300
 */
#define SAVEMAP 0
#define TRANSFORMMAP 0

#include <ros/ros.h>
#include <iostream>
#include <vector>

#include <fstream>
#include <chrono>
#include <pcl/io/pcd_io.h>
#include <pcl/common/transforms.h>
#include <ros/package.h>
//#include <pcl_conversions/pcl_conversions.h>
#include "pcl_conversions.h"

#include <pcl/filters/voxel_grid.h>
#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>

#include <sensor_msgs/PointCloud2.h>


void parseColumns(const std::string &line, std::vector<std::string> *columns)
{
        std::istringstream ss(line);
        std::string column;
        while (std::getline(ss, column, ','))
        {
                columns->push_back(column);
        }
}

ros::Publisher totalMapPointCloudPublisher;
ros::Publisher northMapPointCloudPublisher;
ros::Publisher southMapPointCloudPublisher;


static double voxel_leaf_size = 0.9;
static double voxel_leaf_size_1 = 1.0;
static double voxel_leaf_size_2 = 1.1;

static std::string crop_cord;
static double crop_value_min, crop_value_max, crop_value_mean, crop_value_range;

int seq_ = 0;

int
main (int argc, char** argv)
{
        crop_cord = "x";
        crop_value_min = -300;
        crop_value_max = 700;
        crop_value_mean = (crop_value_min + crop_value_max)/2;
        crop_value_range = 50;

        ros::init(argc, argv, "map_pub");
        ros::NodeHandle nodeHandle;
        std::string path_map_pub_node = ros::package::getPath("map_pub");
        totalMapPointCloudPublisher = nodeHandle.advertise<sensor_msgs::PointCloud2>("points_map", 1, true);
        northMapPointCloudPublisher = nodeHandle.advertise<sensor_msgs::PointCloud2>("points_map_north", 1, true);
        southMapPointCloudPublisher = nodeHandle.advertise<sensor_msgs::PointCloud2>("points_map_south", 1, true);

        sensor_msgs::PointCloud2 total_map_ptcloud;
        sensor_msgs::PointCloud2 total_map_ptcloud_north;
        sensor_msgs::PointCloud2 total_map_ptcloud_south;

        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZI>);
        pcl::PointCloud<pcl::PointXYZI>::Ptr filtered_cloud_ptr(new pcl::PointCloud<pcl::PointXYZI>());
        pcl::PointCloud<pcl::PointXYZI>::Ptr filtered_cloud_ptr_1(new pcl::PointCloud<pcl::PointXYZI>());
        pcl::PointCloud<pcl::PointXYZI>::Ptr filtered_cloud_ptr_2(new pcl::PointCloud<pcl::PointXYZI>());
        pcl::PointCloud<pcl::PointXYZI>::Ptr filtered_cloud_total(new pcl::PointCloud<pcl::PointXYZI>());
        pcl::PointCloud<pcl::PointXYZI>::Ptr filtered_cloud_total_north(new pcl::PointCloud<pcl::PointXYZI>());
        pcl::PointCloud<pcl::PointXYZI>::Ptr filtered_cloud_total_south(new pcl::PointCloud<pcl::PointXYZI>());

        pcl::PassThrough<pcl::PointXYZI> pass;


        std::stringstream ss;
        for (int i = 0; i< 1; i++)
        {
                ss << i;
                pcl::PointCloud<pcl::PointXYZI>::Ptr tmp_cloud (new pcl::PointCloud<pcl::PointXYZI>);

                // if (pcl::io::loadPCDFile<pcl::PointXYZI> (local_lidmap_path + ss.str()+ ".pcd", *tmp_cloud) == -1) //* load the file
                if (pcl::io::loadPCDFile<pcl::PointXYZI> (path_map_pub_node + "/done_map/total_map.pcd", *tmp_cloud) == -1) //* load the file
                {
                        PCL_ERROR ("Couldn't read file .pcd file \n");
                        return (-1);
                }
                else
                {
#if TRANSFORMMAP
                        Eigen::Matrix4f init_guess_;
                        init_guess_ << 1.0000000,  0.0000000,  0.0000000, 0,
                        0.0000000,  0.0000000,  -1.0000000, 0,
                        0.0000000,  1.0000000,  0.0000000, 0,
                        0,0,0,1;
                        pcl::transformPointCloud(*tmp_cloud, *cloud, init_guess_);
#else
                        *cloud += *tmp_cloud;
#endif



#if SAVEMAP
                        if(pcl::io::savePCDFileBinary("total_map_transformed.pcd", *cloud) == -1) {
                                std::cout << "Failed saving " << "total_map.pcd" << "." << std::endl;
                        }
                        std::cout << "Saved " << "total_map_transformed.pcd" << " (" << cloud->size() << " points)" << std::endl;

#endif
                }
                ss.str ("");
        }

        std::cout << "Loaded "
                  << cloud->width * cloud->height
                  << " data points from .pcd "
                  << std::endl;
        pcl::VoxelGrid<pcl::PointXYZI> voxel_grid_filter;
        voxel_grid_filter.setLeafSize(voxel_leaf_size, voxel_leaf_size, voxel_leaf_size);
        voxel_grid_filter.setInputCloud(cloud);
        voxel_grid_filter.filter(*filtered_cloud_ptr);
        std::cout << "After Filtered is :  "
                  << filtered_cloud_ptr->width * filtered_cloud_ptr->height
                  << " data points"
                  << std::endl;

        voxel_grid_filter.setLeafSize(voxel_leaf_size_1, voxel_leaf_size_1, voxel_leaf_size_1);
        voxel_grid_filter.setInputCloud(cloud);
        voxel_grid_filter.filter(*filtered_cloud_ptr_1);
        std::cout << "filtered_cloud_ptr_1 After Filtered is :  "
                  << filtered_cloud_ptr_1->width * filtered_cloud_ptr_1->height
                  << " data points"
                  << std::endl;

        voxel_grid_filter.setLeafSize(voxel_leaf_size_2, voxel_leaf_size_2, voxel_leaf_size_2);
        voxel_grid_filter.setInputCloud(cloud);
        voxel_grid_filter.filter(*filtered_cloud_ptr_2);
        std::cout << "filtered_cloud_ptr_2 After Filtered is :  "
                  << filtered_cloud_ptr_2->width * filtered_cloud_ptr_2->height
                  << " data points"
                  << std::endl;

        *filtered_cloud_total += *filtered_cloud_ptr;
        *filtered_cloud_total += *filtered_cloud_ptr_1;
        *filtered_cloud_total += *filtered_cloud_ptr_2;

        std::cout << "filtered_cloud_total is :  "
                  << filtered_cloud_total->width * filtered_cloud_total->height
                  << " data points"
                  << std::endl;

        pcl::toROSMsg(*filtered_cloud_total, total_map_ptcloud);
        //total_map_ptcloud.header.stamp = ros::Time::now();
        total_map_ptcloud.header.seq = ++seq_;
        total_map_ptcloud.header.frame_id = "/map";

        totalMapPointCloudPublisher.publish(total_map_ptcloud);
        std::cout << "publish total_map:  "
                  << total_map_ptcloud.height*total_map_ptcloud.width
                  << " data points"
                  << std::endl;

        pass.setInputCloud(filtered_cloud_total);
        pass.setFilterFieldName(crop_cord);
        pass.setFilterLimits(crop_value_mean - crop_value_range, crop_value_max);
        pass.filter(*filtered_cloud_total_north);
        pcl::toROSMsg(*filtered_cloud_total_north, total_map_ptcloud_north);

        //total_map_ptcloud_north.header.stamp = ros::Time::now();
        total_map_ptcloud_north.header.seq = seq_;
        total_map_ptcloud_north.header.frame_id = "/map";
        northMapPointCloudPublisher.publish(total_map_ptcloud_north);
        std::cout << "publish total_map_north:  "
                  << total_map_ptcloud_north.height*total_map_ptcloud_north.width
                  << " data points"
                  << std::endl;


        pass.setInputCloud(filtered_cloud_total);
        pass.setFilterFieldName(crop_cord);
        pass.setFilterLimits(crop_value_min, crop_value_mean + crop_value_range);
        pass.filter(*filtered_cloud_total_south);
        pcl::toROSMsg(*filtered_cloud_total_south, total_map_ptcloud_south);

        //total_map_ptcloud_south.header.stamp = ros::Time::now();
        total_map_ptcloud_south.header.seq = seq_;
        total_map_ptcloud_south.header.frame_id = "/map";
        southMapPointCloudPublisher.publish(total_map_ptcloud_south);
        std::cout << "publish total_map_south:  "
                  << total_map_ptcloud_south.height*total_map_ptcloud_south.width
                  << " data points"
                  << std::endl;

        ros::spin();
        return 0;
}
