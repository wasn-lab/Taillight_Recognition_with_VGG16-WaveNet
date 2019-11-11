/*
 * bus_vehinfo_receive.h
 *
 * * Created on: Dec 14, 2017
 *       Author: Bo Chun, Xu
 *	 Institute: ITRI ICL U300
 */

#pragma once

// ROS
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>


class Read
{
 public:
  /*!
   * Constructor.
   * @param nodeHandle the ROS node handle.
   */
  Read(ros::NodeHandle& nodeHandle);

  /*!
   * Destructor.
   */
  virtual ~Read();

 private:
  /*!
   * Reads and verifies the ROS parameters.
   * @return true if successful.
   */
  bool readParameters();

  /*!
   * Initializes node.
   */
  void initialize();

  /*!
   * Read the point cloud from a .ply or .vtk file.
   * @param filePath the path to the .ply or .vtk file.
   * @param pointCloudFrameId the id of the frame of the point cloud data.
   * @return true if successful.
   */
  bool readFile(const std::string& filePath, const std::string& pointCloudFrameId);

  /*!
   * Timer callback function.
   * @param timerEvent the timer event.
   */
  void timerCallback(const ros::TimerEvent& timerEvent);

  /*!
   * Publish the point cloud as a PointCloud2.
   * @return true if successful.
   */
  bool publish();

  //! ROS node handle.
  ros::NodeHandle& nodeHandle_;

  //! Point cloud message to publish.
  sensor_msgs::PointCloud2::Ptr pointCloudMessage_;

  //! Point cloud publisher.
  ros::Publisher pointCloudPublisher_;

  //! Timer for publishing the point cloud.
  ros::Timer timer_;

  //! Path to the point cloud file.
  std::string filePath_;

  //! Point cloud topic to be published at.
  std::string pointCloudTopic_;

  //! Point cloud frame id.
  std::string pointCloudFrameId_;

  //! If true, continous publishing is used.
  //! If false, point cloud is only published once.
  bool isContinousPublishing_;

  //! Duration between publishing steps.
  ros::Duration updateDuration_;
};

