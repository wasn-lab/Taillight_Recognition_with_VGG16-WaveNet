#ifndef RAY_GROUND_FILTER_H_
#define RAY_GROUND_FILTER_H_

#include <string>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/extract_indices.h>

using namespace std;
using namespace pcl;

class RayGroundFilter
{
  public:
    RayGroundFilter (float sensor_height,
                     float general_max_slope,
                     float local_max_slope,
                     float radial_divider_angle,
                     float concentric_divider_distance,
                     float min_height_threshold,
                     float clipping_height,
                     float min_point_distance,
                     float reclass_distance_threshold);

    template <typename PointT>
    PointIndices
    compute (typename PointCloud<PointT>::Ptr input);

  private:

    std::string input_point_topic_;

    double sensor_height_;                //meters
    double general_max_slope_;            //degrees
    double local_max_slope_;              //degrees
    double radial_divider_angle_;         //distance in rads between dividers
    double concentric_divider_distance_;  //distance in meters between concentric divisions
    double min_height_threshold_;  //minimum height threshold regardless the slope, useful for close points
    double clipping_height_;  //the points higher than this will be removed from the input cloud.
    double min_point_distance_;  //minimum distance from the origin to consider a point as valid
    double reclass_distance_threshold_;  //distance between points at which re classification will occur

    size_t radial_dividers_num_;
    size_t concentric_dividers_num_;

    struct PointXYZIRTColor
    {
        PointXYZ point;

        float radius;       //cylindric coords on XY Plane
        float theta;        //angle deg on XY plane

        size_t radial_div;  //index of the radial divsion to which this point belongs to
        size_t concentric_div;  //index of the concentric division to which this points belongs to

        size_t original_index;  //index of this point in the source pointcloud
    };
    typedef std::vector<PointXYZIRTColor> PointCloudXYZIRTColor;

    /*!
     *
     * @param[in] in_cloud Input Point Cloud to be organized in radial segments
     * @param[out] out_organized_points Custom Point Cloud filled with XYZRTZColor data
     * @param[out] out_radial_divided_indices Indices of the points in the original cloud for each radial segment
     * @param[out] out_radial_ordered_clouds Vector of Points Clouds, each element will contain the points ordered
     */
    void
    ConvertXYZIToRTZColor (const PointCloud<PointXYZ>::Ptr in_cloud,
                           PointCloudXYZIRTColor& out_organized_points,
                           std::vector<PointIndices>& out_radial_divided_indices,
                           std::vector<PointCloudXYZIRTColor>& out_radial_ordered_clouds);

    /*!
     * Classifies Points in the PointCoud as Ground and Not Ground
     * @param in_radial_ordered_clouds Vector of an Ordered PointsCloud ordered by radial distance from the origin
     * @param out_ground_indices Returns the indices of the points classified as ground in the original PointCloud
     * @param out_no_ground_indices Returns the indices of the points classified as not ground in the original PointCloud
     */
    void
    ClassifyPointCloud (std::vector<PointCloudXYZIRTColor>& in_radial_ordered_clouds,
                        PointIndices& out_ground_indices,
                        PointIndices& out_no_ground_indices);

    /*!
     * Removes the points higher than a threshold
     * @param in_cloud_ptr PointCloud to perform Clipping
     * @param in_clip_height Maximum allowed height in the cloud
     * @param out_clipped_cloud_ptr Resultung PointCloud with the points removed
     */
    void
    ClipCloud (const PointCloud<PointXYZ>::Ptr in_cloud_ptr,
               double in_clip_height,
               PointCloud<PointXYZ>::Ptr out_clipped_cloud_ptr);

    /*!
     * Removes points up to a certain distance in the XY Plane
     * @param in_cloud_ptr Input PointCloud
     * @param in_min_distance Minimum valid distance, points closer than this will be removed.
     * @param out_filtered_cloud_ptr Resulting PointCloud with the invalid points removed.
     */
    void
    RemovePointsUpTo (const PointCloud<PointXYZ>::Ptr in_cloud_ptr,
                      double in_min_distance,
                      PointCloud<PointXYZ>::Ptr out_filtered_cloud_ptr);

    friend class RayGroundFilter_clipCloud_Test;

};

#endif  // RAY_GROUND_FILTER_H_
