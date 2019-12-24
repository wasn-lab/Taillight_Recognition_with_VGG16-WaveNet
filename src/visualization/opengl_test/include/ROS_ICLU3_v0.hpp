#ifndef ROS_API_H
#define ROS_API_H

// Version control
//----------------------------------------//
#include "GUI_version_control.h"
//----------------------------------------//

// #include <ROS_interface.hpp>
// #include <ROS_interface_v2.hpp>
// #include <ROS_interface_v3.hpp>
#include <ROS_interface_v4.hpp>

#define __DEBUG__
// #define __SUB_IMAGES__
// #define __SUB_POINT_CLOUD__



// nickname for topic_id
enum class MSG_ID{
    // tfGeoPoseStamped
    ego_pose,
    vehicle_info,
    dynamic_path,
    //
    flag_info_1,
    flag_info_2,
    flag_info_3,
    // Image
    camera_front_right, // front-right
    camera_front_center, // front-center
    camera_front_left, // front-left
    camera_front_top, // front top-down
    camera_right_fore, // right-front
    camera_right_rear, // right-rear
    camera_left_fore, // left-front
    camera_left_rear, // left-rear
    camera_rear_center, // back
    // 2D bounding box
    bounding_box_image_front_all,
    bounding_box_image_front_right,
    bounding_box_image_front_center,
    bounding_box_image_front_left,
    bounding_box_image_front_top,
    bounding_box_image_right_fore,
    bounding_box_image_right_rear,
    bounding_box_image_left_fore,
    bounding_box_image_left_rear,
    bounding_box_image_rear_center,
    // PointCloud
    point_cloud_raw,
    point_cloud_map,
    // Detection, tracking and pp.
    lidar_bounding_box_raw,
    lidar_bounding_box_tracking,
    lidar_bounding_box_pp,
    //
    nlos_box,
    nlos_gf,
    //
    GUI_operatio,
    GUI_state,
    //
    GUI_screen_out,
    GUI_fps_out,
    // NUM_TOPICS
};



class ROS_API{
public:
    std::string gui_name;
    // the ROS_interface
    ROS_INTERFACE ros_interface;
    std::string path_pkg;


    // Data validation (only be used after calling update)
    std::vector<bool>               got_on_any_topic;
    std::vector< boost::any >       any_ptr_list;
    std::vector< ros::Time >        msg_time_list;
    std::vector< TIME_STAMP::FPS>   fps_list;


    // Methods
    ROS_API();
    // Setup node and start
    bool start(int argc, char **argv, std::string node_name_in=std::string("ROS_interface"));
    // Check if the ROS is started
    bool is_running();

    // Public methods
    std::string get_pkg_path();
    bool update(); // Updating data

    // New interfaces - boost::any and (void *)
    //---------------------------------------------------------//
    bool get_any_message(const int topic_id, boost::any & content_out_ptr);
    bool get_any_message(const int topic_id, boost::any & content_out_ptr, ros::Time &msg_stamp);
    //
    template <class _T> bool get_message(const int topic_id, _T & content_out){
        if ( topic_id >= got_on_any_topic.size() || !got_on_any_topic[topic_id] ){
            return false;
        }
        std::shared_ptr< _T > *_ptr_ptr = boost::any_cast< std::shared_ptr< _T > >( &(any_ptr_list[topic_id]) );
        content_out = *(*_ptr_ptr); // <-- Note: be carefull with cv::Mat
        return true;
    }
    template <class _T> bool get_message(const int topic_id, std::shared_ptr<_T> & content_out_ptr){
        if ( topic_id >= got_on_any_topic.size() || !got_on_any_topic[topic_id] ){
            return false;
        }
        std::shared_ptr< _T > *_ptr_ptr = boost::any_cast< std::shared_ptr< _T > >( &(any_ptr_list[topic_id]) );
        content_out_ptr = *_ptr_ptr;
        return true;
    }
    template <class _T> bool get_message(const int topic_id, std::shared_ptr<_T> & content_out_ptr, ros::Time &msg_stamp){
        if ( topic_id >= got_on_any_topic.size() || !got_on_any_topic[topic_id] ){
            return false;
        }
        std::shared_ptr< _T > *_ptr_ptr = boost::any_cast< std::shared_ptr< _T > >( &(any_ptr_list[topic_id]) );
        content_out_ptr = *_ptr_ptr;
        msg_stamp = msg_time_list[topic_id];
        return true;
    }
    //---------------------------------------------------------//

    // Transforms
    //---------------------------------------------------------//
    bool get_tf(std::string base_fram, std::string to_frame, geometry_msgs::TransformStamped & tf_out, bool is_time_traveling=false);
    geometry_msgs::TransformStamped get_tf(std::string base_fram, std::string to_frame, bool & is_sucessed, bool is_time_traveling=false);
    bool get_tf(std::string at_frame, geometry_msgs::TransformStamped & tf_out, bool is_time_traveling=false);
    geometry_msgs::TransformStamped get_tf(std::string at_frame, bool & is_sucessed, bool is_time_traveling=false);
    bool get_tf(const int topic_id, geometry_msgs::TransformStamped & tf_out, bool is_time_traveling=false);
    geometry_msgs::TransformStamped get_tf(const int topic_id, bool & is_sucessed, bool is_time_traveling=false);
    //---------------------------------------------------------//

private:
    bool _is_initialized;
    bool _set_up_topics();
};








/*
namespace ROS_API_TOOL{
    template <class _T> bool get_message(ROS_API & ros_api, const int topic_id, _T & content_out);
    template <class _T> bool get_message(ROS_API & ros_api, const int topic_id, std::shared_ptr<_T> & content_out_ptr);
    template <class _T> bool get_message(ROS_API & ros_api, const int topic_id, std::shared_ptr<_T> & content_out_ptr, ros::Time &msg_stamp);
}
*/
namespace ROS_API_TOOL
{
    template <class _T> bool get_message(ROS_API & ros_api, const int topic_id, _T & content_out){
        if ( !ros_api.got_on_any_topic[topic_id] ){
            return false;
        }
        std::shared_ptr< _T > *_ptr_ptr = boost::any_cast< std::shared_ptr< _T > >( &(ros_api.any_ptr_list[topic_id]) );
        content_out = *(*_ptr_ptr); // <-- Note: be carefull with cv::Mat
        return true;
    }
    template <class _T> bool get_message(ROS_API & ros_api, const int topic_id, std::shared_ptr<_T> & content_out_ptr){
        if ( !ros_api.got_on_any_topic[topic_id] ){
            return false;
        }
        std::shared_ptr< _T > *_ptr_ptr = boost::any_cast< std::shared_ptr< _T > >( &(ros_api.any_ptr_list[topic_id]) );
        content_out_ptr = *_ptr_ptr;
        return true;
    }
    template <class _T> bool get_message(ROS_API & ros_api, const int topic_id, std::shared_ptr<_T> & content_out_ptr, ros::Time &msg_stamp){
        if ( !ros_api.got_on_any_topic[topic_id] ){
            return false;
        }
        std::shared_ptr< _T > *_ptr_ptr = boost::any_cast< std::shared_ptr< _T > >( &(ros_api.any_ptr_list[topic_id]) );
        content_out_ptr = *_ptr_ptr;
        msg_stamp = ros_api.msg_time_list[topic_id];
        return true;
    }
}

/*
// Using put_any() with content_in: (The following syntex makes a clear poiter transfer withoud copying or sharring)
//---------------------------------------//
boost::any any_ptr;
{
    std::shared_ptr< _T > _content_ptr = std::make_shared< _T >( content_in ); // <-- If we already get the std::shared_ptr, ignore this line
    any_ptr = _content_ptr;
} // <-- Note: the _content_ptr is destroyed when leaveing the scope, thus the use_count for the _ptr in any_ptr is "1" (unique).
buffwr_obj.put_any(any_ptr, true, _time_in, true);
//---------------------------------------//

// Using put_any() with content_in_ptr: (The following syntex makes a clear poiter transfer withoud copying or sharring)
//---------------------------------------//
boost::any any_ptr;
{
    std::shared_ptr< _T > _content_ptr = std::make_shared< _T >( *content_in_ptr ); // <-- If we already get the std::shared_ptr, ignore this line
    any_ptr = _content_ptr;
} // <-- Note: the _content_ptr is destroyed when leaveing the scope, thus the use_count for the _ptr in any_ptr is "1" (unique).
buffwr_obj.put_any(any_ptr, true, _time_in, true);
//---------------------------------------//

// Using front_any() with content_out_ptr: (The following syntex makes a clear poiter transfer withoud copying or sharring)
//---------------------------------------//
std::shared_ptr< _T > content_out_ptr;
{
    boost::any any_ptr;
    bool result = buffwr_list[topic_id]->front_any(any_ptr, true, _current_slice_time);
    if (result){
        // content_out_ptr = boost::any_cast< std::shared_ptr< cv::Mat > >( any_ptr ); // <-- Not good, this makes a copy
        std::shared_ptr< cv::Mat > *_ptr_ptr = boost::any_cast< std::shared_ptr< cv::Mat > >( &any_ptr );
        content_out_ptr = *_ptr_ptr;
    }
} // <-- Note: the _ptr_ptr is destroyed when leaving this scope, thus the use_count for content_out_ptr is "1" (unique).
//---------------------------------------//
*/

#endif // ROS_API_H
