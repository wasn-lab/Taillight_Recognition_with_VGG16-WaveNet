#include <ROS_ICLU3_v0.hpp>


ROS_API::ROS_API():
    _is_initialized(false),
    gui_name("passenger")
{
    // TODO: replace the following hardcoded path to an auto-detected one
    // path_pkg = "/home/benson516_itri/catkin_ws/src/opengl_test_ROS/opengl_test/";
    path_pkg = ros::package::getPath("opengl_test");
    if (path_pkg.back() != '/'){
        path_pkg += "/";
    }
    std::cout << "path_pkg = <" << path_pkg << ">\n";
}

// Setup node and start
bool ROS_API::start(int argc, char **argv, std::string node_name_in){
    // Setup the ROS interface
    ros_interface.setup_node(argc, argv, node_name_in);
    // Setup topics
    _set_up_topics();

    // Get parameters
    //-------------------------------//
    // Private parameters
    ros::NodeHandle pnh("~");
    // Param_name, variable_set, default_value
    pnh.param<std::string>("gui_name", gui_name, gui_name);
    //-------------------------------//

    //
    // Initialize vectors
    if (!_is_initialized){
        //
        got_on_any_topic.resize( ros_interface.get_count_of_all_topics(), false);
        any_ptr_list.resize( ros_interface.get_count_of_all_topics() );
        msg_time_list.resize( ros_interface.get_count_of_all_topics(), ros::Time(0) );
        // FPS
        fps_list.resize( ros_interface.get_count_of_all_topics() );
        for(size_t i=0; i < fps_list.size(); ++i){
            fps_list[i].set_name( ros_interface.get_topic_name(i) );
        }
        // end FPS
        _is_initialized = true;
    }
    //
    // start
    return ros_interface.start();
}

// Check if the ROS is started
bool ROS_API::is_running(){
    return ros_interface.is_running();
}

// Get the path of the package
std::string ROS_API::get_pkg_path(){
    return path_pkg;
}

// Updating data
bool ROS_API::update(){
    bool _updated = false;


    // All topics
    for (int topic_id=0; topic_id < any_ptr_list.size(); ++topic_id){
        if ( ros_interface.is_topic_a_input(topic_id) ){
            got_on_any_topic[topic_id] = ros_interface.get_any_message(topic_id, any_ptr_list[topic_id], msg_time_list[topic_id] );
            // if (got_on_any_topic[topic_id]){ fps_list[topic_id].stamp(); } // <-- Update FPS
            fps_list[topic_id].update(got_on_any_topic[topic_id]); // <-- Update FPS
            _updated |= got_on_any_topic[topic_id];
        }
    }
    //
    return _updated;
}

// New interfaces - boost::any and (void *)
//---------------------------------------------------------//
bool ROS_API::get_any_message(const int topic_id, boost::any & content_out_ptr){
    if ( topic_id >= got_on_any_topic.size() || !got_on_any_topic[topic_id] ){
        return false;
    }
    content_out_ptr = any_ptr_list[topic_id];
    return true;
}
bool ROS_API::get_any_message(const int topic_id, boost::any & content_out_ptr, ros::Time &msg_stamp){
    if ( topic_id >= got_on_any_topic.size() || !got_on_any_topic[topic_id] ){
        return false;
    }
    content_out_ptr = any_ptr_list[topic_id];
    msg_stamp = msg_time_list[topic_id];
    return true;
}
//---------------------------------------------------------//

// Transforms
//---------------------------------------------------------//
bool ROS_API::get_tf(std::string base_fram, std::string to_frame, geometry_msgs::TransformStamped & tf_out, bool is_time_traveling){
    return ros_interface.get_tf(base_fram, to_frame, tf_out, is_time_traveling, ros_interface.get_current_slice_time());
}
geometry_msgs::TransformStamped ROS_API::get_tf(std::string base_fram, std::string to_frame, bool & is_sucessed, bool is_time_traveling){
    return ros_interface.get_tf(base_fram, to_frame, is_sucessed, is_time_traveling, ros_interface.get_current_slice_time());
}
bool ROS_API::get_tf(std::string at_frame, geometry_msgs::TransformStamped & tf_out, bool is_time_traveling){
    return ros_interface.get_tf(at_frame, tf_out, is_time_traveling, ros_interface.get_current_slice_time());
}
geometry_msgs::TransformStamped ROS_API::get_tf(std::string at_frame, bool & is_sucessed, bool is_time_traveling){
    return ros_interface.get_tf(at_frame, is_sucessed, is_time_traveling, ros_interface.get_current_slice_time());
}
bool ROS_API::get_tf(const int topic_id, geometry_msgs::TransformStamped & tf_out, bool is_time_traveling){
    return ros_interface.get_tf(topic_id, tf_out, is_time_traveling, msg_time_list[topic_id]);
}
geometry_msgs::TransformStamped ROS_API::get_tf(const int topic_id, bool & is_sucessed, bool is_time_traveling){
    return ros_interface.get_tf(topic_id, is_sucessed, is_time_traveling, msg_time_list[topic_id]);
}
//---------------------------------------------------------//



//===========================================//
bool ROS_API::_set_up_topics(){
    {
        using MSG::M_TYPE;
#if __ROS_INTERFACE_VER__ == 1
        // tfGeoPoseStamped
        ros_interface.add_a_topic( int(MSG_ID::ego_pose), "current_pose", int(M_TYPE::tfGeoPoseStamped), true, 10, 100, "GUI_map", true, "GUI_base");
        // Vehicle info
        ros_interface.add_a_topic( int(MSG_ID::vehicle_info), "taichung_veh_info", int(M_TYPE::ITRICarInfoCarA), true, 100, 100, "GUI_base");
        ros_interface.add_a_topic( int(MSG_ID::dynamic_path), "dynamic_path_para", int(M_TYPE::ITRIDynamicPath), true, 100, 100, "GUI_base");
        ros_interface.add_a_topic( int(MSG_ID::local_path), "nav_path_astar_final", int(M_TYPE::NavPath), true, 100, 100, "GUI_map");
        // Flag_info
        ros_interface.add_a_topic( int(MSG_ID::flag_info_1), "Flag_Info01", int(M_TYPE::ITRIFlagInfo), true, 100, 100, "GUI_base");
        ros_interface.add_a_topic( int(MSG_ID::flag_info_2), "Flag_Info02", int(M_TYPE::ITRIFlagInfo), true, 100, 100, "GUI_base");
        ros_interface.add_a_topic( int(MSG_ID::flag_info_3), "Flag_Info03", int(M_TYPE::ITRIFlagInfo), true, 100, 100, "GUI_base");
        // Image
        ros_interface.add_a_topic( int(MSG_ID::camera_front_right), "camera/1/0/image_sync", int(M_TYPE::Image), true, 2, 20, "GUI_base");
        ros_interface.add_a_topic( int(MSG_ID::camera_front_center), "camera/1/1/image_sync", int(M_TYPE::Image), true, 2, 20, "GUI_base");
        ros_interface.add_a_topic( int(MSG_ID::camera_front_left), "camera/1/2/image_sync", int(M_TYPE::Image), true, 2, 20, "GUI_base");
        ros_interface.add_a_topic( int(MSG_ID::camera_front_top), "camera/0/2/image_sync", int(M_TYPE::Image), true, 2, 20, "GUI_base");
        ros_interface.add_a_topic( int(MSG_ID::camera_right_fore), "camera/2/0/image", int(M_TYPE::Image), true, 2, 20, "GUI_base");
        ros_interface.add_a_topic( int(MSG_ID::camera_right_rear), "camera/2/1/image", int(M_TYPE::Image), true, 2, 20, "GUI_base");
        ros_interface.add_a_topic( int(MSG_ID::camera_left_fore), "camera/0/0/image", int(M_TYPE::Image), true, 2, 20, "GUI_base");
        ros_interface.add_a_topic( int(MSG_ID::camera_left_rear), "camera/0/1/image", int(M_TYPE::Image), true, 2, 20, "GUI_base");
        ros_interface.add_a_topic( int(MSG_ID::camera_rear_center), "camera/2/2/image", int(M_TYPE::Image), true, 2, 20, "GUI_base");
        // 2D bounding box
        ros_interface.add_a_topic( int(MSG_ID::bounding_box_image_front_all), "CamMsg", int(M_TYPE::ITRICamObj), true, 10, 20, "GUI_base");
        ros_interface.add_a_topic( int(MSG_ID::bounding_box_image_front_top), "CamObj4", int(M_TYPE::ITRICamObj), true, 10, 20, "GUI_base");
        // PointCloud
        ros_interface.add_a_topic( int(MSG_ID::point_cloud_raw), "LidFrontLeft_sync", int(M_TYPE::ITRIPointCloud), true, 2, 20, "GUI_base");
        ros_interface.add_a_topic( int(MSG_ID::point_cloud_map), "points_map", int(M_TYPE::PointCloud2), true, 2, 20, "GUI_map");
        // Detection, tracking and pp.
        ros_interface.add_a_topic( int(MSG_ID::lidar_bounding_box_raw), "LidRoi", int(M_TYPE::ITRI3DBoundingBox), true, 10, 20, "GUI_base");
        ros_interface.add_a_topic( int(MSG_ID::lidar_bounding_box_tracking), "LiDAR_Track", int(M_TYPE::ITRICamObj), true, 10, 20, "GUI_base"); // <-- The tracking resuly is on map frame
        // NLOS boxs
        ros_interface.add_a_topic( int(MSG_ID::nlos_box), "V2X_msg", int(M_TYPE::ITRICamObj), true, 10, 20, "GUI_NLOS");
        ros_interface.add_a_topic( int(MSG_ID::nlos_gf), "NLOS_GF", int(M_TYPE::ITRITransObj), true, 10, 20, "GUI_map");
        // GUI operatios
        ros_interface.add_a_topic( int(MSG_ID::GUI_operatio), "GUI2/operation", int(M_TYPE::GUI2_op), true, 100, 100);
        ros_interface.add_a_topic( int(MSG_ID::GUI_state), "GUI2/state", int(M_TYPE::GUI2_op), false, 100, 1);
        //
#elif __ROS_INTERFACE_VER__ == 2
        // tfGeoPoseStamped
        ros_interface.add_a_topic( int(MSG_ID::ego_pose), "current_pose", int(M_TYPE::tfGeoPoseStamped), true, 10, 100, "GUI_map", true, "GUI_base");
        // Vehicle info
        ros_interface.add_a_topic( int(MSG_ID::vehicle_info), "veh_info", int(M_TYPE::ITRICarInfo), true, 100, 100, "GUI_base");
        ros_interface.add_a_topic( int(MSG_ID::dynamic_path), "dynamic_path_para", int(M_TYPE::ITRIDynamicPath), true, 100, 100, "GUI_base");
        ros_interface.add_a_topic( int(MSG_ID::local_path), "nav_path_astar_final", int(M_TYPE::NavPath), true, 100, 100, "GUI_map");
        // Flag_info
        ros_interface.add_a_topic( int(MSG_ID::flag_info_1), "Flag_Info01", int(M_TYPE::ITRIFlagInfo), true, 100, 100, "GUI_base");
        ros_interface.add_a_topic( int(MSG_ID::flag_info_2), "Flag_Info02", int(M_TYPE::ITRIFlagInfo), true, 100, 100, "GUI_base");
        ros_interface.add_a_topic( int(MSG_ID::flag_info_3), "Flag_Info03", int(M_TYPE::ITRIFlagInfo), true, 100, 100, "GUI_base");
        // Image
    #if __HINO_VER__ == 1
        ros_interface.add_a_topic( int(MSG_ID::camera_front_right), "gmsl_camera/port_a/cam_0/image_raw", int(M_TYPE::CompressedImageJpegOnly), true, 2, 20, "GUI_base");
        ros_interface.add_a_topic( int(MSG_ID::camera_front_center), "gmsl_camera/port_a/cam_1/image_raw", int(M_TYPE::CompressedImageJpegOnly), true, 2, 20, "GUI_base");
        ros_interface.add_a_topic( int(MSG_ID::camera_front_left), "gmsl_camera/port_a/cam_2/image_raw", int(M_TYPE::CompressedImageJpegOnly), true, 2, 20, "GUI_base");
        ros_interface.add_a_topic( int(MSG_ID::camera_front_top), "gmsl_camera/port_b/cam_0/image_raw", int(M_TYPE::CompressedImageJpegOnly), true, 2, 20, "GUI_base");
        ros_interface.add_a_topic( int(MSG_ID::camera_right_fore), "gmsl_camera/port_c/cam_0/image_raw", int(M_TYPE::CompressedImageJpegOnly), true, 2, 20, "GUI_base");
        ros_interface.add_a_topic( int(MSG_ID::camera_right_rear), "gmsl_camera/port_c/cam_1/image_raw", int(M_TYPE::CompressedImageJpegOnly), true, 2, 20, "GUI_base");
        ros_interface.add_a_topic( int(MSG_ID::camera_left_fore), "gmsl_camera/port_b/cam_1/image_raw", int(M_TYPE::CompressedImageJpegOnly), true, 2, 20, "GUI_base");
        ros_interface.add_a_topic( int(MSG_ID::camera_left_rear), "gmsl_camera/port_b/cam_2/image_raw", int(M_TYPE::CompressedImageJpegOnly), true, 2, 20, "GUI_base");
        ros_interface.add_a_topic( int(MSG_ID::camera_rear_center), "gmsl_camera/port_c/cam_2/image_raw", int(M_TYPE::CompressedImageJpegOnly), true, 2, 20, "GUI_base");
    #elif __HINO_VER__ == 2
        ros_interface.add_a_topic( int(MSG_ID::camera_front_right), "cam/F_right", int(M_TYPE::Image), true, 2, 20, "GUI_base");
        ros_interface.add_a_topic( int(MSG_ID::camera_front_center), "cam/F_center", int(M_TYPE::Image), true, 2, 20, "GUI_base");
        ros_interface.add_a_topic( int(MSG_ID::camera_front_left), "cam/F_left", int(M_TYPE::Image), true, 2, 20, "GUI_base");
        ros_interface.add_a_topic( int(MSG_ID::camera_front_top), "cam/F_top", int(M_TYPE::Image), true, 2, 20, "GUI_base");
        ros_interface.add_a_topic( int(MSG_ID::camera_right_fore), "cam/R_front", int(M_TYPE::Image), true, 2, 20, "GUI_base");
        ros_interface.add_a_topic( int(MSG_ID::camera_right_rear), "cam/R_rear", int(M_TYPE::Image), true, 2, 20, "GUI_base");
        ros_interface.add_a_topic( int(MSG_ID::camera_left_fore), "cam/L_front", int(M_TYPE::Image), true, 2, 20, "GUI_base");
        ros_interface.add_a_topic( int(MSG_ID::camera_left_rear), "cam/L_rear", int(M_TYPE::Image), true, 2, 20, "GUI_base");
        ros_interface.add_a_topic( int(MSG_ID::camera_rear_center), "cam/B_top", int(M_TYPE::Image), true, 2, 20, "GUI_base");
    #endif

        // 2D bounding box
        ros_interface.add_a_topic( int(MSG_ID::bounding_box_image_front_right), "CamObjFrontRight", int(M_TYPE::ITRIDetectedObjectArray), true, 10, 20, "GUI_base");
        ros_interface.add_a_topic( int(MSG_ID::bounding_box_image_front_center), "CamObjFrontCenter", int(M_TYPE::ITRIDetectedObjectArray), true, 10, 20, "GUI_base");
        ros_interface.add_a_topic( int(MSG_ID::bounding_box_image_front_left), "CamObjFrontLeft", int(M_TYPE::ITRIDetectedObjectArray), true, 10, 20, "GUI_base");
        ros_interface.add_a_topic( int(MSG_ID::bounding_box_image_front_top), "CamObjFrontTop", int(M_TYPE::ITRIDetectedObjectArray), true, 10, 20, "GUI_base");
        ros_interface.add_a_topic( int(MSG_ID::bounding_box_image_right_fore), "CamObjRightFront", int(M_TYPE::ITRIDetectedObjectArray), true, 10, 20, "GUI_base");
        ros_interface.add_a_topic( int(MSG_ID::bounding_box_image_right_rear), "CamObjRightBack", int(M_TYPE::ITRIDetectedObjectArray), true, 10, 20, "GUI_base");
        ros_interface.add_a_topic( int(MSG_ID::bounding_box_image_left_fore), "CamObjLeftFront", int(M_TYPE::ITRIDetectedObjectArray), true, 10, 20, "GUI_base");
        ros_interface.add_a_topic( int(MSG_ID::bounding_box_image_left_rear), "CamObjLeftBack", int(M_TYPE::ITRIDetectedObjectArray), true, 10, 20, "GUI_base");
        ros_interface.add_a_topic( int(MSG_ID::bounding_box_image_rear_center), "CamObjBackTop", int(M_TYPE::ITRIDetectedObjectArray), true, 10, 20, "GUI_base");
        // PointCloud
        ros_interface.add_a_topic( int(MSG_ID::point_cloud_raw), "LidarAll", int(M_TYPE::PointCloud2), true, 2, 20, "GUI_base");
        ros_interface.add_a_topic( int(MSG_ID::point_cloud_map), "points_map", int(M_TYPE::PointCloud2), true, 2, 20, "GUI_map");
        // Detection, tracking and pp.
        ros_interface.add_a_topic( int(MSG_ID::lidar_bounding_box_raw), "LidarDetection", int(M_TYPE::ITRIDetectedObjectArray), true, 10, 20, "GUI_base");
        ros_interface.add_a_topic( int(MSG_ID::lidar_bounding_box_tracking), "LiDAR_Track", int(M_TYPE::ITRIDetectedObjectArray), true, 10, 20, "GUI_base"); // <-- The tracking resuly is on map frame
        // NLOS boxs
        ros_interface.add_a_topic( int(MSG_ID::nlos_box), "V2X_msg", int(M_TYPE::ITRICamObj), true, 10, 20, "GUI_NLOS");
        ros_interface.add_a_topic( int(MSG_ID::nlos_gf), "NLOS_GF", int(M_TYPE::ITRITransObj), true, 10, 20, "GUI_map");
        // GUI operatios
        ros_interface.add_a_topic( int(MSG_ID::GUI_operatio), "GUI2/operation", int(M_TYPE::GUI2_op), true, 100, 100);
        ros_interface.add_a_topic( int(MSG_ID::GUI_state), "GUI2/state", int(M_TYPE::GUI2_op), false, 100, 1);
#endif  // __ROS_INTERFACE_VER__
        // GUI rendered draw
        ros_interface.add_a_topic( int(MSG_ID::GUI_screen_out), "GUI/screen_out", int(M_TYPE::Image), false, 10, 1);
        ros_interface.add_a_topic( int(MSG_ID::GUI_fps_out), "GUI/topic_fps_out", int(M_TYPE::String), false, 10, 1);
    }
    //------------------------------------------------//
    return true;
}
