#include "rmlv2PathPlan_navPath.h"


#include <cmath>       // ceil


rmlv2PathPlan_navPath::rmlv2PathPlan_navPath(
    std::string _path_Assets_in,
    int _ROS_topic_id_in
):
    _ROS_topic_id(_ROS_topic_id_in),
    //
    rm_path(_path_Assets_in, _ROS_topic_id_in, 0),
    // rm_path(_path_Assets_in, _ROS_topic_id_in, 1),
    rm_text(_path_Assets_in, _ROS_topic_id_in)
{
    //
	Init();
}
void rmlv2PathPlan_navPath::Init(){
    //
    _sim_time = 15.0f; // sec.
    _granularity = glm::vec2(0.2f, 0.087f); // 20 cm, 5 deg.
    _max_sim_point = 100; // 90;
    // _max_sim_point = int(_sim_time); // 90;

    // Section
    geometry_msgs::Quaternion q_init;
    q_init.x = 0.0f;
    q_init.y = 0.0f;
    q_init.z = 0.0f;
    q_init.w = 1.0f;
    update_section_orientation(q_init);

    rm_path.set_line_width(2.0f);
    // rm_path.set_color_head( glm::vec3(1.0f, 0.5f, 0.0f) );
    // rm_path.set_color_tail( glm::vec3(0.0f, 0.5f, 1.0f) );
    rm_path.set_color_head( glm::vec3(0.0f, 0.5f, 1.0f) );
    rm_path.set_color_tail( glm::vec3(1.0f, 0.5f, 0.0f) );


    // Clean the path
    rm_path.insert_curve_Points(_path);


    // For adjusting the model pose by public methods
    attach_pose_model_by_model_ref_ptr( *rm_path.get_model_m_ptr() );
    attach_pose_model_by_model_ref_ptr( *rm_text.get_model_m_ptr() );

}

void rmlv2PathPlan_navPath::Update(float dt){
    // Update the data (buffer variables) here
}
void rmlv2PathPlan_navPath::Update(ROS_INTERFACE &ros_interface){
    // Update the data (buffer variables) here
}

void rmlv2PathPlan_navPath::Update(ROS_API &ros_api){
    // Update the data (buffer variables) here
    // test, use transform
    ros::Time msg_time;
    bool _result = false;
    _result = ros_api.get_message(_ROS_topic_id, msg_out_ptr, msg_time);

    if (_result){
        update_GL_data( ros_api );
        // rm_text.insert_text();
    }



    //
    rm_path.Update(ros_api);
    rm_text.Update(ros_api);
}


void rmlv2PathPlan_navPath::Render(std::shared_ptr<ViewManager> &_camera_ptr){

    rm_path.Render(_camera_ptr);
    rm_text.Render(_camera_ptr);
}

void rmlv2PathPlan_navPath::update_GL_data( ROS_API &ros_api ){

    size_t path_size = msg_out_ptr->poses.size();

    if (path_size == 0){
        _path.clear();
        rm_path.insert_curve_Points(_path);
        return;
    }

    if (path_size > _max_sim_point){
        std::cout << "[rmlv2PathPlan_navPath] path exceeds the maximum plotable length ( path_size = " << path_size << "), dropping tailing points.\n";
        path_size = _max_sim_point;
    }


    // Update section
    update_section_orientation(msg_out_ptr->poses[0].pose.orientation);

    // Generate points

    // Calculate paths
    _path.clear();
    glm::vec3 point3D_on_path;
    int _j = 0;
    // path #1
    for (size_t i=0; i < path_size; ++i ){
        point3D_on_path[0] = msg_out_ptr->poses[i].pose.position.x;
        point3D_on_path[1] = msg_out_ptr->poses[i].pose.position.y;
        point3D_on_path[2] = msg_out_ptr->poses[i].pose.position.z;
        // point3D_on_path += float(_j) * glm::vec3(0.0f, 0.0f, 0.01f);
        _path.push_back( point3D_on_path );
       _j++;
    }
    // std::cout << "--- middle _path point = " << _path[ _path.size()-1 ].x << ", " << _path[ _path.size()-1].y << "\n";



    // std::cout << "---\n_path = \n";
    // for (size_t i=0; i < _path.size(); ++i){
    //     std::cout << "(" << _path[i].x << ", " << _path[i].y << ")\n";
    // }
    // std::cout << "\n";

    rm_path.insert_curve_Points(_path);

    // // Reset
    // text_list.clear();
    //
    // // Insert texts
    // rm_text.insert_text(text_list);

}



bool rmlv2PathPlan_navPath::update_section_orientation(geometry_msgs::Quaternion q_in){

    glm::quat _rot_q(q_in.w, q_in.x, q_in.y, q_in.z);
    glm::mat4 _rot_m = glm::toMat4(_rot_q);

    section_vertexes.clear();
    //
    // section_vertexes.resize(4);
    // // board
    // section_vertexes[0] = glm::vec3(0.0f, -1.0f, -1.0f);
    // section_vertexes[1] = glm::vec3(0.0f, -1.0f, 1.0f);
    // section_vertexes[2] = glm::vec3(0.0f, 1.0f, 1.0f);
    // section_vertexes[3] = glm::vec3(0.0f, 1.0f, -1.0f);
    // glm::mat4 _delta_T = glm::scale(glm::mat4(1.0), glm::vec3(1.0f, 0.1f, 1.0f) );
    // rm_path.set_close_loop(true);

    // // footprint
    // section_vertexes[0] = glm::vec3(-1.0f, -1.0f, 0.0f);
    // section_vertexes[1] = glm::vec3(-1.0f, 1.0f, 0.0f);
    // section_vertexes[2] = glm::vec3(1.0f, 1.0f, 0.01f);
    // section_vertexes[3] = glm::vec3(1.0f, -1.0f, 0.01f);
    // glm::mat4 _delta_T = glm::translate(glm::mat4(1.0), glm::vec3(3.0f, 0.0f, -2.6f) );
    // _delta_T = glm::scale(_delta_T, glm::vec3(4.0f, 1.4f, 1.0f) );
    // rm_path.set_close_loop(true);

    // flat board
    section_vertexes.push_back( glm::vec3(0.0f, -1.0f, 0.0f) );
    section_vertexes.push_back( glm::vec3(0.0f, 1.0f, 0.0f) );
    glm::mat4 _delta_T(1.0f);
    _delta_T = glm::scale(_delta_T, glm::vec3(4.0f, 1.4f, 1.0f) );
    _delta_T = _rot_m * _delta_T;
    _delta_T = glm::translate(glm::mat4(1.0), glm::vec3(0.0f, 0.0f, -2.6f) ) * _delta_T;
    rm_path.set_close_loop(false);

    // Reshape and insert
    for (size_t i=0; i < section_vertexes.size(); ++i){
        section_vertexes[i] = (_delta_T * glm::vec4(section_vertexes[i], 1.0f)).xyz();
    }
    rm_path.insert_section_vertexes(section_vertexes);
    return false;
}
