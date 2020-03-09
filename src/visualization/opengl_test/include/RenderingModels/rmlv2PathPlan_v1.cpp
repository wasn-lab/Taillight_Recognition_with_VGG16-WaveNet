#include "rmlv2PathPlan_v1.h"


#include <cmath>       // ceil


rmlv2PathPlan_v1::rmlv2PathPlan_v1(
    std::string _path_Assets_in,
    int _ROS_topic_id_in,
    std::string data_representation_frame_in
):
    _ROS_topic_id(_ROS_topic_id_in),
    data_representation_frame(data_representation_frame_in),
    //
    rm_path(_path_Assets_in, _ROS_topic_id_in, 0),
    // rm_path(_path_Assets_in, _ROS_topic_id_in, 1),
    rm_text(_path_Assets_in, _ROS_topic_id_in)
{
    //
	Init();
}
void rmlv2PathPlan_v1::Init(){
    //
    _sim_time = 15.0f; // sec.
    _granularity = glm::vec2(0.2f, 0.087f); // 20 cm, 5 deg.
    _max_sim_point = 90;
    // _max_sim_point = int(_sim_time); // 90;

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
    _delta_T = glm::translate(glm::mat4(1.0), glm::vec3(0.0f, 0.0f, -2.6f) );
    _delta_T = glm::scale(_delta_T, glm::vec3(4.0f, 1.4f, 1.0f) );
    rm_path.set_close_loop(false);

    // Reshape and insert
    for (size_t i=0; i < section_vertexes.size(); ++i){
        section_vertexes[i] = (_delta_T * glm::vec4(section_vertexes[i], 1.0f)).xyz();
    }
    rm_path.insert_section_vertexes(section_vertexes);

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

void rmlv2PathPlan_v1::Update(float dt){
    // Update the data (buffer variables) here
}
void rmlv2PathPlan_v1::Update(ROS_INTERFACE &ros_interface){
    // Update the data (buffer variables) here
}

void rmlv2PathPlan_v1::Update(ROS_API &ros_api){
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


void rmlv2PathPlan_v1::Render(std::shared_ptr<ViewManager> &_camera_ptr){

    rm_path.Render(_camera_ptr);
    rm_text.Render(_camera_ptr);
}

void rmlv2PathPlan_v1::update_GL_data( ROS_API &ros_api ){



    // 2 paths
    std::vector<glm::vec2> param_1, param_2;
    std::vector<glm::vec2> param_1_t, param_2_t;

    // 1st section
    param_1.push_back( glm::vec2( msg_out_ptr->XP1_0, msg_out_ptr->YP1_0) );
    param_1.push_back( glm::vec2( msg_out_ptr->XP1_1, msg_out_ptr->YP1_1) );
    param_1.push_back( glm::vec2( msg_out_ptr->XP1_2, msg_out_ptr->YP1_2) );
    param_1.push_back( glm::vec2( msg_out_ptr->XP1_3, msg_out_ptr->YP1_3) );
    param_1.push_back( glm::vec2( msg_out_ptr->XP1_4, msg_out_ptr->YP1_4) );
    param_1.push_back( glm::vec2( msg_out_ptr->XP1_5, msg_out_ptr->YP1_5) );
    // 2nd section
    param_2.push_back( glm::vec2( msg_out_ptr->XP2_0, msg_out_ptr->YP2_0) );
    param_2.push_back( glm::vec2( msg_out_ptr->XP2_1, msg_out_ptr->YP2_1) );
    param_2.push_back( glm::vec2( msg_out_ptr->XP2_2, msg_out_ptr->YP2_2) );
    param_2.push_back( glm::vec2( msg_out_ptr->XP2_3, msg_out_ptr->YP2_3) );
    param_2.push_back( glm::vec2( msg_out_ptr->XP2_4, msg_out_ptr->YP2_4) );
    param_2.push_back( glm::vec2( msg_out_ptr->XP2_5, msg_out_ptr->YP2_5) );
    // std::cout << "param_1[0] = " << param_1[0].x << ", " << param_1[0].y << "\n";
    //


    // Get tf
    //----------------------------------//
    bool tf_successed = false;
    glm::mat4 _tf_m = ROStf2GLMmatrix(
        ros_api.get_tf(
            ros_api.ros_interface.get_topic_param(_ROS_topic_id).frame_id,
            data_representation_frame,
            tf_successed
        )
    );
    //
    if (!tf_successed){
        std::cout << "[rmlv2PathPlan_v1] No tf got, cannot transform the path.\n";
        return;
    }
    //----------------------------------//
    // end Get tf

    // Coordinate transformation
    // Path #1
    param_1_t.resize( param_1.size() );
    param_1_t[0] = ( _tf_m * glm::vec4( param_1[0], 0.0f, 1.0f) ).xy();
    for (size_t i=1; i < param_1.size(); ++i){
        // Note: Don't do the translation for the rest of the parameters
        param_1_t[i] = ( _tf_m * glm::vec4( param_1[i], 0.0f, 0.0f) ).xy();
    }
    // Path #2
    param_2_t.resize( param_2.size() );
    param_2_t[0] = ( _tf_m * glm::vec4( param_2[0], 0.0f, 1.0f) ).xy();
    for (size_t i=1; i < param_1.size(); ++i){
        // Note: Don't do the translation for the rest of the parameters
        param_2_t[i] = ( _tf_m * glm::vec4( param_2[i], 0.0f, 0.0f) ).xy();
    }
    // std::cout << "new param_1[0] = " << param_1_t[0].x << ", " << param_1_t[0].y << "\n";




    // Generate points
    int num_segment_per_path = _max_sim_point/2-1;
    float dT = 1.0f/float(num_segment_per_path);

    // Calculate paths
    _path.clear();
    glm::vec3 point3D_on_path;
    int _j = 0;
    // path #1
    for (size_t i=0; i <= num_segment_per_path; ++i ){
        float sim_T = i*dT;
        get_point3D_poly(param_1_t, sim_T, point3D_on_path);
        point3D_on_path += float(_j) * glm::vec3(0.0f, 0.0f, 0.01f);
        _path.push_back( point3D_on_path );
       _j++;
    }
    // std::cout << "--- middle _path point = " << _path[ _path.size()-1 ].x << ", " << _path[ _path.size()-1].y << "\n";

    // path #2, note: remove the first point
    // for (size_t i=(num_segment_per_path*0.2); i <= num_segment_per_path; ++i ){
    for (size_t i=1; i <= num_segment_per_path; ++i ){
        float sim_T = i*dT;
        get_point3D_poly(param_2_t, sim_T, point3D_on_path);
        point3D_on_path += float(_j) * glm::vec3(0.0f, 0.0f, 0.01f);
        _path.push_back( point3D_on_path );
        _j++;
    }
    // std::cout << "--- last _path point = " << _path[ _path.size()-1 ].x << ", " << _path[ _path.size()-1].y << "\n";
    // std::cout << "_path.size() = " << _path.size() << "\n";

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




//
void rmlv2PathPlan_v1::get_point3D_poly(const std::vector<glm::vec2> &param, double dT, glm::vec3 &point3D_out){

    //
    double _t = 1.0;
    point3D_out = glm::vec3(0.0f);
    for (size_t i=0; i < param.size(); ++i){
        point3D_out += glm::vec3( float( param[i][0]*_t ), float( param[i][1]*_t ), 0.0f);
        // Next level
        _t *= dT;
    }

}
