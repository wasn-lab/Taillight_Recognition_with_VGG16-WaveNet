#include "rmlv2PathPlanFake.h"


#include <math.h>       // ceil


rmlv2PathPlanFake::rmlv2PathPlanFake(
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
void rmlv2PathPlanFake::Init(){
    //
    _sim_time = 15.0f; // sec.
    _granularity = glm::vec2(0.2f, 0.087f); // 20 cm, 5 deg.
    _max_sim_point = 90;
    // _max_sim_point = int(_sim_time); // 90;

    //
    section_vertexes.resize(4);
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
    section_vertexes[0] = glm::vec3(0.0f, -1.0f, 0.0f);
    section_vertexes[1] = glm::vec3(0.0f, 1.0f, 0.0f);
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




    // For adjusting the model pose by public methods
    attach_pose_model_by_model_ref_ptr( *rm_path.get_model_m_ptr() );
    attach_pose_model_by_model_ref_ptr( *rm_text.get_model_m_ptr() );

}

void rmlv2PathPlanFake::Update(float dt){
    // Update the data (buffer variables) here
}
void rmlv2PathPlanFake::Update(ROS_INTERFACE &ros_interface){
    // Update the data (buffer variables) here
}

void rmlv2PathPlanFake::Update(ROS_API &ros_api){
    // Update the data (buffer variables) here
    // test, use transform
    ros::Time msg_time;
    bool _result = false;
    _result = ros_api.get_message(_ROS_topic_id, msg_out_ptr, msg_time);

    if (_result){
        update_GL_data();
        // rm_text.insert_text();
    }

    //
    rm_path.Update(ros_api);
    rm_text.Update(ros_api);
}


void rmlv2PathPlanFake::Render(std::shared_ptr<ViewManager> &_camera_ptr){

    rm_path.Render(_camera_ptr);
    rm_text.Render(_camera_ptr);
}

void rmlv2PathPlanFake::update_GL_data(){

    glm::vec3 pose2D_0(0.0f, 0.0f, 0.0f);
    // glm::vec3 pose2D_0(-5.5f, 0.0f, 0.0f);
    glm::vec2 twist2D( msg_out_ptr->ego_speed, msg_out_ptr->yaw_rate*0.017453292519943295 );



    //
    int num_steps = 0;
    float dT = _sim_time;
    //
    int idx_primary_axis = 0;
    glm::vec2 cross_freq = glm::abs(twist2D/_granularity);
    // Find the primary axis (between translation and rotation)
    if (cross_freq.x >= cross_freq.y){
        idx_primary_axis = 0;
    }else{
        idx_primary_axis = 1;
    }
    if (cross_freq[idx_primary_axis] <= 0.0){
        // Not moving
        dT = _sim_time;
        num_steps = 1;
    } else{
        // else
        dT = 1.0/cross_freq[idx_primary_axis]; // Positive value
        num_steps = int( ceil(_sim_time/dT) );
    }

    // Saturation
    if (num_steps > _max_sim_point){
        num_steps = _max_sim_point;
        dT = _sim_time/float(num_steps);
    }

    // Calculate path
    _path.clear();
    _path.push_back(pose2D_0);
    glm::vec3 pose2D_on_path;
    for (size_t i=1; i <= num_steps; ++i ){
        float sim_T = i*dT;
        get_pose2D_sim(pose2D_0, twist2D, sim_T, pose2D_on_path);
        _path.push_back( glm::vec3(pose2D_on_path.xy(), 0.0f) );
    }
    // std::cout << "_path.size() = " << _path.size() << "\n";

    rm_path.insert_curve_Points(_path);

    // // Reset
    // text_list.clear();
    //
    // // Insert texts
    // rm_text.insert_text(text_list);

}



//
void rmlv2PathPlanFake::get_pose2D_sim(const glm::vec3 &pose2D_0, const glm::vec2 &twist2D, double dT, glm::vec3 &pose2D_out){
    double dtheta = twist2D[1]*dT;
    //
    // double x_0 = pose2D_0.x;
    // double y_0 = pose2D_0.y;
    // double theta_0 = pose2D_0.z;
    // double vel_1 = twist2D.x;
    // double omega_1 = twist2D.y;
    //
    // double x_1, y_1, theta_1, r_1;
    //
    if (std::abs(dtheta) < 0.017453292519943295){ // 1 deg., small angle
        pose2D_out[2] = pose2D_0[2] + dtheta;
        pose2D_out.x = pose2D_0.x + twist2D[0]*dT*std::cos( 0.5*(pose2D_out[2] + pose2D_0[2]) );
        pose2D_out.y = pose2D_0.y + twist2D[0]*dT*std::sin( 0.5*(pose2D_out[2] + pose2D_0[2]) );
    }else{ //
        pose2D_out[2] = pose2D_0[2] + dtheta;
        double r_1 = twist2D[0]/twist2D[1];
        pose2D_out.x = pose2D_0.x + r_1*( std::sin(pose2D_out[2]) - std::sin(pose2D_0[2]) );
        pose2D_out.y = pose2D_0.y - r_1*( std::cos(pose2D_out[2]) - std::cos(pose2D_0[2]) );
    }
}
