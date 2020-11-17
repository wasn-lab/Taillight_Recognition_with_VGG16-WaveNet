#include "rmlv2ObjectTracking.h"




rmlv2ObjectTracking::rmlv2ObjectTracking(
    std::string _path_Assets_in,
    int _ROS_topic_id_in,
    std::string ref_frame_in
):
    _ROS_topic_id(_ROS_topic_id_in),
    _ref_frame(ref_frame_in),
    //
    rm_polylines3D(_path_Assets_in, ref_frame_in),
    rm_circle(_path_Assets_in, ref_frame_in),
    rm_text(_path_Assets_in, ref_frame_in)
{
    // init_paths(_path_Assets_in);
    //
	Init();
}
void rmlv2ObjectTracking::Init(){


    // For adjusting the model pose by public methods
    attach_pose_model_by_model_ref_ptr( *rm_polylines3D.get_model_m_ptr() );
    attach_pose_model_by_model_ref_ptr( *rm_circle.get_model_m_ptr() );
    attach_pose_model_by_model_ref_ptr( *rm_text.get_model_m_ptr() );


    //Load model to shader _program_ptr
	LoadModel();

}

void rmlv2ObjectTracking::LoadModel(){

    // Line width
    rm_polylines3D.set_line_width(5.0f);
    rm_circle.set_line_width(1.0f);

    //
    float max_track_time = 10; // sec
    float max_miss_sec = 10; // sec.
    float sample_rate = 10.0; // Hz
    //
    _max_miss_count = max_track_time*sample_rate;
    _max_track_points = max_miss_sec*sample_rate;

}

void rmlv2ObjectTracking::Update(float dt){
    // Update the data (buffer variables) here
}
void rmlv2ObjectTracking::Update(ROS_INTERFACE &ros_interface){
    // Update the data (buffer variables) here
}

void rmlv2ObjectTracking::Update(ROS_API &ros_api){
    // Update the data (buffer variables) here
    // test, use transform
    ros::Time msg_time;
    bool _result = false;
    _result = ros_api.get_message(_ROS_topic_id, msg_out_ptr, msg_time);

    if (_result){
        update_GL_data(ros_api);
        // rm_text.insert_text();
    }

    //
    rm_polylines3D.Update(ros_api);
    rm_circle.Update(ros_api);
    rm_text.Update(ros_api);
}


void rmlv2ObjectTracking::Render(std::shared_ptr<ViewManager> &_camera_ptr){

    rm_polylines3D.Render(_camera_ptr);
    rm_circle.Render(_camera_ptr);
    rm_text.Render(_camera_ptr);
}



void rmlv2ObjectTracking::update_GL_data(ROS_API &ros_api){
    // Reset
    text_list.clear();
    //
    if (msg_out_ptr->objects.size() == 0){
        // Insert texts
        rm_text.insert_text(text_list);
        return;
    }


    // Update transform
    //--------------------------------//
    glm::mat4 tf_box_to_ref(1.0f);
    if (_ref_frame.size() > 0){
        // Get tf
        bool tf_successed = false;
        tf_box_to_ref = ROStf2GLMmatrix(ros_api.get_tf(_ref_frame, ros_api.ros_interface.get_topic_param(_ROS_topic_id).frame_id, tf_successed));
        // end Get tf
    }
    //--------------------------------//

    long long num_box = msg_out_ptr->objects.size();
    // if (num_box > _max_num_box){
    //     num_box = _max_num_box;
    // }

    // Initialize
    // std::vector<rmCircle::circle_data> circle_list;

    auto * _point_1_ptr = &(msg_out_ptr->objects[0].bPoint.p0);
    auto * _point_2_ptr = &(msg_out_ptr->objects[0].bPoint.p0);
    size_t _j = 0;
    for (size_t i = 0; i < num_box; i++)
    {
        long long obj_id = msg_out_ptr->objects[i].camInfo[0].id;
        _point_1_ptr = &(msg_out_ptr->objects[i].bPoint.p0);
        _point_2_ptr = &(msg_out_ptr->objects[i].bPoint.p7);
        glm::vec3 point_pose_ori = (  0.5f*(glm::vec3(_point_1_ptr->x, _point_1_ptr->y, _point_1_ptr->z) + glm::vec3(_point_2_ptr->x, _point_2_ptr->y, _point_2_ptr->z)) + glm::vec3(0.0f, 0.0f, 0.0f)  );
        glm::vec3 point_pose = ( tf_box_to_ref*glm::vec4(point_pose_ori, 1.0f) ).xyz();
        float diag_distance = glm::l2Norm(glm::vec3(_point_1_ptr->x, _point_1_ptr->y, _point_1_ptr->z) - glm::vec3(_point_2_ptr->x, _point_2_ptr->y, _point_2_ptr->z));

        // Reset count
        std::map<long long,int>::iterator it_1 = obj_miss_count.find(obj_id);
        if (it_1 == obj_miss_count.end()){
            obj_miss_count[obj_id] = 0;
        }else{
            obj_miss_count[obj_id] -= 2; // We will add this by one later
        }
        //


        // A line
        //-------------------------//
        std::queue<rmPolyLines3D::point_data> &a_line_queue = line_map[obj_id];
        a_line_queue.emplace(
            point_pose,
            glm::vec3(1.0f, 0.0f, 0.0f)
        );
        while(a_line_queue.size() > _max_track_points){ // test, track for 100 points
            a_line_queue.pop();
        }
        // rm_polylines3D.push_back_a_line_queue(a_line_queue);
        //-------------------------//

        // Circle
        auto point_tmp = a_line_queue.back();
        circle_map[obj_id] = rmCircle::circle_data(
                                    point_tmp.position,
                                    diag_distance*0.5, //1.0f,
                                    glm::vec3(1.0f, 1.0f, 0.0f)
                                );

        // // // A set of circles
        // // //-------------------------//
        // // std::queue<rmPolyLines3D::point_data> point_queue_tmp = a_line_queue;
        // // while ( !point_queue_tmp.empty() ){
        // //     auto point_tmp = point_queue_tmp.front();
        // //     circle_list.emplace_back(
        // //         point_tmp.position,
        // //         0.5f,
        // //         glm::vec3(1.0f, 0.0f, 0.0f)
        // //     );
        // //     point_queue_tmp.pop();
        // // }
        // // //-------------------------//
        //
        //
        // // Only insert the circle at the last point
        // auto point_tmp = a_line_queue.back();
        // circle_list.emplace_back(
        //     point_tmp.position,
        //     diag_distance*0.7, //1.0f,
        //     glm::vec3(1.0f, 1.0f, 0.0f)
        // );


    }


    // Iterate through tracked objects
    for (std::map<long long,int>::iterator it=obj_miss_count.begin(); it!=obj_miss_count.end(); ++it){
        it->second++;
        if (it->second > _max_miss_count){ // test, miss count
            // line_map[it->first] = std::queue<rmPolyLines3D::point_data>();
            // it->second = 0;
            line_map.erase(it->first);
            circle_map.erase(it->first);
            obj_miss_count.erase(it);
            continue;
        }else if (it->second < 0){
            it->second = 0;
        }
    }

    // Draw lines and circles
    rm_polylines3D.reset_line_list();
    for (std::map<long long,int>::iterator it=obj_miss_count.begin(); it!=obj_miss_count.end(); ++it){
        std::queue<rmPolyLines3D::point_data> &a_line_queue = line_map[it->first];
        rm_polylines3D.push_back_a_line_queue(a_line_queue);

        // // // A set of circles
        // // //-------------------------//
        // // std::queue<rmPolyLines3D::point_data> point_queue_tmp = a_line_queue;
        // // while ( !point_queue_tmp.empty() ){
        // //     auto point_tmp = point_queue_tmp.front();
        // //     circle_list.emplace_back(
        // //         point_tmp.position,
        // //         0.5f,
        // //         glm::vec3(1.0f, 0.0f, 0.0f)
        // //     );
        // //     point_queue_tmp.pop();
        // // }
        // // //-------------------------//
        //
        //
        // // Only insert the circle at the last point
        // auto point_tmp = a_line_queue.back();
        // circle_list.emplace_back(
        //     point_tmp.position,
        //     diag_distance*0.7, //1.0f,
        //     glm::vec3(1.0f, 1.0f, 0.0f)
        // );

    }
    // std::cout << "line_map.size() = " << line_map.size() << ", ";
    // std::cout << "circle_map.size() = " << circle_map.size() << ", ";
    // std::cout << "obj_miss_count.size() = " << obj_miss_count.size() << "\n";

    // Insert circles
    // rm_circle.insert_circle(circle_list);
    rm_circle.insert_circle(circle_map);


    // Insert texts
    // rm_text.insert_text(text_list);
}
