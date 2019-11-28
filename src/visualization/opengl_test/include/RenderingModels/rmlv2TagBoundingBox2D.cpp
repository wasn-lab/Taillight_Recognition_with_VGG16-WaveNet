#include "rmlv2TagBoundingBox2D.h"
#include "object_class_def.hpp"



OBJECT_CLASS obj_class_lv2;

rmlv2TagBoundingBox2D::rmlv2TagBoundingBox2D(
    std::string _path_Assets_in,
    int _ROS_topic_id_in,
    bool is_perspected_in,
    bool is_moveable_in
):
    is_perspected(is_perspected_in),
    is_moveable(is_moveable_in),
    _ROS_topic_id(_ROS_topic_id_in),
    //
    rm_box(_path_Assets_in, _ROS_topic_id_in, is_perspected_in, is_moveable_in),
    rm_text(_path_Assets_in, _ROS_topic_id_in)
{
    //
	Init();
}
void rmlv2TagBoundingBox2D::Init(){


    // For adjusting the model pose by public methods
    attach_pose_model_by_model_ref_ptr( *rm_box.get_model_m_ptr() );
    attach_pose_model_by_model_ref_ptr( *rm_text.get_model_m_ptr() );

}

void rmlv2TagBoundingBox2D::Update(float dt){
    // Update the data (buffer variables) here
}
void rmlv2TagBoundingBox2D::Update(ROS_INTERFACE &ros_interface){
    // Update the data (buffer variables) here
}

void rmlv2TagBoundingBox2D::Update(ROS_API &ros_api){
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
    rm_box.Update(ros_api);
    rm_text.Update(ros_api);
}


void rmlv2TagBoundingBox2D::Render(std::shared_ptr<ViewManager> &_camera_ptr){

    rm_box.Render(_camera_ptr);
    rm_text.Render(_camera_ptr);
}
void rmlv2TagBoundingBox2D::Reshape(const glm::ivec2 & viewport_size_in){
    _viewport_size = viewport_size_in;
    updateBoardGeo();
    rm_box.Reshape(_viewport_size);
    rm_text.Reshape(_viewport_size);
}



void rmlv2TagBoundingBox2D::update_GL_data(){
    // Reset
    if (is_perspected){
        text2Din3D_list.clear();
    }else{
        text2Dflat_list.clear();
    }

    //
    if (msg_out_ptr->objects.size() == 0){
        // Insert texts
        if (is_perspected){
            rm_text.insert_text(text2Din3D_list);
        }else{
            rm_text.insert_text(text2Dflat_list);
        }
        return;
    }
    long long num_box = msg_out_ptr->objects.size();
    /*
    if (num_box > _max_num_box){
        num_box = _max_num_box;
    }
    */


    //
    size_t _box_count = 0;
    for (size_t i = 0; i < num_box; i++)
	{
        //
        auto & _box = msg_out_ptr->objects[i];
        box_param_cv _a_box_param_cv(_box.camInfo.u, _box.camInfo.v, _box.camInfo.width, _box.camInfo.height, _box.classId);
        box_param_gl _a_box_param_gl;
        convert_cv_to_normalized_gl(_a_box_param_cv, _a_box_param_gl);
        if (!is_gl_box_valid(_a_box_param_gl)){
            continue; // Don't add to buffer
        }
        _box_count++;
        //
        // glm::vec3 _tag_color = rmlv2TagBoundingBox2D_ns::get_obj_class_color(_a_box_param_gl.obj_class);
        // std::string _tag_str( "#" + std::to_string(_box.camInfo.id) + "\n" + rmlv2TagBoundingBox2D_ns::get_obj_class_string(_box.classId) );
        glm::vec3 _tag_color = obj_class_lv2.get_color(_a_box_param_gl.obj_class);
        // std::string _tag_str( "#" + std::to_string(_box.camInfo.id) + "\n" + obj_class_lv2.get_string(_box.classId) );
        std::string _tag_str(obj_class_lv2.get_string(_box.classId) );
        if (is_perspected){
            text2Din3D_list.emplace_back(
                _tag_str,
                _a_box_param_gl.xy_list[0],
                0.1,
                _tag_color,
                ALIGN_X::LEFT,
                ALIGN_Y::BUTTON,
                1
            );
        }else{
            text2Dflat_list.emplace_back(
                _tag_str,
                _a_box_param_gl.xy_list[0],
                24,
                _tag_color,
                ALIGN_X::LEFT,
                ALIGN_Y::BUTTON,
                1,
                0,
                !is_moveable,
                false
            );
        }
        //
	}

    // Insert texts
    if (is_perspected){
        rm_text.insert_text(text2Din3D_list);
    }else{
        rm_text.insert_text(text2Dflat_list);
    }
}
