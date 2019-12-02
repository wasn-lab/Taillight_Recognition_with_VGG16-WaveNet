#include "rmlv2TrafficLightImage.h"




rmlv2TrafficLightImage::rmlv2TrafficLightImage(
    std::string _path_Assets_in,
    int _ROS_topic_id_in,
    std::string frame_id_in,
    glm::vec4 color_vec4_in,
    bool is_perspected_in,
    bool is_moveable_in
):
    _ROS_topic_id(_ROS_topic_id_in),
    is_light_online(false),
    //
    rm_board(_path_Assets_in, frame_id_in, color_vec4_in, is_perspected_in, is_moveable_in),
    rm_text(_path_Assets_in, _ROS_topic_id_in),
    rm_image_word(_path_Assets_in, _ROS_topic_id_in)
{
    //
	Init();
}
void rmlv2TrafficLightImage::Init(){

    // For adjusting the model pose by public methods
    attach_pose_model_by_model_ref_ptr( *rm_board.get_model_m_ptr() );
    attach_pose_model_by_model_ref_ptr( *rm_text.get_model_m_ptr() );
    attach_pose_model_by_model_ref_ptr( *rm_image_word.get_model_m_ptr() );


    rm_board.shape.setBoardSizePixel(280, 80);
    rm_board.shape.setBoardPositionCVPixel(-5,95,1,ALIGN_X::RIGHT, ALIGN_Y::TOP);
    rm_text.shape = rm_board.shape;
    rm_image_word.shape = rm_board.shape;

    // Enter the image file name
    std::vector<std::string> image_name_list;
    image_name_list.push_back("TFL_red_off.png");
    image_name_list.push_back("TFL_red_on.png");
    image_name_list.push_back("TFL_yello_off.png");
    image_name_list.push_back("TFL_yello_on.png");
    image_name_list.push_back("TFL_green_off.png");
    image_name_list.push_back("TFL_green_on.png");
    rm_image_word.setup_image_dictionary(image_name_list);



    // Initialized the text to display
    _put_text(0, 0, false);
    // Reset
    text2D_flat_list.clear();
    image_flat_list.clear();

}

void rmlv2TrafficLightImage::Update(float dt){
    // Update the data (buffer variables) here
}
void rmlv2TrafficLightImage::Update(ROS_INTERFACE &ros_interface){
    // Update the data (buffer variables) here
}

void rmlv2TrafficLightImage::Update(ROS_API &ros_api){
    // Update the data (buffer variables) here
    // test, use transform
    ros::Time msg_time;
    bool _result = false;
    _result = ros_api.get_message(_ROS_topic_id, msg_out_ptr, msg_time);

    if (_result){
        update_GL_data();
        // rm_image_word.insert_text();
    }

    //
    rm_board.Update(ros_api);
    rm_text.Update(ros_api);
    rm_image_word.Update(ros_api);
}


void rmlv2TrafficLightImage::Render(std::shared_ptr<ViewManager> &_camera_ptr){

    rm_board.Render(_camera_ptr);
    rm_text.Render(_camera_ptr);
    rm_image_word.Render(_camera_ptr);

}

void rmlv2TrafficLightImage::Reshape(const glm::ivec2 & viewport_size_in){
    _viewport_size = viewport_size_in;
    // updateBoardGeo();
    rm_board.Reshape(viewport_size_in);
    rm_text.shape = rm_board.shape;
    rm_text.Reshape(viewport_size_in);
    rm_image_word.shape = rm_board.shape;
    rm_image_word.Reshape(viewport_size_in);
}

void rmlv2TrafficLightImage::update_GL_data(){
    // Reset
    text2D_flat_list.clear();
    image_flat_list.clear();

    int light_status = int(msg_out_ptr->Dspace_Flag02);
    int light_CD     = int(msg_out_ptr->Dspace_Flag03);  // Count-down time

    //
    if (light_status == 0 && light_CD == 0){
        is_light_online = false;
    }else{
        is_light_online = true;
    }
    //
    _put_text(light_status, light_CD, is_light_online);

}


void rmlv2TrafficLightImage::_put_text(int light_status, int light_CD, bool is_enabled){
    //
    std::vector<int> _image_id_out {0,2,4};
    std::string _str_out;
    glm::vec3 _text_color(0.3f);

    if (is_enabled){
        //
        switch (light_status){
            case 1: // red
                _image_id_out[0] = 1;
                _text_color = glm::vec3(1.0f, 0.0f, 0.0f);
                break;
            case 3: // yello
                _image_id_out[1] = 3;
                _text_color = glm::vec3(0.897f, 0.837f, 0.0f);
                break;
            case 2: // green
                _image_id_out[2] = 5;
                _text_color = glm::vec3(0.0f, 0.886f, 0.046f);
                break;
            default:
                _str_out += "--:     ";
                break;
        }
        // Light CD
        _str_out = std::to_string(light_CD) + " s";
    }
    //

    //
    image_flat_list.emplace_back(
        rmImageArray::vec2str(_image_id_out),
        glm::vec2(float(8), float(rm_image_word.shape.board_height*0.5)),
        // glm::vec2( 0.0f, 0.0f),
        60,
        glm::vec3(1.0f),
        ALIGN_X::LEFT,
        ALIGN_Y::CENTER,
        0,
        0,
        false,
        false
    );
    //
    text2D_flat_list.emplace_back(
        _str_out,
        glm::vec2(float(200), float(rm_text.shape.board_height*0.5)),
        // glm::vec2( 0.0f, 0.0f),
        36,
        _text_color,
        ALIGN_X::LEFT,
        ALIGN_Y::CENTER,
        0,
        0,
        false,
        false
    );

    // Insert texts
    rm_text.insert_text( text2D_flat_list );
    rm_image_word.insert_text( image_flat_list );
}
