#include "rmlv2TrafficLight.h"


#include <math.h>       // ceil


rmlv2TrafficLight::rmlv2TrafficLight(
    std::string _path_Assets_in,
    int _ROS_topic_id_in
):
    _ROS_topic_id(_ROS_topic_id_in),
    //
    rm_text(_path_Assets_in, _ROS_topic_id_in)
{
    init_paths(_path_Assets_in);
    //
	Init();
}
void rmlv2TrafficLight::Init(){


    image_list.emplace_back(_path_Assets, "TFL_red_on.png", false, true, false);
    image_list.emplace_back(_path_Assets, "TFL_red_off.png", false, true, false);
    image_list.emplace_back(_path_Assets, "TFL_yello_on.png", false, true, false);
    image_list.emplace_back(_path_Assets, "TFL_yello_off.png", false, true, false);
    image_list.emplace_back(_path_Assets, "TFL_green_on.png", false, true, false);
    image_list.emplace_back(_path_Assets, "TFL_green_off.png", false, true, false);
    image_list.emplace_back(_path_Assets, "TFL_left_on.png", false, true, false);
    image_list.emplace_back(_path_Assets, "TFL_left_off.png", false, true, false);
    image_list.emplace_back(_path_Assets, "TFL_forward_on.png", false, true, false);
    image_list.emplace_back(_path_Assets, "TFL_forward_off.png", false, true, false);
    image_list.emplace_back(_path_Assets, "TFL_right_on.png", false, true, false);
    image_list.emplace_back(_path_Assets, "TFL_right_off.png", false, true, false);
    image_list.emplace_back(_path_Assets, "TFL_counter_background.png", false, true, false);

    //
    light_status.resize(6, -1);
    light_status[0] = 0; // red
    light_status[1] = 0; // yellow
    light_status[2] = 1; // green
    //

    // Change light status
    image_list[ int(IMAGE_ID::RED_ON) ].set_enable( light_status[0] == 1 );
    image_list[ int(IMAGE_ID::RED_OFF) ].set_enable( light_status[0] == 0 );
    image_list[ int(IMAGE_ID::YELLO_ON) ].set_enable( light_status[1] == 1 );
    image_list[ int(IMAGE_ID::YELLOW_OFF) ].set_enable( light_status[1] == 0 );
    image_list[ int(IMAGE_ID::GREEN_ON) ].set_enable( light_status[2] == 1 );
    image_list[ int(IMAGE_ID::GREEN_OFF) ].set_enable( light_status[2] == 0 );
    image_list[ int(IMAGE_ID::LEFT_ON) ].set_enable( light_status[3] == 1 );
    image_list[ int(IMAGE_ID::LEFT_OFF) ].set_enable( light_status[3] == 0 );
    image_list[ int(IMAGE_ID::FORWARD_ON) ].set_enable( light_status[4] == 1 );
    image_list[ int(IMAGE_ID::FORWARD_OFF) ].set_enable( light_status[4] == 0 );
    image_list[ int(IMAGE_ID::RIGHT_ON) ].set_enable( light_status[5] == 1 );
    image_list[ int(IMAGE_ID::RIGHT_OFF) ].set_enable( light_status[5] == 0 );



    // For adjusting the model pose by public methods
    for (size_t i=0; i < image_list.size(); ++i){
        attach_pose_model_by_model_ref_ptr( *(image_list[i].get_model_m_ptr()) );
    }
    attach_pose_model_by_model_ref_ptr( *rm_text.get_model_m_ptr() );

}

void rmlv2TrafficLight::Update(float dt){
    // Update the data (buffer variables) here
}
void rmlv2TrafficLight::Update(ROS_INTERFACE &ros_interface){
    // Update the data (buffer variables) here
}

void rmlv2TrafficLight::Update(ROS_API &ros_api){
    // Update the data (buffer variables) here
    // test, use transform
    ros::Time msg_time;
    bool _result = false;
    _result = ros_api.get_message(_ROS_topic_id, msg_out_ptr, msg_time);

    if (_result){
        update_GL_data( ros_api );
    }


    // for (size_t i=0; i < image_list.size(); ++i){
    //     image_list.Update(ros_api);
    // }
    rm_text.Update(ros_api);
}
void rmlv2TrafficLight::Reshape(const glm::ivec2 & viewport_size_in){
    _viewport_size = viewport_size_in;
    updateBoardGeo();

    for (size_t i=0; i < image_list.size(); ++i){
        image_list[i].Reshape(_viewport_size);
    }
    rm_text.Reshape(_viewport_size);
}

void rmlv2TrafficLight::Render(std::shared_ptr<ViewManager> &_camera_ptr){

    for (size_t i=0; i < image_list.size(); ++i){
        if ( image_list[i].get_enable() )
            image_list[i].Render(_camera_ptr);
    }
    rm_text.Render(_camera_ptr);
}

void rmlv2TrafficLight::update_GL_data( ROS_API &ros_api ){


    // // Reset
    // text_list.clear();
    //
    // // Insert texts
    // rm_text.insert_text(text_list);

}

// Size, position
void rmlv2TrafficLight::setTrafficLightHeightPixel( int height_in){
    height = height_in;
    updateBoardGeo();
}
void rmlv2TrafficLight::setTrafficLightPositionCVPixel(
    int cv_x,
    int cv_y,
    int ref_point_mode_in
){ // For now, Left-up corner only
    cv_lt_x = cv_x;
    cv_lt_y = cv_y;
    ref_point_mode = ref_point_mode_in;
    updateBoardGeo();
}




void rmlv2TrafficLight::setupTrafficLightSize(){

    // From left to right
    // Red
    image_list[ int(IMAGE_ID::RED_ON) ].shape.setBoardSizePixel(height, false);
    image_list[ int(IMAGE_ID::RED_OFF) ].shape.setBoardSizePixel(height, false);
    // Yello
    image_list[ int(IMAGE_ID::YELLO_ON) ].shape.setBoardSizePixel(height, false);
    image_list[ int(IMAGE_ID::YELLOW_OFF) ].shape.setBoardSizePixel(height, false);
    // Green
    image_list[ int(IMAGE_ID::GREEN_ON) ].shape.setBoardSizePixel(height, false);
    image_list[ int(IMAGE_ID::GREEN_OFF) ].shape.setBoardSizePixel(height, false);
    // Left
    image_list[ int(IMAGE_ID::LEFT_ON) ].shape.setBoardSizePixel(height, false);
    image_list[ int(IMAGE_ID::LEFT_OFF) ].shape.setBoardSizePixel(height, false);
    // Forward
    image_list[ int(IMAGE_ID::FORWARD_ON) ].shape.setBoardSizePixel(height, false);
    image_list[ int(IMAGE_ID::FORWARD_OFF) ].shape.setBoardSizePixel(height, false);
    // Right
    image_list[ int(IMAGE_ID::RIGHT_ON) ].shape.setBoardSizePixel(height, false);
    image_list[ int(IMAGE_ID::RIGHT_OFF) ].shape.setBoardSizePixel(height, false);
    // Counter background
    image_list[ int(IMAGE_ID::COUNTER_BACKGROUNG) ].shape.setBoardSizePixel(height, false);
}
void rmlv2TrafficLight::setupTrafficLightPosition(){
    std::cout << "here\n";
    ALIGN_X al_x = ALIGN_X::LEFT;
    ALIGN_Y al_y = ALIGN_Y::TOP;
    int cv_x = cv_lt_x;
    int cv_y = cv_lt_y;
    // From left to right
    // Red
    image_list[ int(IMAGE_ID::RED_ON) ].shape.setBoardPositionCVPixel(cv_x, cv_y, ref_point_mode, al_x, al_y);
    image_list[ int(IMAGE_ID::RED_OFF) ].shape.setBoardPositionCVPixel(cv_x, cv_y, ref_point_mode, al_x, al_y);
    cv_x += image_list[ int(IMAGE_ID::RED_ON) ].shape.board_width;
    // Yello
    image_list[ int(IMAGE_ID::YELLO_ON) ].shape.setBoardPositionCVPixel(cv_x, cv_y, ref_point_mode, al_x, al_y);
    image_list[ int(IMAGE_ID::YELLOW_OFF) ].shape.setBoardPositionCVPixel(cv_x, cv_y, ref_point_mode, al_x, al_y);
    cv_x += image_list[ int(IMAGE_ID::YELLO_ON) ].shape.board_width;
    // Green
    image_list[ int(IMAGE_ID::GREEN_ON) ].shape.setBoardPositionCVPixel(cv_x, cv_y, ref_point_mode, al_x, al_y);
    image_list[ int(IMAGE_ID::GREEN_OFF) ].shape.setBoardPositionCVPixel(cv_x, cv_y, ref_point_mode, al_x, al_y);
    cv_x += image_list[ int(IMAGE_ID::GREEN_ON) ].shape.board_width;
    // Left
    image_list[ int(IMAGE_ID::LEFT_ON) ].shape.setBoardPositionCVPixel(cv_x, cv_y, ref_point_mode, al_x, al_y);
    image_list[ int(IMAGE_ID::LEFT_OFF) ].shape.setBoardPositionCVPixel(cv_x, cv_y, ref_point_mode, al_x, al_y);
    cv_x += image_list[ int(IMAGE_ID::LEFT_ON) ].shape.board_width;
    // Forward
    image_list[ int(IMAGE_ID::FORWARD_ON) ].shape.setBoardPositionCVPixel(cv_x, cv_y, ref_point_mode, al_x, al_y);
    image_list[ int(IMAGE_ID::FORWARD_OFF) ].shape.setBoardPositionCVPixel(cv_x, cv_y, ref_point_mode, al_x, al_y);
    cv_x += image_list[ int(IMAGE_ID::FORWARD_ON) ].shape.board_width;
    // Right
    image_list[ int(IMAGE_ID::RIGHT_ON) ].shape.setBoardPositionCVPixel(cv_x, cv_y, ref_point_mode, al_x, al_y);
    image_list[ int(IMAGE_ID::RIGHT_OFF) ].shape.setBoardPositionCVPixel(cv_x, cv_y, ref_point_mode, al_x, al_y);
    cv_x += image_list[ int(IMAGE_ID::RIGHT_ON) ].shape.board_width;
    // Counter background
    image_list[ int(IMAGE_ID::COUNTER_BACKGROUNG) ].shape.setBoardPositionCVPixel(cv_x, cv_y, ref_point_mode, al_x, al_y);
}
