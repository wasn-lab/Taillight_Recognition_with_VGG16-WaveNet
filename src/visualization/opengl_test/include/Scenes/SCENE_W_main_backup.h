#ifndef SCENE_W_MAIN_H
#define SCENE_W_MAIN_H

#include "Scene.h"

//

class SCENE_W_main : public Scene
{
public:
	SCENE_W_main(std::string pkg_path_in);

    // Interaction events
    void ROSTopicEvent(ROS_API &ros_api);

private:
    inline static bool cal_viewport_w(int w, int h, int &cx, int &cy, int &vw, int &vh){
        double asp = 1.5833333333;
        int im_w = w/7;
        int im_h = int(im_w/asp);
        cx = 0;
        cy = im_h;
        vw = w;
        vh = h-im_h;
        return true;
    }

};


SCENE_W_main::SCENE_W_main(std::string pkg_path_in)
{
	_camera_ptr.reset(new ViewManager());
    _camera_ptr->assign_cal_viewport(&cal_viewport_w);

    _pkg_path = (pkg_path_in);
    _Assets_path = (pkg_path_in + "Assets/");

    // Grid
    std::shared_ptr<rmGrid> _rmGrid_ptr;
    // Image
    std::shared_ptr<rmImageBoard> _image_board_ptr;
    // PointCloud
    std::shared_ptr<rmPointCloud> pc_ptr_1;
    // Text
    std::shared_ptr<rmText2D> _text2D_ptr;
    std::shared_ptr<rmText3D_v2> _text3D_ptr;
    // Bounding box 2D
    std::shared_ptr<rmlv2TagBoundingBox2D> _box2D_ptr;
    // rmlv2PathPlanFake
    std::shared_ptr<rmlv2PathPlanFake> _fake_path_ptr;
    // rmColorBoard
    std::shared_ptr<rmColorBoard> _color_board_ptr;

    /*
    // Back ground image (static)
    _image_board_ptr.reset(new rmImageBoard(_Assets_path, "view_3.jpg", false, false, true) );
    _image_board_ptr->alpha = 1.0;
    _image_board_ptr->color_transform = glm::vec4(1.0f);
    _rm_BaseModel.push_back( _image_board_ptr );
    */

    /*
    // Back ground image (dynamic) front-center camera
    _image_board_ptr.reset(new rmImageBoard(_Assets_path, int(MSG_ID::camera_front_center), false, false, true) );
    _image_board_ptr->alpha = 1.0;
    _image_board_ptr->color_transform = glm::vec4(1.0f);
    _rm_BaseModel.push_back( _image_board_ptr );
    // Bounding box for front-center camera
    _box2D_ptr.reset(new rmlv2TagBoundingBox2D(_Assets_path, int(MSG_ID::bounding_box_image_front_all), false, false ) );
    _box2D_ptr->setup_params(608, 384, 608*1, 0);
    // _box2D_ptr->alpha = 0.7;
    _rm_BaseModel.push_back( _box2D_ptr );
    */


    /*
    // Top-level top-centered back image (dynamic) <-- "Rear-sight mirror"
    _image_board_ptr.reset(new rmImageBoard(_Assets_path, int(MSG_ID::camera_rear_center), false, true, true) );
    _image_board_ptr->alpha = 1.0;
    _image_board_ptr->color_transform = glm::vec4(1.0f);
    _image_board_ptr->Rotate(glm::vec3(0.0f,1.0f,0.0f), M_PI); // Flip vertically
    // _image_board_ptr->Rotate(glm::vec3(0.0f,0.0f,1.0f), M_PI/6.0);
    // _image_board_ptr->shape.setBoardSizeRatio(0.2f, false);
    _image_board_ptr->shape.setBoardSizePixel(100, false);
    _image_board_ptr->shape.setBoardPositionCVPixel(0, 0, 1, ALIGN_X::RIGHT, ALIGN_Y::TOP );
    _rm_BaseModel.push_back( _image_board_ptr );
    */



    // Grid ground
    _rmGrid_ptr.reset(new rmGrid(_Assets_path, "map", "base" ) );
    _rmGrid_ptr->set_grid_param(1.0, 1.0, 10, 10, -6.0f, false);
    _rm_BaseModel.push_back( _rmGrid_ptr );
    /*
    // Grid local
    _rmGrid_ptr.reset(new rmGrid(_Assets_path, "base", "base" ) );
    _rmGrid_ptr->set_grid_param(1.0, 1.0, 10, 10, -3.0f, false, glm::vec3(0.2f, 0.2f, 0.5f));
    _rm_BaseModel.push_back( _rmGrid_ptr );
    */

    /*
    // rmModelLoaderObj
	std::shared_ptr<rmModelLoaderObj> bottle( new rmModelLoaderObj(_Assets_path, "Potion_bottle.obj", "bottle_mana.png") );
	std::shared_ptr<rmModelLoaderObj> box( new rmModelLoaderObj(_Assets_path, "box_realistic.obj", "box_texture_color.png") );
	bottle->Scale(glm::vec3(0.01, 0.01, 0.01));
	bottle->Rotate(glm::vec3(1, 0, 0), 3.1415926 / 2 * 3);
	bottle->Translate(glm::vec3(0.0, 0.5, 0.0));
	_rm_BaseModel.push_back(bottle);
	_rm_BaseModel.push_back(box);
    */



    // Map
    pc_ptr_1.reset(new rmPointCloud(_Assets_path, int(MSG_ID::point_cloud_map)) );
    pc_ptr_1->set_color(glm::vec3(0.68627451f, 0.0f, 0.76862745f));
    _rm_BaseModel.push_back( pc_ptr_1 );
    // Raw data
    pc_ptr_1.reset(new rmPointCloud(_Assets_path, int(MSG_ID::point_cloud_raw)) );
    pc_ptr_1->set_color(glm::vec3(1.0f));
    _rm_BaseModel.push_back( pc_ptr_1 );



    // rmlv2ObjectTracking
    _rm_BaseModel.push_back( std::shared_ptr<rmlv2ObjectTracking>(new rmlv2ObjectTracking(_Assets_path, int(MSG_ID::lidar_bounding_box_tracking), "map") ) );
    // Taged Lidar bounding box (tracking, rendering in wire)
    _rm_BaseModel.push_back( std::shared_ptr<rmlv2TagBoundingBox3D>(new rmlv2TagBoundingBox3D(_Assets_path, int(MSG_ID::lidar_bounding_box_tracking)) ) );

    // Lidar bounding box (rendering in face)
    // _rm_BaseModel.push_back( std::shared_ptr<rmLidarBoundingBox>(new rmLidarBoundingBox(_Assets_path, int(MSG_ID::lidar_bounding_box_raw)) ) );


    // Bounding box 2D
    // _rm_BaseModel.push_back( std::shared_ptr<rmlv2TagBoundingBox2D>(new rmlv2TagBoundingBox2D(_Assets_path, int(MSG_ID::bounding_box_image_front_all)) ) );

    // Sweeping object
    // _rm_BaseModel.push_back( std::shared_ptr<rmSweepingObject>(new rmSweepingObject(_Assets_path, "base" ) ) );

    // rmlv2PathPlanFake
    _fake_path_ptr.reset(   new rmlv2PathPlanFake(_Assets_path, int(MSG_ID::vehicle_info) )   );
    _fake_path_ptr->Translate(glm::vec3(-5.5f, 0.0f, 0.0f));
    _rm_BaseModel.push_back( _fake_path_ptr );


    // Circle
    // _rm_BaseModel.push_back( std::shared_ptr<rmCircle>(new rmCircle(_Assets_path, "base" ) ) );
    // rmPolyLines3D
    // _rm_BaseModel.push_back( std::shared_ptr<rmPolyLines3D>(new rmPolyLines3D(_Assets_path, "base") ) );



    /*
    // static image
    _image_board_ptr.reset(new rmImageBoard(_Assets_path, "clownfish4.png", true, true, false) );
    _image_board_ptr->Translate(glm::vec3(5.0f, 0.0f, 3.0f));
    _image_board_ptr->Rotate(glm::vec3(0.0f,0.0f,1.0f), M_PI); // Flip
    _image_board_ptr->Rotate(glm::vec3(1.0f,0.0f,0.0f), M_PI/2.0);
    _image_board_ptr->Rotate(glm::vec3(0.0f,1.0f,0.0f), M_PI/2.0);
    _image_board_ptr->shape.setBoardSize(11.08, true);
    // _image_board_ptr->Scale( glm::vec3(3.5f));
    // _image_board_ptr->Scale( glm::vec3(4.0f/3.0f, 1.0f, 1.0f));
    _rm_BaseModel.push_back( _image_board_ptr );
    */


    // Dynamic image, front-center camera
    _image_board_ptr.reset(new rmImageBoard(_Assets_path, int(MSG_ID::camera_front_center), true, true, false) );
    _image_board_ptr->Translate(glm::vec3(2.77f, 0.0f, 3.0f));
    // _image_board_ptr->Rotate(glm::vec3(0.0f,0.0f,1.0f), 0.0); // view angle
    _image_board_ptr->Rotate(glm::vec3(0.0f,0.0f,1.0f), M_PI); // Flip
    _image_board_ptr->Rotate(glm::vec3(1.0f,0.0f,0.0f), M_PI/2.0);
    _image_board_ptr->Rotate(glm::vec3(0.0f,1.0f,0.0f), M_PI/2.0);
    _image_board_ptr->shape.setBoardSize(11.08, true);
    // _image_board_ptr->Scale( glm::vec3(3.5f));
    // _image_board_ptr->Scale( glm::vec3(4.0f/3.0f, 1.0f, 1.0f));
    _image_board_ptr->alpha = 0.7;
    _rm_BaseModel.push_back( _image_board_ptr );
    // Bounding box for front-center camera
    _box2D_ptr.reset(new rmlv2TagBoundingBox2D(_Assets_path, int(MSG_ID::bounding_box_image_front_all), true, true ) );
    _box2D_ptr->setup_params(608, 384, 608*1, 0);
    _box2D_ptr->Translate(glm::vec3(2.77f, 0.0f, 3.0f));
    // _box2D_ptr->Rotate(glm::vec3(0.0f,0.0f,1.0f), 0.0); // view angle
    _box2D_ptr->Rotate(glm::vec3(0.0f,0.0f,1.0f), M_PI); // Flip
    _box2D_ptr->Rotate(glm::vec3(1.0f,0.0f,0.0f), M_PI/2.0);
    _box2D_ptr->Rotate(glm::vec3(0.0f,1.0f,0.0f), M_PI/2.0);
    _box2D_ptr->shape.setBoardSize(11.08, true);
    // _box2D_ptr->Scale( glm::vec3(3.5f));
    // _box2D_ptr->Scale( glm::vec3(4.0f/3.0f, 1.0f, 1.0f));
    // _box2D_ptr->alpha = 0.7;
    _rm_BaseModel.push_back( _box2D_ptr );


    // Dynamic image, front-right camera
    _image_board_ptr.reset(new rmImageBoard(_Assets_path, int(MSG_ID::camera_front_right), true, true, false) );
    _image_board_ptr->Translate(glm::vec3(0.0f, -10.33f, 3.0f));
    _image_board_ptr->Rotate(glm::vec3(0.0f,0.0f,1.0f), -M_PI/6.0); // view angle
    _image_board_ptr->Rotate(glm::vec3(0.0f,0.0f,1.0f), M_PI); // Flip
    _image_board_ptr->Rotate(glm::vec3(1.0f,0.0f,0.0f), M_PI/2.0);
    _image_board_ptr->Rotate(glm::vec3(0.0f,1.0f,0.0f), M_PI/2.0);
    _image_board_ptr->shape.setBoardSize(11.08, true);
    // _image_board_ptr->Scale( glm::vec3(3.5f));
    // _image_board_ptr->Scale( glm::vec3(4.0f/3.0f, 1.0f, 1.0f));
    _image_board_ptr->alpha = 0.7;
    _rm_BaseModel.push_back( _image_board_ptr );
    // Bounding box for front-right camera
    _box2D_ptr.reset(new rmlv2TagBoundingBox2D(_Assets_path, int(MSG_ID::bounding_box_image_front_all), true, true ) );
    _box2D_ptr->setup_params(608, 384, 608*2, 0);
    _box2D_ptr->Translate(glm::vec3(0.0f, -10.33f, 3.0f));
    _box2D_ptr->Rotate(glm::vec3(0.0f,0.0f,1.0f), -M_PI/6.0); // view angle
    _box2D_ptr->Rotate(glm::vec3(0.0f,0.0f,1.0f), M_PI); // Flip
    _box2D_ptr->Rotate(glm::vec3(1.0f,0.0f,0.0f), M_PI/2.0);
    _box2D_ptr->Rotate(glm::vec3(0.0f,1.0f,0.0f), M_PI/2.0);
    _box2D_ptr->shape.setBoardSize(11.08, true);
    // _box2D_ptr->Scale( glm::vec3(3.5f));
    // _box2D_ptr->Scale( glm::vec3(4.0f/3.0f, 1.0f, 1.0f));
    // _box2D_ptr->alpha = 0.7;
    _rm_BaseModel.push_back( _box2D_ptr );


    // Dynamic image, front-left camera
    _image_board_ptr.reset(new rmImageBoard(_Assets_path, int(MSG_ID::camera_front_left), true, true, false) );
    _image_board_ptr->Translate(glm::vec3(0.0f, 10.33f, 3.0f));
    _image_board_ptr->Rotate(glm::vec3(0.0f,0.0f,1.0f), M_PI/6.0); // view angle
    _image_board_ptr->Rotate(glm::vec3(0.0f,0.0f,1.0f), M_PI); // Flip
    _image_board_ptr->Rotate(glm::vec3(1.0f,0.0f,0.0f), M_PI/2.0);
    _image_board_ptr->Rotate(glm::vec3(0.0f,1.0f,0.0f), M_PI/2.0);
    _image_board_ptr->shape.setBoardSize(11.08, true);
    // _image_board_ptr->Scale( glm::vec3(3.5f));
    // _image_board_ptr->Scale( glm::vec3(4.0f/3.0f, 1.0f, 1.0f));
    _image_board_ptr->alpha = 0.7;
    _rm_BaseModel.push_back( _image_board_ptr );
    // Bounding box for front-left camera
    _box2D_ptr.reset(new rmlv2TagBoundingBox2D(_Assets_path, int(MSG_ID::bounding_box_image_front_all), true, true ) );
    _box2D_ptr->setup_params(608, 384, 608*0, 0);
    _box2D_ptr->Translate(glm::vec3(0.0f, 10.33f, 3.0f));
    _box2D_ptr->Rotate(glm::vec3(0.0f,0.0f,1.0f), M_PI/6.0); // view angle
    _box2D_ptr->Rotate(glm::vec3(0.0f,0.0f,1.0f), M_PI); // Flip
    _box2D_ptr->Rotate(glm::vec3(1.0f,0.0f,0.0f), M_PI/2.0);
    _box2D_ptr->Rotate(glm::vec3(0.0f,1.0f,0.0f), M_PI/2.0);
    _box2D_ptr->shape.setBoardSize(11.08, true);
    // _box2D_ptr->Scale( glm::vec3(3.5f));
    // _box2D_ptr->Scale( glm::vec3(4.0f/3.0f, 1.0f, 1.0f));
    // _box2D_ptr->alpha = 0.7;
    _rm_BaseModel.push_back( _box2D_ptr );



    // Dynamic image, front-down camera
    _image_board_ptr.reset(new rmImageBoard(_Assets_path, int(MSG_ID::camera_front_top), true, true, false) );
    _image_board_ptr->Translate(glm::vec3(-3.0f, 0.0f, 8.0f));
    _image_board_ptr->Rotate(glm::vec3(0.0f,1.0f,0.0f), -M_PI/6.0); // view angle
    _image_board_ptr->Rotate(glm::vec3(0.0f,0.0f,1.0f), M_PI); // Flip
    _image_board_ptr->Rotate(glm::vec3(1.0f,0.0f,0.0f), M_PI/2.0);
    _image_board_ptr->Rotate(glm::vec3(0.0f,1.0f,0.0f), M_PI/2.0);
    _image_board_ptr->shape.setBoardSize(11.08, true);
    // _image_board_ptr->Scale( glm::vec3(3.5f));
    // _image_board_ptr->Scale( glm::vec3(4.0f/3.0f, 1.0f, 1.0f));
    _image_board_ptr->alpha = 0.7;
    _rm_BaseModel.push_back( _image_board_ptr );
    // Bounding box for front-left camera
    _box2D_ptr.reset(new rmlv2TagBoundingBox2D(_Assets_path, int(MSG_ID::bounding_box_image_front_top), true, true ) );
    _box2D_ptr->setup_params(608, 384, 608*0, 0);
    _box2D_ptr->Translate(glm::vec3(-3.0f, 0.0f, 8.0f));
    _box2D_ptr->Rotate(glm::vec3(0.0f,1.0f,0.0f), -M_PI/6.0); // view angle
    _box2D_ptr->Rotate(glm::vec3(0.0f,0.0f,1.0f), M_PI); // Flip
    _box2D_ptr->Rotate(glm::vec3(1.0f,0.0f,0.0f), M_PI/2.0);
    _box2D_ptr->Rotate(glm::vec3(0.0f,1.0f,0.0f), M_PI/2.0);
    _box2D_ptr->shape.setBoardSize(11.08, true);
    // _box2D_ptr->Scale( glm::vec3(3.5f));
    // _box2D_ptr->Scale( glm::vec3(4.0f/3.0f, 1.0f, 1.0f));
    // _box2D_ptr->alpha = 0.7;
    _rm_BaseModel.push_back( _box2D_ptr );



    // _text3D_ptr.reset( new rmText3D_v2(_Assets_path, "base" ) );
    // _text3D_ptr->Translate(glm::vec3(1.0f, -2.0f, -2.0f));
    // _text3D_ptr->Rotate(glm::vec3(0.0f,0.0f,1.0f), M_PI); // Flip
    // _text3D_ptr->Rotate(glm::vec3(1.0f,0.0f,0.0f), M_PI/2.0);
    // _text3D_ptr->Rotate(glm::vec3(0.0f,1.0f,0.0f), M_PI/2.0);
    // // _text3D_ptr->setBoardSizeRatio(0.5, true);
    // // _text3D_ptr->Translate(glm::vec3(-0.5f, -0.5f, 0.0f));
    // // _text3D_ptr->Scale( glm::vec3(3.5f));
    // // _text3D_ptr->Scale( glm::vec3(4.0f/3.0f, 1.0f, 1.0f));
    // _rm_BaseModel.push_back( _text3D_ptr );


    /*
    _text2D_ptr.reset( new rmText2D() );
    _rm_BaseModel.push_back( _text2D_ptr );
    */


    // test, rmText3D:    400 x "Hello world" --> CPU 104%, GPU 85%
    // test, rmText3D_v2: 400 x "Hello world" --> CPU 60%, GPU 57%
    /*
    for (size_t i=0; i < 400; ++i){
        std::cout << "i = " << i << "\n";
        _text3D_ptr.reset( new rmText3D_v2(_Assets_path ) );
        _text3D_ptr->Translate(glm::vec3(0.0f, 0.0f, (6.0f + 1.0f*i) ));
        _rm_BaseModel.push_back( _text3D_ptr );
    }
    */


    /*
    // rmColorBoard
    _color_board_ptr.reset( new rmColorBoard(_Assets_path, "base", glm::vec4(0.0f, 0.2f, 0.8f, 0.5f), false, true ) );
    _color_board_ptr->shape.setBoardSizePixel(150, 100);
    _color_board_ptr->shape.setBoardPositionCVPixel(-10,10,1,ALIGN_X::RIGHT, ALIGN_Y::TOP );
    _rm_BaseModel.push_back( _color_board_ptr );
    */


    // rmlv2SpeedMeter
    _rm_BaseModel.push_back( std::shared_ptr<rmlv2SpeedMeter>( new rmlv2SpeedMeter(_Assets_path, int(MSG_ID::vehicle_info) ) ) );


}





// Interaction events
//------------------------------------------------------//
void SCENE_W_main::ROSTopicEvent(ROS_API &ros_api){
    std::shared_ptr< opengl_test::GUI2_op > _GUI2_op_ptr;
    bool result = ros_api.get_message( int(MSG_ID::GUI_operatio), _GUI2_op_ptr);
    if (!result){
        return;
    }
    // else
    std::cout << "---\n";
    std::cout << "cam_type: " << _GUI2_op_ptr->cam_type << "\n";
    std::cout << "image3D: " << _GUI2_op_ptr->image3D << "\n";
    std::cout << "image_surr: " << _GUI2_op_ptr->image_surr << "\n";
    std::cout << "cam_motion: " << _GUI2_op_ptr->cam_motion << "\n";

    // State
    //----------------------------------------//
    // cam_type
    if (_GUI2_op_ptr->cam_type == "follow"){
        switchCameraMode( 0, ros_api);
    }else if (_GUI2_op_ptr->cam_type == "static"){
        switchCameraMode( 1, ros_api);
    }else if (_GUI2_op_ptr->cam_type == "toggle_cam"){
        KeyBoardEvent('c', ros_api);
    }
    // image3D
    if (_GUI2_op_ptr->image3D == "on"){
        //
    }else if (_GUI2_op_ptr->image3D == "off"){
        //
    }
    // image_surr
    if (_GUI2_op_ptr->image_surr == "on"){
        //
    }else if (_GUI2_op_ptr->image_surr == "off"){
        //
    }
    //----------------------------------------//
    // Event
    //----------------------------------------//
    // cam_motion
    if (_GUI2_op_ptr->cam_motion == "reset"){
        KeyBoardEvent('z', ros_api);
    }else if (_GUI2_op_ptr->cam_motion == "zoom_in"){
        KeyBoardEvent('w', ros_api);
    }else if (_GUI2_op_ptr->cam_motion == "zoom_out"){
        KeyBoardEvent('s', ros_api);
    }
    //----------------------------------------//

    //Response
    //----------------------------------------//
    opengl_test::GUI2_op res_data;
    // cam_type
    if (camera_mode == 0)
        res_data.cam_type = "follow";
    else if (camera_mode == 1)
        res_data.cam_type = "static";
    // image3D
    res_data.image3D = "on";
    // image_surr
    res_data.image_surr = "on";
    //----------------------------------------//
    //
    ros_api.ros_interface.send_GUI2_op( int(MSG_ID::GUI_operatio), res_data);
}

#endif  // SCENE_W_MAIN_H
