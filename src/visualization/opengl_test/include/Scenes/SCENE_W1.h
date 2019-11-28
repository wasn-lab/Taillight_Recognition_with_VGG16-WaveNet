#ifndef SCENE_W1_H
#define SCENE_W1_H

#include "Scene.h"

// Version control
//----------------------------------------//
#include "GUI_version_control.h"
//----------------------------------------//

class SCENE_W1 : public Scene
{
public:
	SCENE_W1(std::string pkg_path_in);

private:
    inline static bool cal_viewport_w(int w, int h, int &cx, int &cy, int &vw, int &vh){
        double asp = _IMAGE_ASP_;
        int im_w = w/7;
        int im_h = int(im_w/asp);
        cx = im_w*1;
        cy = 0;
        vw = im_w;
        vh = im_h;
        return true;
    }
};



SCENE_W1::SCENE_W1(std::string pkg_path_in)
{
	_camera_ptr.reset(new ViewManager());
    // _camera_ptr->assign_cal_viewport(&cal_viewport_w);
    // Layout
    //----------------------------------------//
    attach_cal_viewport_func_ptr(0, &cal_viewport_w);
    switch_layout(0);
    //----------------------------------------//

    _pkg_path = (pkg_path_in);
    _Assets_path = (pkg_path_in + "Assets/");


    // Bounding box 2D
    std::shared_ptr<rmBoundingBox2D> _box2D_ptr;


    // Back ground image rmImageDynamicBackground
    // std::shared_ptr<rmImageDynamicBackground> _image_background_2_ptr(new rmImageDynamicBackground(_Assets_path, int(MSG_ID::camera_2)) );
    std::shared_ptr<rmImageBoard> _image_background_2_ptr(new rmImageBoard(_Assets_path, int(MSG_ID::camera_left_fore), false, false, true) );
    _image_background_2_ptr->alpha = 1.0;
    _image_background_2_ptr->color_transform = glm::vec4(1.0f);
    _rm_BaseModel.push_back( _image_background_2_ptr );

#if __ROS_INTERFACE_VER__ == 2
    // Bounding box for front-right camera
    _box2D_ptr.reset(new rmBoundingBox2D(_Assets_path, int(MSG_ID::bounding_box_image_left_fore), false, false ) );
    _box2D_ptr->setup_params(_IMAGE_W_, _IMAGE_H_, _IMAGE_W_*0, 0);
    // _box2D_ptr->alpha = 0.7;
    _rm_BaseModel.push_back( _box2D_ptr );
#endif  // __ROS_INTERFACE_VER__

}

#endif  // SCENE_W1_H
