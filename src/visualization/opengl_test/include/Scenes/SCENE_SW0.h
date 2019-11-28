#ifndef SCENE_SW0_H
#define SCENE_SW0_H

#include "Scene.h"

// Version control
//----------------------------------------//
#include "GUI_version_control.h"
//----------------------------------------//

class SCENE_SW0 : public Scene
{
public:
	SCENE_SW0(std::string pkg_path_in);

private:
    inline static bool cal_viewport_w_2(int w, int h, int &cx, int &cy, int &vw, int &vh){
        // Surrounding cameras queued as "side bars"
        double asp = _IMAGE_ASP_;
        int im_w = h/2;
        int im_h = int(im_w/asp);
        cx = 0;
        cy = 0;
        vw = im_h;
        vh = im_w;
        return true;
    }
};



SCENE_SW0::SCENE_SW0(std::string pkg_path_in)
{
	_camera_ptr.reset(new ViewManager());
    // _camera_ptr->assign_cal_viewport(&cal_viewport_w);
    // Layout
    //----------------------------------------//
    attach_cal_viewport_func_ptr(2, &cal_viewport_w_2);
    switch_layout(0);
    //----------------------------------------//

    _pkg_path = (pkg_path_in);
    _Assets_path = (pkg_path_in + "Assets/");


    // Bounding box 2D
    std::shared_ptr<rmBoundingBox2D> _box2D_ptr;


    // Back ground image rmImageDynamicBackground
    std::shared_ptr<rmImageBoard> _image_background_2_ptr(new rmImageBoard(_Assets_path, int(MSG_ID::camera_left_rear), false, true, true) );
    _image_background_2_ptr->alpha = 1.0;
    _image_background_2_ptr->color_transform = glm::vec4(1.0f);
    _image_background_2_ptr->Translate(glm::vec3(0.0f, 0.0f, -1.0f)); // Move to background
    _image_background_2_ptr->Rotate(glm::vec3(0.0f,0.0f,1.0f), M_PI/2.0); // Rotate
    _image_background_2_ptr->shape.setBoardSizeRatio(1.0f, 1.0f); // Full size
    _rm_BaseModel.push_back( _image_background_2_ptr );

#if __ROS_INTERFACE_VER__ == 2
    // Bounding box for front-right camera
    _box2D_ptr.reset(new rmBoundingBox2D(_Assets_path, int(MSG_ID::bounding_box_image_left_rear), false, true ) );
    _box2D_ptr->setup_params(_IMAGE_W_, _IMAGE_H_, _IMAGE_W_*0, 0);
    // _box2D_ptr->alpha = 0.7;
    _box2D_ptr->Rotate(glm::vec3(0.0f,0.0f,1.0f), M_PI/2.0); // Rotate
    _box2D_ptr->shape.setBoardSizeRatio(1.0f, 1.0f); // Full size
    _rm_BaseModel.push_back( _box2D_ptr );
#endif  // __ROS_INTERFACE_VER__

}

#endif  // SCENE_SW0_H
