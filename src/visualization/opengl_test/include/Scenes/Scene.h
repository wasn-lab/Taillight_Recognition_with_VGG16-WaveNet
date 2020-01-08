#ifndef SCENE_H
#define SCENE_H

#include <vector>
#include <string>
// Debug
#include <iostream>

//
// #include "ViewManager.h"
#include "ViewManager_v2.h"
#include <ROS_ICLU3_v0.hpp>

// Render models
#include "rmBaseModel.h"
#include "rmModelLoaderObj.h"
#include "rmPointCloud.h"
#include "rmLidarBoundingBox.h"
#include "rmImageBoard.h"
#include "rmColorBoard.h"
#include "rmBoundingBox2D.h"
#include "rmlv2TagBoundingBox2D.h"
#include "rmlv2TagBoundingBox3D.h"
#include "rmGrid.h"
#include "rmPolyLines3D.h"
#include "rmSweepingObject.h"
#include "rmText2D.h"
// #include "rmText3D.h"
#include "rmText3D_v2.h"
#include "rmCircle.h"
#include "rmlv2ObjectTracking.h"
// #include "rmlv2PathPlanFake.h"
#include "rmlv2PathPlan_v1.h"
#include "rmlv2SpeedMeter.h"
// #include "rmlv2TrafficLight.h"
#include "rmlv2TrafficLightSimple.h"
#include "rmMagicPowder.h"
#include "rmImageArray.h"
// The following are not finished yet
#include "rmlv2TrafficLightImage.h"
#include "rmlv2PathPlan_navPath.h"
//

class Scene
{
public:
    Scene();
	Scene(std::string pkg_path_in);
    //
    void enable(bool enable_in);
    void attach_cal_viewport_func_ptr(int layout_id, bool (*)( int , int ,int &, int &, int &, int &) );
    // Layout mode
    bool switch_layout(int layout_id);
    inline int get_layout_mode(){ return _layout_mode; };
    //
	void Render();
	void Update(float dt);
    void Update(ROS_INTERFACE &ros_interface);
    void Update(ROS_API &ros_api);
    void Reshape();
    void Reshape(int full_window_width, int full_window_height);

    // Interaction events
    virtual void ROSTopicEvent(ROS_API &ros_api);
    virtual void MouseEvent(int button,int state,int x,int y);
	virtual void KeyBoardEvent(int key);
	virtual void KeyBoardEvent(unsigned char key, ROS_API &ros_api);
	virtual void MenuEvent(int item);

    //
    virtual void perSceneROSTopicEvent(ROS_API &ros_api){  /*empty*/ }
    virtual void perSceneKeyBoardEvent(unsigned char key){  /*empty*/ }
    //

    // Camera mode
    int get_camera_motion_mode(){ return camera_motion_mode; }
    int get_camera_view_mode(){ return camera_view_mode; }
    // Follow, static, ...etc.
    virtual void resetDefaultCaemraModel(ROS_API &ros_api);
    virtual void switchCameraMotionMode(int mode_in, ROS_API &ros_api);
    virtual void switchCameraViewMode(int mode_in, ROS_API &ros_api);

    // ViewManager
    std::shared_ptr<ViewManager> GetCamera(){ return _camera_ptr; }

protected:
    bool is_enabled;
    bool is_initialized;
    std::string _pkg_path;
    std::string _Assets_path;
	std::shared_ptr<ViewManager> _camera_ptr;

    // camera motion
    int camera_motion_mode;
    std::string camera_ref_frame;
    // Camera view
    int camera_view_mode;

    int _layout_mode;
    std::map<int, bool (*)( int , int ,int &, int &, int &, int &)> _cal_viewport_map;

    // Render models
    std::vector< std::shared_ptr<rmBaseModel> > _rm_BaseModel;


    static bool cal_viewport_dis(int w, int h, int &cx, int &cy, int &vw, int &vh){
        cx = 0;
        cy = 0;
        vw = 1;
        vh = 0;
        return true;
    }
};

#endif  // Scene_H
