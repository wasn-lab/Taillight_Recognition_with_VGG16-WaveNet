#ifndef RM_LV2_TRAFFIC_LIGHT_H
#define RM_LV2_TRAFFIC_LIGHT_H

#include "rmBaseModel.h"

//
#include "rmImageBoard.h"
#include "rmText3D_v2.h"


class rmlv2TrafficLight : public rmBaseModel
{
    enum class IMAGE_ID{
        RED_ON,
        RED_OFF,
        YELLO_ON,
        YELLOW_OFF,
        GREEN_ON,
        GREEN_OFF,
        LEFT_ON,
        LEFT_OFF,
        FORWARD_ON,
        FORWARD_OFF,
        RIGHT_ON,
        RIGHT_OFF,
        COUNTER_BACKGROUNG
    };

public:
    rmlv2TrafficLight(
        std::string _path_Assets_in,
        int _ROS_topic_id_in
    );
    //
	void Update(float dt);
    void Update(ROS_INTERFACE &ros_interface);
    void Update(ROS_API &ros_api);
	void Render(std::shared_ptr<ViewManager> &_camera_ptr);
    void Reshape(const glm::ivec2 & viewport_size_in);


    // Size, position
    void setTrafficLightHeightPixel( int height_in);
    void setTrafficLightPositionCVPixel( int cv_x, int cv_y, int ref_point_mode_in=0);
    // For now, Left-up corner only
    // ref_point_mode:
    // (the position of the origin of the viewport coordinate to describe the position of the shape)
    // 0: upper-left corner
    // 1: upper-right corner
    // 2: lower-left corner
    // 3: lower-right corner


    void updateBoardGeo(){
        setupTrafficLightSize();
        setupTrafficLightPosition();
        for (size_t i=0; i < image_list.size(); ++i){
            image_list[i].updateBoardGeo();
        }
        rm_text.updateBoardGeo();
    }


protected:
    void Init();
    //
    int _ROS_topic_id;
    std::shared_ptr< msgs::Flag_Info  > msg_out_ptr;
    // ros::Time msg_time;

    // params
    int cv_lt_x; // left-top
    int cv_lt_y; // left-top
    int ref_point_mode;
    int height;


    void setupTrafficLightSize();
    void setupTrafficLightPosition();


    //
    std::vector<rmImageBoard> image_list;
    rmText3D_v2 rm_text;

    void update_GL_data( ROS_API &ros_api );


private:
    std::vector<int> light_status;
    // size: 6 (r, y, g, l, f, rt)
    // status:
    // -1 - no this light
    // 0 - off
    // 1 - on

    std::vector<rmText3D_v2::text_billboard_data> text_list;

};

#endif // RM_LV2_TRAFFIC_LIGHT_H
