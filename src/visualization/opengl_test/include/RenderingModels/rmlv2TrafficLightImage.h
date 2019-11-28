#ifndef RM_LV2_TRAFFIC_LIGHT_IMAGE_H
#define RM_LV2_TRAFFIC_LIGHT_IMAGE_H

#include "rmBaseModel.h"

//
#include "rmColorBoard.h"
#include "rmText3D_v2.h"
#include "rmImageArray.h"


class rmlv2TrafficLightImage : public rmBaseModel
{
public:
    rmlv2TrafficLightImage(
        std::string _path_Assets_in,
        int _ROS_topic_id_in,
        std::string frame_id_in="",
        glm::vec4 color_vec4_in=glm::vec4(0.3f,0.3f,0.3f,0.3f),
        bool is_perspected_in=false,
        bool is_moveable_in=true
    );
    //
	void Update(float dt);
    void Update(ROS_INTERFACE &ros_interface);
    void Update(ROS_API &ros_api);
	void Render(std::shared_ptr<ViewManager> &_camera_ptr);
    void Reshape(const glm::ivec2 & viewport_size_in);


    bool is_light_online;


protected:
    void Init();
    //
    int _ROS_topic_id;
    std::shared_ptr< msgs::Flag_Info > msg_out_ptr;
    // ros::Time msg_time;
    // std::string _frame_id;

    void _put_text(int light_status, int light_CD, bool is_enabled);

    //
    rmColorBoard rm_board;
    rmText3D_v2 rm_text;
    rmImageArray rm_image_word;

    void update_GL_data();

private:

    std::vector<rmText3D_v2::text2Dflat_data> text2D_flat_list;
    std::vector<rmImageArray::text2Dflat_data> image_flat_list;

};

#endif // RM_LV2_TRAFFIC_LIGHT_IMAGE_H
