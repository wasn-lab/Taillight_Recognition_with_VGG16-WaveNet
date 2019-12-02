#ifndef RM_LV2_SPEED_METER_H
#define RM_LV2_SPEED_METER_H

#include "rmBaseModel.h"

//
#include "rmColorBoard.h"
#include "rmText3D_v2.h"


class rmlv2SpeedMeter : public rmBaseModel
{
public:
    rmlv2SpeedMeter(
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


protected:
    void Init();
    //
    int _ROS_topic_id;
    // std::shared_ptr< msgs::LidRoi > msg_out_ptr;
    std::shared_ptr< msgs::VehInfo > msg_out_ptr;
    // ros::Time msg_time;
    // std::string _frame_id;

    void _put_text(float speed, bool is_enabled);

    //
    rmColorBoard rm_board;
    rmText3D_v2 rm_text;

    void update_GL_data();

private:

    std::vector<rmText3D_v2::text2Dflat_data> text2D_flat_list;

};

#endif // RM_LV2_SPEED_METER_H
