#ifndef RM_LV2_PATH_PLAN_V1_H
#define RM_LV2_PATH_PLAN_V1_H

#include "rmBaseModel.h"

//
#include "rmSweepingObject.h"
#include "rmText3D_v2.h"


class rmlv2PathPlan_v1 : public rmBaseModel
{
public:
    rmlv2PathPlan_v1(
        std::string _path_Assets_in,
        int _ROS_topic_id_in,
        std::string data_representation_frame_in="map"
    );
    //
	void Update(float dt);
    void Update(ROS_INTERFACE &ros_interface);
    void Update(ROS_API &ros_api);
	void Render(std::shared_ptr<ViewManager> &_camera_ptr);


protected:
    void Init();
    //
    int _ROS_topic_id;
    // std::shared_ptr< msgs::VehInfo  > msg_out_ptr;
    std::shared_ptr< msgs::DynamicPath  > msg_out_ptr;
    // ros::Time msg_time;

    // The reference frame of the data in message
    std::string data_representation_frame;

    //
    rmSweepingObject rm_path;
    rmText3D_v2 rm_text;

    void update_GL_data( ROS_API &ros_api );

    //
    void get_point3D_poly(const std::vector<glm::vec2> &param, double dT, glm::vec3 &point3D_out);

    // Param
    float _sim_time; // sec.
    glm::vec2 _granularity; // 20 cm, 5 deg.
    int _max_sim_point;

private:
    std::vector<glm::vec3> section_vertexes;
    std::vector<glm::vec3> _path;
    std::vector<rmText3D_v2::text_billboard_data> text_list;

};

#endif // RM_LV2_PATH_PLAN_V1_H
