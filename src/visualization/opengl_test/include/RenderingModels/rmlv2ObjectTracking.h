#ifndef RM_LV2_OBJECT_TRACKING_H
#define RM_LV2_OBJECT_TRACKING_H

#include "rmBaseModel.h"

#include <queue>
#include <map>

//
#include "rmPolyLines3D.h"
#include "rmCircle.h"
#include "rmText3D_v2.h"


class rmlv2ObjectTracking : public rmBaseModel
{
public:
    rmlv2ObjectTracking(
        std::string _path_Assets_in,
        int _ROS_topic_id_in,
        std::string ref_frame_in
    );
    //
	void Update(float dt);
    void Update(ROS_INTERFACE &ros_interface);
    void Update(ROS_API &ros_api);
	void Render(std::shared_ptr<ViewManager> &_camera_ptr);

    //
    inline void set_max_miss_count(int max_miss_count_in){ _max_miss_count = max_miss_count_in; }
    inline void set_max_track_points(int max_track_points_in){ _max_track_points = max_track_points_in; }
    //
    size_t get_num_tracked_object(){ return obj_miss_count.size(); }

protected:
    void Init();
    virtual void LoadModel();
    //
    int _ROS_topic_id;
    // std::shared_ptr< msgs::LidRoi > msg_out_ptr;
    std::shared_ptr< msgs::DetectedObjectArray > msg_out_ptr;
    // ros::Time msg_time;
    std::string _ref_frame; // The reference frame for tracking

    //
    rmPolyLines3D rm_polylines3D;
    rmCircle rm_circle;
    rmText3D_v2 rm_text;

    void update_GL_data(ROS_API &ros_api);

    // Parameters
    int _max_miss_count;
    int _max_track_points;

private:


    // 3D text
    std::vector<rmText3D_v2::text_billboard_data> text_list;

    // buffer map: id --> a_line
    std::map<long long, std::queue<rmPolyLines3D::point_data> > line_map;
    std::map<long long, rmCircle::circle_data> circle_map;
    std::map<long long, int> obj_miss_count;

};

#endif // RM_LV2_OBJECT_TRACKING_H
