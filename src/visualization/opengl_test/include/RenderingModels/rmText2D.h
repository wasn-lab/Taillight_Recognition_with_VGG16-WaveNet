#ifndef RM_TEXT_2D_H
#define RM_TEXT_2D_H

#include "rmBaseModel.h"

#include <queue>          // std::queue




class rmText2D : public rmBaseModel
{
public:
    // Different alignment
    //--------------------------------------//
    enum class ALIGN_X{
        LEFT,
        CENTER,
        RIGHT
    };
    enum class ALIGN_Y{
        TOP,
        CENTER,
        BUTTON
    };
    //--------------------------------------//


    // Different drawing method
    //--------------------------------------//
    // text2D in 3D space
    struct text2D_data{
        std::string text;
        glm::vec2   position_2D;
        float       size; // The "hight of line" in meter
        glm::vec3   color;
        ALIGN_X     align_x;
        ALIGN_Y     align_y;
        text2D_data(
            const std::string &text_in,
            const glm::vec2 &position_2D_in=glm::vec2(0.0f),
            float size_in=1.0f,
            const glm::vec3 &color_in=glm::vec3(1.0f),
            ALIGN_X align_x_in=ALIGN_X::LEFT,
            ALIGN_Y align_y_in=ALIGN_Y::TOP
        ):
            text(text_in),
            position_2D(position_2D_in),
            size(size_in),
            color(color_in),
            align_x(align_x_in),
            align_y(align_y_in)
        {
        }
    };
    //--------------------------------------//
    // end Different drawing method


    rmText2D();
    //
	void Update(float dt);
    void Update(ROS_INTERFACE &ros_interface);
    void Update(ROS_API &ros_api);
	void Render(std::shared_ptr<ViewManager> &_camera_ptr);

    // Insert method for texts
    //-------------------------------------//
    inline void insert_text(const text2D_data & data_in ){
        text2D_buffer.push( data_in );
    }
    //-------------------------------------//


    void _draw_one_text2D(std::shared_ptr<ViewManager> &_camera_ptr, text2D_data &_data_in);

protected:
    void Init();
    virtual void LoadModel();
    //
    // int _ROS_topic_id;
    // std::shared_ptr< msgs::LidRoi > box3d_out_ptr;
    // ros::Time msg_time;

    void selectFont2D(int newfont);
    void text2D_output(float x, float y, std::string string_in);

private:

    // Queue
    std::queue<text2D_data> text2D_buffer;

};

#endif // RM_TEXT_2D_H
