#ifndef RM_COLOR_BOARD_H
#define RM_COLOR_BOARD_H

#include "rmBaseModel.h"
#include "GL2DShape.hpp" // GL2DShape

/*

bool is_perspected_in=true,
bool is_moveable_in=true,
bool is_color_transformed_in=false


This is a combined model for all legacy image boards
- rmImageStatic             --> image_file_in,      true, (don't care), false
- rmImageDynamic            --> _ROS_topic_id_in,   true, (don't care), false
- rmImageStaticBackground   --> image_file_in,      false, false, true
- rmImageDynamicBackground  --> _ROS_topic_id_in,   false, false, true

We also got some new combinations
    - (any), false, true, true     <-- Flat image which can move/scaled/rotated, also possible to move to topest/lowest layer

Note:
    - For (is_perspected_in=false, is_moveable_in=true),  The image is set to top layer, (Front board)
    - For (is_perspected_in=false, is_moveable_in=false), The image is set to low layer, (background)
*/



class rmColorBoard : public rmBaseModel
{
public:

    rmColorBoard(
        std::string _path_Assets_in,
        std::string frame_id_in="",
        glm::vec4 color_vec4_in=glm::vec4(1.0f,1.0f,1.0f,1.0f),
        bool is_perspected_in=true,
        bool is_moveable_in=true
    );
    rmColorBoard(
        std::string _path_Assets_in,
        int _ROS_topic_id_in,
        glm::vec4 color_vec4_in=glm::vec4(1.0f,1.0f,1.0f,1.0f),
        bool is_perspected_in=true,
        bool is_moveable_in=true
    );
    //
	void Update(float dt);
    void Update(ROS_INTERFACE &ros_interface);
    void Update(ROS_API &ros_api);
	void Render(std::shared_ptr<ViewManager> &_camera_ptr);
    void Reshape(const glm::ivec2 & viewport_size_in);
    //
    inline glm::mat4 * get_model_m_ptr(){ return &(m_shape.model); }

    // Color
    glm::vec4 color_vec4;
    inline void set_color(const glm::vec3 &color_in, float alpha_in=1.0f){
        color_vec4 = glm::vec4(color_in, alpha_in);
    }

    // Shape
    //-------------------------------------------//
    // For usage, please refer to the GL2DShape
    GL2DShape shape;
    void updateBoardGeo(){
        shape.updateBoardGeo(_viewport_size);
        shape.get_shape(m_shape.shape);
        if ( shape.get_tranlate(translateMatrix) ){
            update_pose_model_by_model_ref();
        }
    }
    //-------------------------------------------//


protected:
    void Init();
    virtual void LoadModel();
    //
    int _ROS_topic_id;
    // std::shared_ptr<cv::Mat> msg_out_ptr;
    // ros::Time msg_time;
    std::string _frame_id;

    // Settings
    bool is_perspected;
    bool is_moveable;

    void update_GL_data();


private:
    // model info
    struct Shape{
        GLuint vao;
        GLuint vbo;
        GLuint m_texture;
        // image
        // size_t width;
        // size_t height;
        //
        glm::mat4 shape;
        glm::mat4 model;
    };
    Shape m_shape;


    //uniform id
	struct
	{
		GLint  mv_matrix;
		GLint  proj_matrix;
        //
        GLint  color_vec4;
	} uniforms;


};

#endif // RM_COLOR_BOARD_H
