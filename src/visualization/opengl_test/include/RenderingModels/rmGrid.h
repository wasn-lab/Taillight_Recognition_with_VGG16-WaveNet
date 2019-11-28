#ifndef RM_GRID_H
#define RM_GRID_H

#include "rmBaseModel.h"



class rmGrid : public rmBaseModel
{
public:
    rmGrid(
        std::string _path_Assets_in,
         std::string ref_frame_in,
         std::string follow_frame_in
     );
    //
	void Update(float dt);
    void Update(ROS_INTERFACE &ros_interface);
    void Update(ROS_API &ros_api);
	void Render(std::shared_ptr<ViewManager> &_camera_ptr);

    void set_grid_param(
        float space_x=1.0,
        float space_y=1.0,
        int half_bin_num_x=20,
        int half_bin_num_y=20,
        float ground_level=0.0f,
        bool is_using_local_ground_level=false,
        glm::vec3 line_color=glm::vec3(0.1f, 0.1f, 0.1f)
    ){
        grid_space[0] = space_x;
        grid_space[1] = space_y;
        half_bin_num[0] = half_bin_num_x;
        half_bin_num[1] = half_bin_num_y;
        _ground_level = ground_level;
        _is_using_local_ground_level = is_using_local_ground_level;
        _line_color = line_color;
        //
        ReLoadModel();
    }


protected:
    void Init();
    virtual void LoadModel();
    void ReLoadModel();
    //
    // int _ROS_topic_id;
    // std::shared_ptr< msgs::LidRoi > msg_out_ptr;
    // ros::Time msg_time;
    std::string _ref_frame;
    std::string _follow_frame;

    void update_GL_data();

private:
    // model info
    struct Shape{
        GLuint vao;
        GLuint vbo;
        GLuint ebo;
        GLuint m_texture;
        //
        int indexCount;

        glm::mat4 model;
    };
    Shape m_shape;

    // The structure for point
    struct vertex_p_c
	{
		glm::vec3     position;
		glm::vec3     color;
	};

    // Parameters
    // Set:
    glm::vec2 grid_space; // meter, (space_x, space_y)
    glm::ivec2 half_bin_num; // The "bin", not "line";  (half_num_x, half_num_y)
    GLfloat _ground_level; // m
    bool _is_using_local_ground_level;
    glm::vec3 _line_color;
    // Derived:
    glm::ivec2 _num_lines_per_dim; // (num_line_x, num_line_y)
    int _num_lines;
    //

    //uniform id
	struct
	{
		GLint  mv_matrix;
		GLint  proj_matrix;
	} uniforms;



};

#endif // RM_GRID_H
