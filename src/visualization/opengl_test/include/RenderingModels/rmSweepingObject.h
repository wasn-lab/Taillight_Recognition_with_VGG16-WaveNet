#ifndef RM_SWEEPING_OGJECT_H
#define RM_SWEEPING_OGJECT_H

#include "rmBaseModel.h"



class rmSweepingObject : public rmBaseModel
{
public:
    rmSweepingObject(std::string _path_Assets_in, std::string frame_id_in, int draw_mode_in=0);
    rmSweepingObject(std::string _path_Assets_in, int _ROS_topic_id_in, int draw_mode_in=0);
    //
	void Update(float dt);
    void Update(ROS_INTERFACE &ros_interface);
    void Update(ROS_API &ros_api);
	void Render(std::shared_ptr<ViewManager> &_camera_ptr);
    //
    inline glm::mat4 * get_model_m_ptr(){ return &(m_shape.model); }

    //
    inline void insert_section_vertexes(const std::vector<glm::vec3> &section_vertexes_in){ section_vertexes = section_vertexes_in;}
    void insert_curve_Points(const std::vector<glm::vec3> &curve_Points_in);

    //
    inline void set_line_width(float line_width_in){ _line_width = line_width_in; }
    void set_color_head(const glm::vec3 & color_in){ _color_head = color_in; }
    void set_color_tail(const glm::vec3 & color_in){ _color_tail = color_in; }
    void set_close_loop(bool is_close_loop_in){  is_close_loop = is_close_loop_in; }

protected:
    void Init();
    virtual void LoadModel();
    //
    int _ROS_topic_id;
    // std::shared_ptr< msgs::LidRoi > msg_out_ptr;
    // ros::Time msg_time;
    std::string _frame_id;

    float _line_width;

    // parameters
    int draw_mode;
    bool is_close_loop;

    void update_GL_data();

    //
    void _update_lookat_matrix_list();

    // Params
    glm::vec3 _color_head;
    glm::vec3 _color_tail;

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

    //
    int _max_num_vertex_of_curve;
    std::vector<glm::vec3> _curve_Points;

    // lookat matrix - a list of transformation matrices for each transaction
    std::vector<glm::mat4> lookat_matrix_list;



    int _max_num_vertex_of_shape;
    std::vector<glm::vec3> section_vertexes;

    //uniform id
	struct
	{
		GLint  mv_matrix;
		GLint  proj_matrix;
        std::vector<GLint> lookat_matrix;
        std::vector<GLint> section_vertexes;
        GLint _num_vertex_of_shape;
        GLint shape_mode;
	} uniforms;

    // std::vector<vertex_p_c> box_template;


};

#endif // RM_SWEEPING_OGJECT_H
