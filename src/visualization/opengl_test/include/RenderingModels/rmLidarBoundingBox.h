#ifndef RM_LIDARBOUNDINGBOX_H
#define RM_LIDARBOUNDINGBOX_H

#include "rmBaseModel.h"



class rmLidarBoundingBox : public rmBaseModel
{
public:
    rmLidarBoundingBox(std::string _path_Assets_in, int _ROS_topic_id_in, glm::vec3 _color_in=glm::vec3(0.29803922, 0.71372549, 0.88235294) );
    //
	void Update(float dt);
    void Update(ROS_INTERFACE &ros_interface);
    void Update(ROS_API &ros_api);
	void Render(std::shared_ptr<ViewManager> &_camera_ptr);
    //
    inline glm::mat4 * get_model_m_ptr(){ return &(m_shape.model); }

    //
    inline void set_color(const glm::vec3 &color_in){ _color = color_in; }
    inline void set_alpha(float alpha_in){ _alpha = alpha_in; }
    inline void set_line_width(float line_width_in){ _line_width = line_width_in; }
    void display_in_wire(bool is_wire_in);

protected:
    void Init();
    virtual void LoadModel();
    //
    int _ROS_topic_id;
    // std::shared_ptr< msgs::LidRoi > msg_out_ptr;
    std::shared_ptr< msgs::DetectedObjectArray > msg_out_ptr;
    // ros::Time msg_time;

    void update_GL_data();

    // Params
    glm::vec3 _color;
    float _alpha;
    float _line_width;
    // Rendering mode
    bool _is_wired;

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
    int _num_vertex_idx_per_box;
    long long _max_num_vertex_idx;
    int _num_vertex_per_box;
    long long _max_num_vertex;
    long long _max_num_box;



    // The box
    float box_half_size = 1.0f;

    //uniform id
	struct
	{
		GLint  mv_matrix;
		GLint  proj_matrix;
        GLint  alpha;
	} uniforms;

    std::vector<vertex_p_c> box_template;

    inline void generate_box_template(){
        box_template.resize(8);
        box_template[0].position = glm::vec3(0.0f, 0.0f, 0.0f);
        box_template[1].position = glm::vec3(0.0f, 0.0f, 1.0f);
        box_template[2].position = glm::vec3(1.0f, 0.0f, 1.0f);
        box_template[3].position = glm::vec3(1.0f, 0.0f, 0.0f);
        box_template[4].position = glm::vec3(0.0f, 1.0f, 0.0f);
        box_template[5].position = glm::vec3(0.0f, 1.0f, 1.0f);
        box_template[6].position = glm::vec3(1.0f, 1.0f, 1.0f);
        box_template[7].position = glm::vec3(1.0f, 1.0f, 0.0f);
    }

};

#endif // RM_LIDARBOUNDINGBOX_H
