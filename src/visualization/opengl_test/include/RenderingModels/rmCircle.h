#ifndef RM_CIRCLE_H
#define RM_CIRCLE_H

#include "rmBaseModel.h"

#include <queue>          // std::queue

class rmCircle : public rmBaseModel
{
public:

    // The structure for point
    struct circle_data
	{
		glm::vec3     position;
        float         radious;
		glm::vec3     color;
        circle_data(
            glm::vec3     position_in=glm::vec3(0.0f),
            float         radious_in=1.0f,
    		glm::vec3     color_in=glm::vec3(1.0f)
        ):
            position(position_in),
            radious(radious_in),
            color(color_in)
        {}
	};


    rmCircle(std::string _path_Assets_in, std::string frame_id_in);
    rmCircle(std::string _path_Assets_in, int _ROS_topic_id_in);
    //
	void Update(float dt);
    void Update(ROS_INTERFACE &ros_interface);
    void Update(ROS_API &ros_api);
	void Render(std::shared_ptr<ViewManager> &_camera_ptr);
    //
    inline glm::mat4 * get_model_m_ptr(){ return &(m_shape.model); }

    // Insert method for circle
    //-------------------------------------//
    void insert_circle(const std::vector<circle_data> & data_list_in );
    void insert_circle(const std::map<long long, circle_data> & data_map_in );
    //-------------------------------------//

    inline void set_line_width(float line_width_in){ _line_width = line_width_in; }

protected:
    void Init();
    virtual void LoadModel();
    //
    int _ROS_topic_id;
    // std::shared_ptr< msgs::LidRoi > msg_out_ptr;
    // ros::Time msg_time;
    std::string _frame_id;

    float _line_width;


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


    int _num_vertex_per_shape;
    long long _max_num_vertex;
    long long _max_num_shape;



    //uniform id
	struct
	{
		GLint  mv_matrix;
		GLint  proj_matrix;
	} uniforms;

    // buffer
    std::vector<circle_data> circle_buffer;


};

#endif // RM_CIRCLE_H
