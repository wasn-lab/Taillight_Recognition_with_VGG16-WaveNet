#ifndef RM_POINTCLOUD_H
#define RM_POINTCLOUD_H

#include "rmBaseModel.h"

class rmPointCloud : public rmBaseModel
{
public:
    rmPointCloud(std::string _path_Assets_in, int _ROS_topic_id_in);
    //
	void Update(float dt);
    void Update(ROS_INTERFACE &ros_interface);
    void Update(ROS_API &ros_api);
	void Render(std::shared_ptr<ViewManager> &_camera_ptr);

    void set_color(glm::vec3 color_in);

    // TIME_STAMP::FPS fps_of_update;

protected:
    void Init();
    virtual void LoadModel();
    //
    int _ROS_topic_id;
    std::shared_ptr< pcl::PointCloud<pcl::PointXYZI> > msg_out_ptr;
    // ros::Time msg_time;

    void update_GL_data();

private:
    // model info
    struct Shape{
        GLuint vao;
        GLuint vbo;
        GLuint m_texture;
        //
        int indexCount;
        //
        glm::vec3 color;

        glm::mat4 model;
    };
    Shape m_shape;

    // The structure for point
    struct vertex_p_c
	{
		glm::vec3     position;
		glm::vec3     color;
	};
    long long _max_num_vertex;

    //uniform id
	struct
	{
		GLint  mv_matrix;
		GLint  proj_matrix;
	} uniforms;

    static inline float random_float()
    {
        static unsigned int seed = 0x13371337;
    	float res;
    	unsigned int tmp;
    	seed *= 16807;
    	tmp = seed ^ (seed >> 4) ^ (seed << 15);
    	*((unsigned int *)&res) = (tmp >> 9) | 0x3F800000;
    	return (res - 1.0f);
    }
};

#endif // RM_POINTCLOUD_H
