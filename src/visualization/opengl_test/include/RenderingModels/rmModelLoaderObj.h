#ifndef RM_MODEL_LOADER_OBJ_H
#define RM_MODEL_LOADER_OBJ_H

#include "rmBaseModel.h"

class rmModelLoaderObj : public rmBaseModel
{
public:
    rmModelLoaderObj(std::string _path_Assets_in, std::string modelFile, std::string textFile);
    //
	void Update(float dt);
    void Update(ROS_INTERFACE &ros_interface);
    void Update(ROS_API &ros_api);
	void Render(std::shared_ptr<ViewManager> &_camera_ptr);

    void set_color(glm::vec3 color_in);

protected:
    void Init();
    virtual void LoadModel();
    //
    // int _ROS_topic_id;
    // std::shared_ptr< pcl::PointCloud<pcl::PointXYZI> > msg_out_ptr;
    // ros::Time msg_time;

    void update_GL_data();

private:
    // model info
    struct Shape{
        GLuint vao;
        GLuint vbo;
        GLuint vboTex;
        GLuint ebo;
        GLuint p_normal;
        GLuint m_texture;
        //
        int materialId;
        int indexCount;

        glm::mat4 model; // Note: this is the pose relative to the mode reference frame, not the world frame
    };
    Shape m_shape;
    //
    std::string objName;
	std::string textName;


    //uniform id
	struct
	{
		GLint  mv_matrix;
		GLint  proj_matrix;
	} uniforms;
};

#endif // RM_MODEL_LOADER_OBJ_H
