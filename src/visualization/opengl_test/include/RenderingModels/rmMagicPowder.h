#ifndef RM_MAGIC_POWDER_H
#define RM_MAGIC_POWDER_H

#include "rmBaseModel.h"

#include <random> // For generating Gaussian random number

class MagicPowderManagement{
public:
    struct Particle {
        glm::vec3  Position, Velocity;
        float      Intensity;
        float      Life;
        Particle():
            Position(0.0f),
            Velocity(0.0f),
            Intensity(0.0f),
            Life(-1.0f)
        {}
    };
    //
    float velocity_ratio;
    float intensity_decay_rate;
    //
    MagicPowderManagement();
    void addParticle(const glm::vec3 &box_position_in, const glm::vec3 &box_size=glm::vec3(1.0f), const glm::vec3 &box_velocity_in=glm::vec3(0.0f) );
    void update();

    // The main container
    std::vector<Particle> particle_list;

private:
    std::default_random_engine rm_g;

    GLuint num_particles;

    TIME_STAMP::Time last_stamp;

    size_t _last_assigned_particle_id;

    size_t FindFirstUnusedParticle();

};













class rmMagicPowder : public rmBaseModel
{
public:
    rmMagicPowder(
        std::string _path_Assets_in,
        int _ROS_topic_id_in,
        std::string ref_frame_in
    );
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
    std::shared_ptr< msgs::DetectedObjectArray > msg_out_ptr;
    // ros::Time msg_time;
    std::string _ref_frame; // The reference frame for tracking


    void add_powder(ROS_API &ros_api);
    void update_GL_data();


    //
    MagicPowderManagement magicPowder_m;

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

#endif // RM_MAGIC_POWDER_H
