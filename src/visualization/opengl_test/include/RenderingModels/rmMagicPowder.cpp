#include "rmMagicPowder.h"







MagicPowderManagement::MagicPowderManagement():
    last_stamp(TIME_PARAM::NOW)
{
    num_particles = 50000;
    particle_list.resize(num_particles);
    _last_assigned_particle_id = 0;
    //
    velocity_ratio = 0.3f;
    intensity_decay_rate = 0.05f;
}
void MagicPowderManagement::addParticle(const glm::vec3 &box_position_in, const glm::vec3 &box_size,const  glm::vec3 &box_velocity_in){
    size_t _idx = FindFirstUnusedParticle();
    Particle &p = particle_list[_idx];

    std::normal_distribution<float> dist_x(box_position_in.x, box_size.x*0.25f );
    std::normal_distribution<float> dist_y(box_position_in.y, box_size.y*0.25f );
    std::normal_distribution<float> dist_z(box_position_in.z, box_size.z*0.25f );
    //
    std::normal_distribution<float> dist_v_x(0.0f, box_size.x*0.25f );
    std::normal_distribution<float> dist_v_y(0.0f, box_size.y*0.25f );
    std::normal_distribution<float> dist_v_z(0.0f, box_size.z*0.25f );

    p.Position = glm::vec3( dist_x(rm_g), dist_y(rm_g), dist_z(rm_g));
    p.Intensity = 1.0f;
    p.Life = 10.0f;
    // p.Velocity = box_velocity_in * velocity_ratio;
    p.Velocity = (box_velocity_in + glm::vec3( dist_v_x(rm_g), dist_v_y(rm_g), dist_v_z(rm_g)) ) * velocity_ratio;
}
void MagicPowderManagement::update(){
    // Calculate dt
    TIME_STAMP::Time current_stamp(TIME_PARAM::NOW);
    float dt = (current_stamp - last_stamp).toSec();
    last_stamp.swap(current_stamp); // swap

    // Uupdate all particles
    for (size_t i = 0; i < particle_list.size(); ++i){
        Particle &p = particle_list[i];
        p.Life -= dt; // reduce life
        if (p.Life > 0.0f){
            // particle is alive, thus update
            p.Position += p.Velocity * dt;
            p.Intensity -= intensity_decay_rate * dt;
            // p.Intensity *= (1.0f - intensity_decay_rate);
            if (p.Intensity < 0.0){
                p.Intensity = 0.0;
                p.Life = 0.0; // Distroid this particle since the Intensity is 0.0
            }
        }
    }
    // end for
}
size_t MagicPowderManagement::FindFirstUnusedParticle(){
    size_t _j = _last_assigned_particle_id;
    // Search from last used particle
    for (size_t i=0; i < particle_list.size(); ++i){
        if (particle_list[_j].Life <= 0.0f){
            _last_assigned_particle_id = _j;
            return _last_assigned_particle_id;
        }
        _j++;
        _j %= particle_list.size();
    }
    // Override next particle if all others are alive
    _last_assigned_particle_id++;
    return _last_assigned_particle_id;
}



//------------------------------------------------------//

rmMagicPowder::rmMagicPowder(
    std::string _path_Assets_in,
    int _ROS_topic_id_in,
    std::string ref_frame_in
):
    _ROS_topic_id(_ROS_topic_id_in),
    _ref_frame(ref_frame_in)
    // fps_of_update( std::string("PC ") + std::to_string(_ROS_topic_id_in) )
{
    _path_Shaders_sub_dir += "MagicPowder/";
    init_paths(_path_Assets_in);
    _max_num_vertex = 5000000; // 5*10^6 // 100000;
	Init();
}
void rmMagicPowder::Init(){
    //
	_program_ptr.reset( new ShaderProgram() );
    // Load shaders
    _program_ptr->AttachShader(get_full_Shader_path("MagicPowder.vs.glsl"), GL_VERTEX_SHADER);
    _program_ptr->AttachShader(get_full_Shader_path("MagicPowder.fs.glsl"), GL_FRAGMENT_SHADER);
    // Link _program_ptr
	_program_ptr->LinkProgram();
    //

    // Cache uniform variable id
	uniforms.proj_matrix = glGetUniformLocation(_program_ptr->GetID(), "proj_matrix");
	uniforms.mv_matrix = glGetUniformLocation(_program_ptr->GetID(), "mv_matrix");

    // Init model matrices
	m_shape.model = glm::mat4(1.0);
    attach_pose_model_by_model_ref_ptr(m_shape.model); // For adjusting the model pose by public methods
    m_shape.color = glm::vec3(1.0);

    //Load model to shader _program_ptr
	LoadModel();

}
void rmMagicPowder::LoadModel(){
    glGenVertexArrays(1, &m_shape.vao);
	glBindVertexArray(m_shape.vao);


	glGenBuffers(1, &m_shape.vbo);
	glBindBuffer(GL_ARRAY_BUFFER, m_shape.vbo);
	// glBufferData(GL_ARRAY_BUFFER, _max_num_vertex * sizeof(vertex_p_c), NULL, GL_STATIC_DRAW);
	glBufferData(GL_ARRAY_BUFFER, _max_num_vertex * sizeof(vertex_p_c), NULL, GL_DYNAMIC_DRAW); // test, change to dynamic draw to assign point cloud

    // Directly assign data to memory of GPU
    //--------------------------------------------//
	vertex_p_c * vertex_ptr = (vertex_p_c *)glMapBufferRange(GL_ARRAY_BUFFER, 0, _max_num_vertex * sizeof(vertex_p_c), GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT);
	int i;
	for (i = 0; i < _max_num_vertex; i++)
	{
		vertex_ptr[i].position[0] = (random_float() * 2.0f - 1.0f) * 100.0f;
		vertex_ptr[i].position[1] = (random_float() * 2.0f - 1.0f) * 100.0f;
		vertex_ptr[i].position[2] = random_float();
        vertex_ptr[i].color[0] = m_shape.color[0]; //
        vertex_ptr[i].color[1] = m_shape.color[1]; //
        vertex_ptr[i].color[2] = m_shape.color[2]; //
	}
	glUnmapBuffer(GL_ARRAY_BUFFER);
    m_shape.indexCount = 0; // 100000; // _max_num_vertex;
    //--------------------------------------------//

	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(vertex_p_c), NULL);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(vertex_p_c), (void *)sizeof(glm::vec3));
	glEnableVertexAttribArray(0);
	glEnableVertexAttribArray(1);

    // Texture
	// glEnable(GL_TEXTURE_2D);
	// glActiveTexture(GL_TEXTURE0);
    glGenTextures(1, &m_shape.m_texture);
	glBindTexture(GL_TEXTURE_2D, m_shape.m_texture);
    //Load texture data from file
    std::string _texture_1("magic_powder.png");
    std::cout << "start loading <" << _texture_1 << ">\n";
	TextureData tdata = Common::Load_png(get_full_Assets_path(_texture_1).c_str());
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, tdata.width, tdata.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, tdata.data);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    std::cout << "Load texture success!\n";

}
void rmMagicPowder::Update(float dt){
    // Update the data (buffer variables) here
}
void rmMagicPowder::Update(ROS_INTERFACE &ros_interface){
    // Update the data (buffer variables) here
}
void rmMagicPowder::Update(ROS_API &ros_api){
    // Update the data (buffer variables) here

    // test, use transform
    ros::Time msg_time;
    bool _result = false;

    _result = ros_api.get_message(_ROS_topic_id, msg_out_ptr, msg_time);

    //
    if (_result){
        add_powder(ros_api); // Add/replace particles and update particles
    }
    // Update particles
    magicPowder_m.update();
    // Assign particles to buffer
    update_GL_data();

    if (_ref_frame.size() > 0){
        // Get tf
        bool tf_successed = false;
        glm::mat4 _model_tf = ROStf2GLMmatrix(ros_api.get_tf(_ref_frame, tf_successed));
        set_pose_modle_ref_by_world(_model_tf);
        // end Get tf
    }
}


void rmMagicPowder::Render(std::shared_ptr<ViewManager> &_camera_ptr){

    glBindVertexArray(m_shape.vao);
	_program_ptr->UseProgram();
    // The transformation matrices and projection matrices
    glUniformMatrix4fv(uniforms.mv_matrix, 1, GL_FALSE, value_ptr( get_mv_matrix(_camera_ptr, m_shape.model) ));
    glUniformMatrix4fv(uniforms.proj_matrix, 1, GL_FALSE, value_ptr(_camera_ptr->GetProjectionMatrix()));
    // Point sprite
    //--------------------------------//
    glEnable(GL_POINT_SPRITE);
    {
        // glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        // glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, m_shape.m_texture);
        glEnable(GL_PROGRAM_POINT_SIZE);
        glDrawArrays(GL_POINTS, 0, m_shape.indexCount); // draw part of points
    }
    // Close
    glDisable(GL_POINT_SPRITE);
    //--------------------------------//
    _program_ptr->CloseProgram();
}



void rmMagicPowder::set_color(glm::vec3 color_in){
    m_shape.color = color_in;
}


void rmMagicPowder::add_powder(ROS_API &ros_api){
    // Magic powder generation and management
    long long num_box = msg_out_ptr->objects.size();
    if (num_box == 0){
        // Update particles
        magicPowder_m.update();
        return;
    }
    // else
    if (num_box > magicPowder_m.particle_list.size() ){
        num_box = magicPowder_m.particle_list.size();
    }


    // Update transform
    //--------------------------------//
    glm::mat4 tf_box_to_ref(1.0f);
    if (_ref_frame.size() > 0){
        // Get tf
        bool tf_successed = false;
        tf_box_to_ref = ROStf2GLMmatrix(ros_api.get_tf(_ref_frame, ros_api.ros_interface.get_topic_param(_ROS_topic_id).frame_id, tf_successed));
        // end Get tf
    }
    //--------------------------------//

    auto * _box_ptr = &(msg_out_ptr->objects[0].bPoint);
    // Add/replace particles
    for (size_t i=0; i < num_box; ++i){
        _box_ptr = &(msg_out_ptr->objects[i].bPoint);
        glm::vec3 p0(_box_ptr->p0.x, _box_ptr->p0.y, _box_ptr->p0.z);
        glm::vec3 p1(_box_ptr->p1.x, _box_ptr->p1.y, _box_ptr->p1.z);
        glm::vec3 p3(_box_ptr->p3.x, _box_ptr->p3.y, _box_ptr->p3.z);
        glm::vec3 p4(_box_ptr->p4.x, _box_ptr->p4.y, _box_ptr->p4.z);
        glm::vec3 p6(_box_ptr->p6.x, _box_ptr->p6.y, _box_ptr->p6.z);
        float _W = glm::l2Norm( p3 -  p0);
        float _L = glm::l2Norm( p4 -  p0);
        float _H = glm::l2Norm( p1 -  p0);
        glm::vec3 center_point = (tf_box_to_ref*glm::vec4( (0.5f*( p0 +  p6)), 1.0f )).xyz();
        for (size_t k=0; k < 3; ++k){
            magicPowder_m.addParticle(
                center_point,
                glm::vec3(_W, _L, _H),
                glm::vec3(0.0f)
            );
        }

    }


}
void rmMagicPowder::update_GL_data(){


    m_shape.indexCount = magicPowder_m.particle_list.size();
    if (m_shape.indexCount > _max_num_vertex){
        m_shape.indexCount = _max_num_vertex;
    }

    // vao vbo
    glBindVertexArray(m_shape.vao);
    glBindBuffer(GL_ARRAY_BUFFER, m_shape.vbo); // Start to use the buffer
    // std::cout << "msg_out_ptr->header.seq = " << msg_out_ptr->header.seq << "\n";
    // vertex_p_c * vertex_ptr = (vertex_p_c *)glMapBufferRange(GL_ARRAY_BUFFER, 0, _max_num_vertex * sizeof(vertex_p_c), GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT);
    vertex_p_c * vertex_ptr = (vertex_p_c *)glMapBufferRange(GL_ARRAY_BUFFER, 0, m_shape.indexCount * sizeof(vertex_p_c), GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT);
    size_t _j = 0;
    for (size_t i = 0; i < magicPowder_m.particle_list.size(); i++){
        auto & _p = magicPowder_m.particle_list[i];
        if (_p.Life > 0.0){ // Only draw the points that remain time
            vertex_ptr[_j].position = _p.Position;
            // If we don't keep udating the color, the color will be lost when resizing the window.
            vertex_ptr[_j].color[0] = m_shape.color[0] * _p.Intensity; //
            vertex_ptr[_j].color[1] = m_shape.color[1] * _p.Intensity; //
            vertex_ptr[_j].color[2] = m_shape.color[2] * _p.Intensity; //
            //
            _j++;
            if (_j >= _max_num_vertex){ // too many points
                break;
            }
        }
    }
    glUnmapBuffer(GL_ARRAY_BUFFER);
    m_shape.indexCount = _j;

}
