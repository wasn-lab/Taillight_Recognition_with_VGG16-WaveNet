#include "rmPointCloud.h"


rmPointCloud::rmPointCloud(std::string _path_Assets_in, int _ROS_topic_id_in):
    _ROS_topic_id(_ROS_topic_id_in)
    // fps_of_update( std::string("PC ") + std::to_string(_ROS_topic_id_in) )
{
    init_paths(_path_Assets_in);
    _max_num_vertex = 5000000; // 5*10^6 // 100000;
	Init();
}
void rmPointCloud::Init(){
    //
	_program_ptr.reset( new ShaderProgram() );
    // Load shaders
    _program_ptr->AttachShader(get_full_Shader_path("PointCloud.vs.glsl"), GL_VERTEX_SHADER);
    _program_ptr->AttachShader(get_full_Shader_path("PointCloud.fs.glsl"), GL_FRAGMENT_SHADER);
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
void rmPointCloud::LoadModel(){
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
    std::string _texture_1("star.png");
    std::cout << "start loading <" << _texture_1 << ">\n";
	TextureData tdata = Common::Load_png(get_full_Assets_path(_texture_1).c_str());
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, tdata.width, tdata.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, tdata.data);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    std::cout << "Load texture success!\n";

}
void rmPointCloud::Update(float dt){
    // Update the data (buffer variables) here
}
void rmPointCloud::Update(ROS_INTERFACE &ros_interface){
    // Update the data (buffer variables) here

    // test, use transform
    ros::Time msg_time;
    bool _result = ros_interface.get_any_pointcloud( _ROS_topic_id, msg_out_ptr, msg_time);

    if (_result){
        //
        update_GL_data();
        //
        // fps_of_update.stamp();  fps_of_update.show();
    }

    // Note: We get the transform update even if there is no new content in for maximum smoothness
    //      (the tf will update even there is no data)
    bool tf_successed = false;
    glm::mat4 _model_tf = ROStf2GLMmatrix(ros_interface.get_tf(_ROS_topic_id, tf_successed, false));
    // glm::mat4 _model_tf = ROStf2GLMmatrix(ros_interface.get_tf(_ROS_topic_id, tf_successed, true, msg_time));
    // m_shape.model = _model_tf;
    set_pose_modle_ref_by_world(_model_tf);
    // Common::print_out_mat4(_model_tf);


}
void rmPointCloud::Update(ROS_API &ros_api){
    // Update the data (buffer variables) here

    // test, use transform
    ros::Time msg_time;
    bool _result = false;

    /*
    // Scops for any_ptr
    {
        boost::any any_ptr;
        _result = ros_api.get_any_message( _ROS_topic_id, any_ptr, msg_time );
        if (_result){
            std::shared_ptr< pcl::PointCloud<pcl::PointXYZI> > *_ptr_ptr = boost::any_cast< std::shared_ptr< pcl::PointCloud<pcl::PointXYZI> > >( &any_ptr );
            msg_out_ptr = *_ptr_ptr;
        }
    }// end Scops for any_ptr
    */

    _result = ros_api.get_message(_ROS_topic_id, msg_out_ptr, msg_time);
    // _result = ROS_API_TOOL::get_message(ros_api, _ROS_topic_id, msg_out_ptr, msg_time);
    // std::cout << "msg_out_ptr.use_count() = " << msg_out_ptr.use_count() << "\n";

    if (_result){
        update_GL_data();
        //
        // fps_of_update.stamp();  fps_of_update.show();
    }

    // Get tf
    bool tf_successed = false;
    glm::mat4 _model_tf = ROStf2GLMmatrix(ros_api.get_tf(_ROS_topic_id, tf_successed));
    set_pose_modle_ref_by_world(_model_tf);
    // end Get tf


    /*
    ROS_INTERFACE &ros_interface = ros_api.ros_interface;
    // Note: We get the transform update even if there is no new content in for maximum smoothness
    //      (the tf will update even there is no data)
    bool tf_successed = false;
    glm::mat4 _model_tf = ROStf2GLMmatrix(ros_interface.get_tf(_ROS_topic_id, tf_successed, false));
    // glm::mat4 _model_tf = ROStf2GLMmatrix(ros_interface.get_tf(_ROS_topic_id, tf_successed, true, msg_time));
    // m_shape.model = _model_tf;
    set_pose_modle_ref_by_world(_model_tf);
    // Common::print_out_mat4(_model_tf);
    */


}


void rmPointCloud::Render(std::shared_ptr<ViewManager> &_camera_ptr){

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



void rmPointCloud::set_color(glm::vec3 color_in){
    m_shape.color = color_in;
}

void rmPointCloud::update_GL_data(){
    m_shape.indexCount = msg_out_ptr->width;
    if (m_shape.indexCount > _max_num_vertex){
        m_shape.indexCount = _max_num_vertex;
    }

    // vao vbo
    glBindVertexArray(m_shape.vao);
    glBindBuffer(GL_ARRAY_BUFFER, m_shape.vbo); // Start to use the buffer

    // std::cout << "msg_out_ptr->header.seq = " << msg_out_ptr->header.seq << "\n";
    // vertex_p_c * vertex_ptr = (vertex_p_c *)glMapBufferRange(GL_ARRAY_BUFFER, 0, _max_num_vertex * sizeof(vertex_p_c), GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT);
    vertex_p_c * vertex_ptr = (vertex_p_c *)glMapBufferRange(GL_ARRAY_BUFFER, 0, m_shape.indexCount * sizeof(vertex_p_c), GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT);
    for (size_t i = 0; i < m_shape.indexCount; i++)
    {
        vertex_ptr[i].position[0] = msg_out_ptr->points[i].x;
        vertex_ptr[i].position[1] = msg_out_ptr->points[i].y;
        vertex_ptr[i].position[2] = msg_out_ptr->points[i].z;
        // If we don't keep udating the color, the color will be lost when resizing the window.
        vertex_ptr[i].color[0] = m_shape.color[0]; //
        vertex_ptr[i].color[1] = m_shape.color[1]; //
        vertex_ptr[i].color[2] = m_shape.color[2]; //
    }
    glUnmapBuffer(GL_ARRAY_BUFFER);
}
