#include "rmGrid.h"




rmGrid::rmGrid(std::string _path_Assets_in, std::string ref_frame_in, std::string follow_frame_in):
    _ref_frame(ref_frame_in), _follow_frame(follow_frame_in)
{
    _path_Shaders_sub_dir += "Grid/";
    init_paths(_path_Assets_in);
    //
	Init();
}
void rmGrid::Init(){
    //
	_program_ptr.reset( new ShaderProgram() );
    // Load shaders
    _program_ptr->AttachShader(get_full_Shader_path("Grid.vs.glsl"), GL_VERTEX_SHADER);
    _program_ptr->AttachShader(get_full_Shader_path("Grid.fs.glsl"), GL_FRAGMENT_SHADER);
    // Link _program_ptr
	_program_ptr->LinkProgram();
    //

    // Cache uniform variable id
	uniforms.proj_matrix = glGetUniformLocation(_program_ptr->GetID(), "proj_matrix");
	uniforms.mv_matrix = glGetUniformLocation(_program_ptr->GetID(), "mv_matrix");

    // Init model matrices
	m_shape.model = glm::mat4(1.0);
    attach_pose_model_by_model_ref_ptr(m_shape.model); // For adjusting the model pose by public methods

    //Load model to shader _program_ptr
	LoadModel();

}
void rmGrid::LoadModel(){

    // Default
    // Setting grid parameters
    grid_space[0] = 1.0; // m, x-axis
    grid_space[1] = 1.0; // m, y-axis
    half_bin_num[0] = 20; // x-axis
    half_bin_num[1] = 20; // y-axis
    //
    _ground_level = 0.0f; // -3.0f;
    _is_using_local_ground_level = false; // true;
    //
    _line_color = glm::vec3(0.1f, 0.1f, 0.1f);
    //



    // GL things
    glGenVertexArrays(1, &m_shape.vao);
	glBindVertexArray(m_shape.vao);
    // vbo
    glGenBuffers(1, &m_shape.vbo);

    ReLoadModel();


    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(vertex_p_c), NULL);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(vertex_p_c), (void *)sizeof(glm::vec3));
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);



}
void rmGrid::ReLoadModel(){

    // Derived parameters
    _num_lines_per_dim[0] = half_bin_num[0] * 2 + 1;
    _num_lines_per_dim[1] = half_bin_num[1] * 2 + 1;
    _num_lines = ( _num_lines_per_dim[0] + _num_lines_per_dim[1] );
    //


    glBindBuffer(GL_ARRAY_BUFFER, m_shape.vbo);
    glBufferData(GL_ARRAY_BUFFER, _num_lines * 2 * sizeof(vertex_p_c), NULL, GL_DYNAMIC_DRAW); // test, change to dynamic draw to assign point cloud
    // Directly assign data to memory of GPU
    //--------------------------------------------//
	vertex_p_c * vertex_ptr = (vertex_p_c *)glMapBufferRange(GL_ARRAY_BUFFER, 0, _num_lines * 2 * sizeof(vertex_p_c), GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT);
    int _j = 0;
    glm::vec3 _vertex_color = _line_color;
    // Span in x-axis
    for (int i = 0; i < _num_lines_per_dim[0]; i++)
	{
        // + side
		vertex_ptr[_j].position[0] = (i - half_bin_num[0]) * grid_space[0];
		vertex_ptr[_j].position[1] = half_bin_num[1]*grid_space[1];
		vertex_ptr[_j].position[2] = 0.0;
        for (size_t _k=0; _k < 3; ++_k){
            vertex_ptr[_j].color[_k] = _vertex_color[_k];
        }
        _j++;
        // - side
        vertex_ptr[_j].position[0] = (i - half_bin_num[0]) * grid_space[0];
		vertex_ptr[_j].position[1] = -1*(half_bin_num[1]*grid_space[1]);
		vertex_ptr[_j].position[2] = 0.0;
        for (size_t _k=0; _k < 3; ++_k){
            vertex_ptr[_j].color[_k] = _vertex_color[_k];
        }
        _j++;
	}
    // Span in y-axis
    // _j = 0;
    for (int i = 0; i < _num_lines_per_dim[1]; i++)
	{
        // + side
		vertex_ptr[_j].position[0] = (half_bin_num[0]*grid_space[0]);
		vertex_ptr[_j].position[1] = (i - half_bin_num[1]) * grid_space[1];
		vertex_ptr[_j].position[2] = 0.0;
        for (size_t _k=0; _k < 3; ++_k){
            vertex_ptr[_j].color[_k] = _vertex_color[_k];
        }
        _j++;
        // - side
        vertex_ptr[_j].position[0] = -1*(half_bin_num[0]*grid_space[0]);
		vertex_ptr[_j].position[1] = (i - half_bin_num[1]) * grid_space[1];
		vertex_ptr[_j].position[2] = 0.0;
        for (size_t _k=0; _k < 3; ++_k){
            vertex_ptr[_j].color[_k] = _vertex_color[_k];
        }
        _j++;
	}
	glUnmapBuffer(GL_ARRAY_BUFFER);
    //--------------------------------------------//

    m_shape.indexCount = _num_lines*2; //  1 * _num_vertex_idx_per_box; // ;
}
void rmGrid::Update(float dt){
    // Update the data (buffer variables) here
}
void rmGrid::Update(ROS_INTERFACE &ros_interface){
    // Update the data (buffer variables) here


    // Note: We get the transform update even if there is no new content in for maximum smoothness
    //      (the tf will update even there is no data)
    bool tf_successed = false;
    glm::mat4 _model_tf = ROStf2GLMmatrix(ros_interface.get_tf("map", tf_successed, false));
    // glm::mat4 _model_tf = ROStf2GLMmatrix(ros_interface.get_tf(_ROS_topic_id, tf_successed, true, msg_time));
    // m_shape.model = _model_tf;
    set_pose_modle_ref_by_world(_model_tf);
    // Common::print_out_mat4(_model_tf);


    // Set up grid center place
    glm::mat4 _follow_center_tf = ROStf2GLMmatrix(ros_interface.get_tf(_ref_frame, _follow_frame, tf_successed));
    glm::vec4 _follow_center_tf_trans = _follow_center_tf[3];
    _follow_center_tf_trans.x = glm::floor(_follow_center_tf_trans.x + grid_space.x/2.0f);
    _follow_center_tf_trans.y = glm::floor(_follow_center_tf_trans.y + grid_space.y/2.0f);
    _follow_center_tf_trans.z = _ground_level;
    //
    m_shape.model = glm::mat4(1.0f);
    m_shape.model[3] = _follow_center_tf_trans;
    //


}

void rmGrid::Update(ROS_API &ros_api){
    // Update the data (buffer variables) here

    // Get tf
    bool tf_successed = false;
    glm::mat4 _model_tf = ROStf2GLMmatrix(ros_api.get_tf(_ref_frame, tf_successed));
    set_pose_modle_ref_by_world(_model_tf);
    // end Get tf

    // Set up grid center place
    glm::mat4 _follow_center_tf = ROStf2GLMmatrix(ros_api.get_tf(_ref_frame, _follow_frame, tf_successed));
    glm::vec4 _follow_center_tf_trans = _follow_center_tf[3];
    _follow_center_tf_trans.x = glm::floor(_follow_center_tf_trans.x / grid_space.x + 0.5f) * grid_space.x;
    _follow_center_tf_trans.y = glm::floor(_follow_center_tf_trans.y / grid_space.y + 0.5f) * grid_space.y;
    if (_is_using_local_ground_level){
        _follow_center_tf_trans.z = _follow_center_tf_trans.z + _ground_level;
    }else{
        _follow_center_tf_trans.z = _ground_level;
    }
    //
    m_shape.model = glm::mat4(1.0f);
    m_shape.model[3] = _follow_center_tf_trans;
    //
}


void rmGrid::Render(std::shared_ptr<ViewManager> &_camera_ptr){

    glBindVertexArray(m_shape.vao);

	_program_ptr->UseProgram();
    // The transformation matrices and projection matrices
    glUniformMatrix4fv(uniforms.mv_matrix, 1, GL_FALSE, value_ptr( get_mv_matrix(_camera_ptr, m_shape.model) ));
    glUniformMatrix4fv(uniforms.proj_matrix, 1, GL_FALSE, value_ptr(_camera_ptr->GetProjectionMatrix()));
    // Setting
    // glLineWidth(5.0);
    // Draw the element according to ebo
    glDrawArrays(GL_LINES, 0, m_shape.indexCount); // draw part of points
    //--------------------------------//
    _program_ptr->CloseProgram();
}
