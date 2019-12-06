#include "rmColorBoard.h"

static const GLfloat window_positions[] =
{
    // Position i, texcord i
	1.0f,-1.0f,1.0f,0.0f,  // right-down
	-1.0f,-1.0f,0.0f,0.0f, // left-down
	-1.0f,1.0f,0.0f,1.0f,  // left-up
	1.0f,1.0f,1.0f,1.0f    // right-up
};


rmColorBoard::rmColorBoard(
    std::string _path_Assets_in,
    std::string frame_id_in,
    glm::vec4 color_vec4_in,
    bool is_perspected_in,
    bool is_moveable_in
):
    _frame_id(frame_id_in),
    color_vec4(color_vec4_in),
    is_perspected(is_perspected_in),
    is_moveable(is_moveable_in)
{
    _path_Shaders_sub_dir += "ImageBoard/";
    init_paths(_path_Assets_in);
    //
	Init();
}
rmColorBoard::rmColorBoard(
    std::string _path_Assets_in,
    int _ROS_topic_id_in,
    glm::vec4 color_vec4_in,
    bool is_perspected_in,
    bool is_moveable_in
):
    _ROS_topic_id(_ROS_topic_id_in),
    color_vec4(color_vec4_in),
    is_perspected(is_perspected_in),
    is_moveable(is_moveable_in)
{
    _path_Shaders_sub_dir += "ImageBoard/";
    init_paths(_path_Assets_in);
    //
	Init();
}

void rmColorBoard::Init(){
    //
	_program_ptr.reset( new ShaderProgram() );
    // Load shaders
    //----------------------------------------//
    // VS
    if (is_perspected){
        _program_ptr->AttachShader(get_full_Shader_path("ImageBoard.vs.Perspected.glsl"), GL_VERTEX_SHADER);
    }else{
        if (is_moveable){
            _program_ptr->AttachShader(get_full_Shader_path("ImageBoard.vs.Moveable.glsl"), GL_VERTEX_SHADER);
        }else{ // Background
            _program_ptr->AttachShader(get_full_Shader_path("ImageBoard.vs.Background.glsl"), GL_VERTEX_SHADER);
        }
    }
    // FS
    _program_ptr->AttachShader(get_full_Shader_path("ColorBoard.fs.glsl"), GL_FRAGMENT_SHADER);
    //----------------------------------------//
    // Link _program_ptr
	_program_ptr->LinkProgram();
    //

    // Initialize variables
    // Init model matrices
    m_shape.shape = glm::mat4(1.0);
	m_shape.model = glm::mat4(1.0);
    attach_pose_model_by_model_ref_ptr(m_shape.model); // For adjusting the model pose by public methods
    //

    // Cache uniform variable id
    //----------------------------------------//
    if (is_perspected){
        uniforms.proj_matrix = glGetUniformLocation(_program_ptr->GetID(), "proj_matrix");
        uniforms.mv_matrix = glGetUniformLocation(_program_ptr->GetID(), "mv_matrix");
    }else if (is_moveable){
        uniforms.mv_matrix = glGetUniformLocation(_program_ptr->GetID(), "mv_matrix");
    }
    uniforms.color_vec4 = glGetUniformLocation(_program_ptr->GetID(), "color_vec4");
    //----------------------------------------//


    //Load model to shader _program_ptr
	LoadModel();

}
void rmColorBoard::LoadModel(){
    glGenVertexArrays(1, &m_shape.vao);
	glBindVertexArray(m_shape.vao);


	glGenBuffers(1, &m_shape.vbo);
	glBindBuffer(GL_ARRAY_BUFFER, m_shape.vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(window_positions), window_positions, GL_STATIC_DRAW);


	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(GL_FLOAT) * 4, NULL);
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(GL_FLOAT) * 4, (const GLvoid*)(sizeof(GL_FLOAT) * 2) );
	glEnableVertexAttribArray(0);
	glEnableVertexAttribArray(1);

}
void rmColorBoard::Update(float dt){
    // Update the data (buffer variables) here
}
void rmColorBoard::Update(ROS_INTERFACE &ros_interface){
    // Update the data (buffer variables) here

}
void rmColorBoard::Update(ROS_API &ros_api){
    // Update the data (buffer variables) here

    // Update transform
    //--------------------------------//
    if (_frame_id.size() > 0){
        // Get tf
        bool tf_successed = false;
        glm::mat4 _model_tf = ROStf2GLMmatrix(ros_api.get_tf(_frame_id, tf_successed));
        set_pose_modle_ref_by_world(_model_tf);
        // end Get tf
    }else{
        if ( ros_api.ros_interface.is_topic_got_frame(_ROS_topic_id) ){
            // Get tf
            bool tf_successed = false;
            glm::mat4 _model_tf = ROStf2GLMmatrix(ros_api.get_tf(_ROS_topic_id, tf_successed));
            set_pose_modle_ref_by_world(_model_tf);
            // end Get tf
        }
    }
    //--------------------------------//
    // end Update transform
}


void rmColorBoard::Render(std::shared_ptr<ViewManager> &_camera_ptr){

    glBindVertexArray(m_shape.vao);
	_program_ptr->UseProgram();

    if (is_perspected){
        // m_shape.model = translateMatrix * rotateMatrix * scaleMatrix;
        // The transformation matrices and projection matrices
        glUniformMatrix4fv(uniforms.mv_matrix, 1, GL_FALSE, value_ptr( get_mv_matrix(_camera_ptr, m_shape.model * m_shape.shape) ));
        glUniformMatrix4fv(uniforms.proj_matrix, 1, GL_FALSE, value_ptr(_camera_ptr->GetProjectionMatrix()));
    }else{
        if (is_moveable){
            // Note: the rotation is mainly for z-axis rotation
            // Note 2: The tranalation/rotation/scale is based on the "center" of the image
            // m_shape.model = translateMatrix * rotateMatrix * scaleMatrix;
            glUniformMatrix4fv(uniforms.mv_matrix, 1, GL_FALSE, value_ptr( m_shape.model * m_shape.shape));
        }else{
            // background
            // Nothing, for saving computation
        }
    }
    // Color transform will resulted in feelable delay in display.
    glUniform4fv(uniforms.color_vec4, 1, value_ptr(color_vec4) );
    //
    glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
    //
    _program_ptr->CloseProgram();
}

void rmColorBoard::Reshape(const glm::ivec2 & viewport_size_in){
    _viewport_size = viewport_size_in;
    updateBoardGeo();
}
