#include "rmSweepingObject.h"
#include <math.h>       /* cos */


rmSweepingObject::rmSweepingObject(std::string _path_Assets_in, std::string frame_id_in, int draw_mode_in):
    _frame_id(frame_id_in),
    draw_mode(draw_mode_in),
    is_close_loop(true)
{
    _path_Shaders_sub_dir += "SweepObject/";
    init_paths(_path_Assets_in);
    //
    _max_num_vertex_of_curve = 100; // 100;
    _max_num_vertex_of_shape = 50;
    //
	Init();
}
rmSweepingObject::rmSweepingObject(std::string _path_Assets_in, int _ROS_topic_id_in, int draw_mode_in):
    _ROS_topic_id(_ROS_topic_id_in),
    draw_mode(draw_mode_in),
    is_close_loop(true)
{
    _path_Shaders_sub_dir += "SweepObject/";
    init_paths(_path_Assets_in);
    //
    _max_num_vertex_of_curve = 100;
    _max_num_vertex_of_shape = 50;
    //
	Init();
}
void rmSweepingObject::Init(){
    //
	_program_ptr.reset( new ShaderProgram() );
    // Load shaders
    _program_ptr->AttachShader(get_full_Shader_path("SweepObject.vs.glsl"), GL_VERTEX_SHADER);
    if (draw_mode == 1)
    {  // Section draw
      _program_ptr->AttachShader(get_full_Shader_path("SweepObject.gs.section.glsl"), GL_GEOMETRY_SHADER);
    }
    else
    {
      _program_ptr->AttachShader(get_full_Shader_path("SweepObject.gs.glsl"), GL_GEOMETRY_SHADER);
    }
    _program_ptr->AttachShader(get_full_Shader_path("SweepObject.fs.glsl"), GL_FRAGMENT_SHADER);
    // Link _program_ptr
	_program_ptr->LinkProgram();
    //

    // Cache uniform variable id
	uniforms.proj_matrix = glGetUniformLocation(_program_ptr->GetID(), "proj_matrix");
	uniforms.mv_matrix = glGetUniformLocation(_program_ptr->GetID(), "mv_matrix");
    //
    uniforms.lookat_matrix.resize(_max_num_vertex_of_curve);
    for (int i = 0; i < _max_num_vertex_of_curve; i++) {
		uniforms.lookat_matrix[i] = glGetUniformLocation(_program_ptr->GetID(), std::string("lookat_matrix[" + std::to_string(i) + "]").c_str());
	}
    //
    uniforms.section_vertexes.resize(_max_num_vertex_of_shape);
    for (int i = 0; i < _max_num_vertex_of_shape; i++) {
		uniforms.section_vertexes[i] = glGetUniformLocation(_program_ptr->GetID(), std::string("section_vertexes[" + std::to_string(i) + "]").c_str());
	}
    uniforms._num_vertex_of_shape = glGetUniformLocation(_program_ptr->GetID(), "_num_vertex_of_shape");
    uniforms.shape_mode = glGetUniformLocation(_program_ptr->GetID(), "shape_mode");
    //

    // Init model matrices
	m_shape.model = glm::mat4(1.0);
    attach_pose_model_by_model_ref_ptr(m_shape.model); // For adjusting the model pose by public methods
    _line_width = 1.0f;

    //
    _color_head = glm::vec3(1.0f, 0.5f, 0.0f);
    _color_tail = glm::vec3(0.0f, 0.5f, 1.0f);

    //Load model to shader _program_ptr
	LoadModel();

}
void rmSweepingObject::LoadModel(){

    // Insert section points
    //-------------------------------------------------//
    section_vertexes.resize(4);
    section_vertexes[0] = glm::vec3(0.0f, -1.0f, -1.0f);
    section_vertexes[1] = glm::vec3(0.0f, -1.0f, 1.0f);
    section_vertexes[2] = glm::vec3(0.0f, 1.0f, 1.0f);
    section_vertexes[3] = glm::vec3(0.0f, 1.0f, -1.0f);
    //
    glm::mat4 _delta_T = glm::scale(glm::mat4(1.0), glm::vec3(1.0f, 0.01f, 1.0f) );
    for (size_t i=0; i < section_vertexes.size(); ++i){
        section_vertexes[i] = (_delta_T * glm::vec4(section_vertexes[i], 1.0f)).xyz();
    }
    //-------------------------------------------------//

    // Insert curve points (example)
    //-------------------------------------------------//
    _curve_Points.push_back( glm::vec3(0.00, 0.00, 0.00) );
    for (int i=1; i < 20; ++i){
        _curve_Points.push_back( _curve_Points[i-1] + glm::vec3(1.0f, 0.0f + float(i/10.0f), 0.0f) );
    }
    m_shape.indexCount = _curve_Points.size();
    //-------------------------------------------------//

    // Update lookat matrix
    _update_lookat_matrix_list();
    //


    // Setup OpenGL
    glGenVertexArrays(1, &m_shape.vao);
	glBindVertexArray(m_shape.vao);

    glGenBuffers(1, &m_shape.vbo);
	glBindBuffer(GL_ARRAY_BUFFER, m_shape.vbo);
    glBufferData(GL_ARRAY_BUFFER, _max_num_vertex_of_curve * sizeof(vertex_p_c), NULL, GL_DYNAMIC_DRAW); // test, change to dynamic draw to assign point cloud
    // Directly assign data to memory of GPU
    //--------------------------------------------//
	vertex_p_c * vertex_ptr = (vertex_p_c *)glMapBufferRange(GL_ARRAY_BUFFER, 0, _max_num_vertex_of_curve * sizeof(vertex_p_c), GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT);
    float _size = 1.0f;
    for (size_t i = 0; i < m_shape.indexCount; i++)
	{
		vertex_ptr[i].position = _size*_curve_Points[i];
		vertex_ptr[i].color[0] = 1.0f * (float(m_shape.indexCount - i))/ float(m_shape.indexCount); //
		vertex_ptr[i].color[1] = 0.5f; //
		vertex_ptr[i].color[2] = 1.0f * (float(i))/ float(m_shape.indexCount); //
	}
	glUnmapBuffer(GL_ARRAY_BUFFER);
    //--------------------------------------------//

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(vertex_p_c), NULL);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(vertex_p_c), (void *)sizeof(glm::vec3));
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);

}
void rmSweepingObject::Update(float dt){
    // Update the data (buffer variables) here
}
void rmSweepingObject::Update(ROS_INTERFACE &ros_interface){
    // Update the data (buffer variables) here
}

void rmSweepingObject::Update(ROS_API &ros_api){
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


void rmSweepingObject::Render(std::shared_ptr<ViewManager> &_camera_ptr){

    glBindVertexArray(m_shape.vao);

	_program_ptr->UseProgram();
    // The transformation matrices and projection matrices
    glUniformMatrix4fv(uniforms.mv_matrix, 1, GL_FALSE, value_ptr( get_mv_matrix(_camera_ptr, m_shape.model) ));
    glUniformMatrix4fv(uniforms.proj_matrix, 1, GL_FALSE, value_ptr(_camera_ptr->GetProjectionMatrix()));
    //
    for (int i = 0; i < m_shape.indexCount; i++) {
		glUniformMatrix4fv( uniforms.lookat_matrix[i], 1, GL_FALSE, value_ptr(lookat_matrix_list[i]) );
	}
    for (int i = 0; i < section_vertexes.size(); i++) {
		glUniform3fv( uniforms.section_vertexes[i], 1, value_ptr(section_vertexes[i]) );
	}
    glUniform1i(uniforms._num_vertex_of_shape, int(section_vertexes.size()) );
    glUniform1i(uniforms.shape_mode, int(is_close_loop) );
    // Setting
    glLineWidth(_line_width);
    // Draw the elements
    glDrawArrays(GL_LINE_STRIP, 0, m_shape.indexCount); // draw part of points
    //--------------------------------//
    glLineWidth(1.0); // default line width
    _program_ptr->CloseProgram();
}


void rmSweepingObject::update_GL_data(){

}



void rmSweepingObject::insert_curve_Points(const std::vector<glm::vec3> &curve_Points_in){
    _curve_Points = curve_Points_in;
    if (_curve_Points.size() > _max_num_vertex_of_curve){
        std::cout << "Warn: Too many _curve_Points: size = " << _curve_Points.size() << "\n";
        _curve_Points.resize(_max_num_vertex_of_curve);
    }
    m_shape.indexCount = _curve_Points.size();
    _update_lookat_matrix_list();

    // vao vbo
    glBindVertexArray(m_shape.vao);
    glBindBuffer(GL_ARRAY_BUFFER, m_shape.vbo); // Start to use the buffer
    // Directly assign data to memory of GPU
    //--------------------------------------------//
	vertex_p_c * vertex_ptr = (vertex_p_c *)glMapBufferRange(GL_ARRAY_BUFFER, 0, _max_num_vertex_of_curve * sizeof(vertex_p_c), GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT);
    for (size_t i = 0; i < m_shape.indexCount; i++)
	{
		vertex_ptr[i].position = _curve_Points[i];
        float _ratio = (float(i))/ float(m_shape.indexCount);
        vertex_ptr[i].color = (1.0f - _ratio) * _color_head + _ratio * _color_tail;
	}
	glUnmapBuffer(GL_ARRAY_BUFFER);
    //--------------------------------------------//
}


// Calculate lookat matrices
void rmSweepingObject::_update_lookat_matrix_list(){
    // Calculate rotaion matrices and scales
    // Calculate elements of lookat_matrix_list (new method, automatically generated according to curve)
    lookat_matrix_list.resize(_max_num_vertex_of_curve);
    lookat_matrix_list[0] = glm::mat4(1.0f);
    //
    if ( _curve_Points.size() > 1){
        glm::mat4 _accumulated_T(1.0f);
        glm::vec3 _d_pre = glm::normalize(_curve_Points[1] - _curve_Points[0]);
        for (size_t i=1; i < (_curve_Points.size() - 1); ++i){
            // direction
            glm::vec3 _d_next = glm::normalize( _curve_Points[i+1] - _curve_Points[i] );
            // angle
            GLfloat _cos_v = glm::dot(_d_pre, _d_next);
            if ( _cos_v > 1.0f){
                _cos_v = 1.0f;
            }
            GLfloat _angle = glm::acos( _cos_v ); // Note: ||_d_pre|| = ||_d_next|| = 1
            //
            if ( _angle < 0.0017 ){ // 0.1 deg
                // std::cout << "The angle between two lookat_matrix is too small.\n";
                lookat_matrix_list[i] = _accumulated_T; // lookat_matrix_list[i-1];
            }else{
                // axis - cross
                glm::vec3 _axis = glm::cross(_d_pre, _d_next);
                glm::vec3 _axis_n = glm::normalize(_axis);
                //
                glm::mat4 _delta_rot = glm::rotate(glm::mat4(1.0f), _angle/2.0f, _axis_n);
                lookat_matrix_list[i] = _delta_rot * _accumulated_T;
                _accumulated_T = _delta_rot * lookat_matrix_list[i];
                //
            }
            _d_pre = _d_next;
        }
        lookat_matrix_list[_curve_Points.size() - 1] = _accumulated_T;
    }


    /*
    if ( _curve_Points.size() > 1){
        glm::vec3 _d_pre = glm::normalize( _d_initial );
        for (size_t i=1; i < (_curve_Points.size() ); ++i){
            // direction
            glm::vec3 _d_next = glm::normalize( _curve_Points[i] - _curve_Points[i-1] );
            // angle
            GLfloat _angle = glm::acos( glm::dot(_d_pre, _d_next) ); // Note: ||_d_pre|| = ||_d_next|| = 1
            if (_angle < 0.000017){ // 0.001 deg
                lookat_matrix_list[i] = lookat_matrix_list[i-1];
            }else{
                // axis - cross
                glm::vec3 _axis = glm::cross(_d_pre, _d_next);
                glm::vec3 _axis_n = glm::normalize(_axis);
                //
                glm::mat4 _delta_rot = glm::rotate(glm::mat4(1.0f), _angle, _axis_n);
                lookat_matrix_list[i] = _delta_rot * lookat_matrix_list[i-1];
            }
            //
            _d_pre = _d_next;
        }
    }
    */
}
