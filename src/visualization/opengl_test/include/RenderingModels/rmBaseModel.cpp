#include "rmBaseModel.h"


rmBaseModel::rmBaseModel():
    // Note: The following field will be shared across all derived classes
    //       However, each serived class can modify this on their own.
    flag_enable(true),
    _path_Assets_sub_dir(""),
    _path_Shaders_sub_dir("Shaders/"),
    _pose_modle_ref_by_world(1.0f), _tmp_pose_model_by_model_ref(1.0f),
    _viewport_size(1,1)
    // board_width(1.0), board_height(1.0), board_aspect_ratio(1.0),
    // board_shape_mode(0)
{
    // The derived class will call this instead of the other constructor if we don't add the constructor in field of derived class.
}
rmBaseModel::rmBaseModel(std::string _path_Assets_in, std::string modelFile, std::string textFile):
    _path_Assets_sub_dir(""),
    _path_Shaders_sub_dir("Shaders/"),
    _pose_modle_ref_by_world(1.0f), _tmp_pose_model_by_model_ref(1.0f),
    _viewport_size(1,1)
    // board_width(1.0), board_height(1.0), board_aspect_ratio(1.0),
    // board_shape_mode(0)
{
    init_paths(_path_Assets_in);
    objName = modelFile;
    textName = textFile;
	Init();
}
rmBaseModel::~rmBaseModel(){

}

void rmBaseModel::Init(){

    // //
	// _program_ptr.reset( new ShaderProgram() );
    // // Load shaders
    // _program_ptr->AttachShader(get_full_Shader_path("ModelLoader.vs.glsl"), GL_VERTEX_SHADER);
    // _program_ptr->AttachShader(get_full_Shader_path("ModelLoader.fs.glsl"), GL_FRAGMENT_SHADER);
    // // Link _program_ptr
	// _program_ptr->LinkProgram();
    // //
    //
	// // Cache uniform variable id
	// uniforms.proj_matrix = glGetUniformLocation(_program_ptr->GetID(), "proj_matrix");
	// uniforms.mv_matrix = glGetUniformLocation(_program_ptr->GetID(), "mv_matrix");
    //
    // // Init model matrices
	// m_shape.model = glm::mat4();
    // attach_pose_model_by_model_ref_ptr(m_shape.model); // For adjusting the model pose by public methods
	// translateMatrix = glm::mat4();
	// rotateMatrix = glm::mat4();
	// scaleMatrix = glm::mat4();

	// _program_ptr->UseProgram();
	///////////////////////////

	//Load model to shader _program_ptr
	LoadModel();
}

void rmBaseModel::LoadModel(){
	// std::vector<tinyobj::shape_t> shapes;
	// std::vector<tinyobj::material_t> materials;
    //
	// std::string err;
    //
    // std::cout << "Start loading <" << objName << ">\n";
	// bool ret = tinyobj::LoadObj(shapes, materials, err, get_full_Assets_path(objName).c_str());
	// if (err.size()>0)
	// {
	// 	printf("Load Models Fail! Please check the solution path");
	// 	return;
	// }
    //
    // std::cout << "Load models success ! Shapes size = " <<  shapes.size() << ", Maerial size = " << materials.size() << "\n";
    //
	// /*
	// *Bind model data
	// */
	// for (int i = 0; i < shapes.size(); i++)
	// {
	// 	glGenVertexArrays(1, &m_shape.vao);
	// 	glBindVertexArray(m_shape.vao);
    //
	// 	glGenBuffers(3, &m_shape.vbo); // <-- Note: The 2nd arg. of glGenBuffers() should be the array, this operation will save buffer name to vbo, vboTex, and ebo
    //     // Hoever, the above usage is not recommanded...
    //     glGenBuffers(1, &m_shape.p_normal);
	// 	glBindBuffer(GL_ARRAY_BUFFER, m_shape.vbo);
	// 	glBufferData(GL_ARRAY_BUFFER, shapes[i].mesh.positions.size() * sizeof(float) + shapes[i].mesh.normals.size() * sizeof(float), NULL, GL_STATIC_DRAW);
    //
	// 	glBufferSubData(GL_ARRAY_BUFFER, 0, shapes[i].mesh.positions.size() * sizeof(float), &shapes[i].mesh.positions[0]);
	// 	glBufferSubData(GL_ARRAY_BUFFER, shapes[i].mesh.positions.size() * sizeof(float), shapes[i].mesh.normals.size() * sizeof(float), &shapes[i].mesh.normals[0]);
    //
	// 	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
	// 	glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 0, (void *)(shapes[i].mesh.positions.size() * sizeof(float)));
    //
	// 	glBindBuffer(GL_ARRAY_BUFFER, m_shape.p_normal);
	// 	glBufferData(GL_ARRAY_BUFFER, shapes[i].mesh.normals.size() * sizeof(float), shapes[i].mesh.normals.data(), GL_STATIC_DRAW);
	// 	glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 0, 0);
    //
	// 	glBindBuffer(GL_ARRAY_BUFFER, m_shape.vboTex);
	// 	glBufferData(GL_ARRAY_BUFFER, shapes[i].mesh.texcoords.size() * sizeof(float), shapes[i].mesh.texcoords.data(), GL_STATIC_DRAW);
	// 	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, 0);
	// 	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_shape.ebo);
	// 	glBufferData(GL_ELEMENT_ARRAY_BUFFER, shapes[i].mesh.indices.size() * sizeof(unsigned int), shapes[i].mesh.indices.data(), GL_STATIC_DRAW);
	// 	m_shape.materialId = shapes[i].mesh.material_ids[0];
	// 	m_shape.indexCount = shapes[i].mesh.indices.size();
    //
	// 	glEnableVertexAttribArray(0);
	// 	glEnableVertexAttribArray(1);
	// 	glEnableVertexAttribArray(2);
	// }
    //
	// /*
	// *Texture setting
	// */
    //
	// //Load texture data from file
    // std::cout << "start loading <" << textName << ">\n";
	// TextureData tdata = Common::Load_png(get_full_Assets_path(textName).c_str());
    //
	// //Generate empty texture
	// glGenTextures(1, &m_shape.m_texture);
	// glBindTexture(GL_TEXTURE_2D, m_shape.m_texture);
    //
	// //Do texture setting
	// glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, tdata.width, tdata.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, tdata.data);
	// glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	// glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	// glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	// glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    //
    // std::cout << "Load texture success!\n";
}

void rmBaseModel::Update(float dt){
    // Update the data (buffer variables) here
}
void rmBaseModel::Update(ROS_INTERFACE &ros_interface){
    // Update the data (buffer variables) here
}
void rmBaseModel::Update(ROS_API &ros_api){
    // Update the data (buffer variables) here
}
void rmBaseModel::Render(std::shared_ptr<ViewManager> &_camera_ptr){
	//Update shaders' input variable
	///////////////////////////
	glBindVertexArray(m_shape.vao);
	_program_ptr->UseProgram();

	// m_shape.model = translateMatrix * rotateMatrix * scaleMatrix;
	glBindTexture(GL_TEXTURE_2D, m_shape.m_texture);
	glUniformMatrix4fv(uniforms.mv_matrix, 1, GL_FALSE, value_ptr(get_mv_matrix(_camera_ptr, m_shape.model)));
	glUniformMatrix4fv(uniforms.proj_matrix, 1, GL_FALSE, value_ptr(_camera_ptr->GetProjectionMatrix()));

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_shape.ebo);
	glDrawElements(GL_TRIANGLES, m_shape.indexCount, GL_UNSIGNED_INT, 0);

	///////////////////////////
    _program_ptr->CloseProgram();
}

// Matrix operation
//------------------------------------------------//
// Legacy "Pre-" operations
void rmBaseModel::Translate(const glm::vec3 &vec){
    preTranslate(vec);
}
void rmBaseModel::Rotate(const glm::vec3 &axis, float angle){
    preRotate(axis, angle);
}
void rmBaseModel::Scale(const glm::vec3 &vec){
    preScale(vec);
}
// "Pre-" operations
void rmBaseModel::preTranslate(const glm::vec3 &vec){
	translateMatrix = translate(translateMatrix, vec);
    update_pose_model_by_model_ref();
}
void rmBaseModel::preRotate(const glm::vec3 &axis, float angle){
	rotateMatrix = rotate(rotateMatrix, angle, axis);
    update_pose_model_by_model_ref();
}
void rmBaseModel::preScale(const glm::vec3 &vec){
	scaleMatrix = scale(scaleMatrix, vec);
    update_pose_model_by_model_ref();
}
// "Post-" operations
void rmBaseModel::postTranslate(const glm::vec3 &vec){
	translateMatrix = translate(glm::mat4(1.0), vec) * translateMatrix;
    update_pose_model_by_model_ref();
}
void rmBaseModel::postRotate(const glm::vec3 &axis, float angle){
	rotateMatrix = rotate(glm::mat4(1.0), angle, axis) * rotateMatrix;
    update_pose_model_by_model_ref();
}
void rmBaseModel::postScale(const glm::vec3 &vec){
	scaleMatrix = scale(glm::mat4(1.0), vec) * scaleMatrix;
    update_pose_model_by_model_ref();
}
//
void rmBaseModel::update_pose_model_by_model_ref(){
    if (_pose_model_by_model_ref_ptr_list.size() > 0){
        glm::mat4 _tmp_m = translateMatrix * rotateMatrix * scaleMatrix;
        for (size_t i=0; i < _pose_model_by_model_ref_ptr_list.size(); ++i){
            *(_pose_model_by_model_ref_ptr_list[i]) = _tmp_m;
        }
    }
}
//------------------------------------------------//
void rmBaseModel::set_pose_modle_ref_by_world(glm::mat4 pose_in){
    _pose_modle_ref_by_world = pose_in;
}
glm::mat4 rmBaseModel::get_pose_modle_ref_by_world(){
    return _pose_modle_ref_by_world;
}
glm::mat4 rmBaseModel::get_mv_matrix(const std::shared_ptr<ViewManager> &_camera_ptr, const glm::mat4 &_model_M){
    // Get the model-view matrix
    // return (_camera_ptr->GetViewMatrix() * _camera_ptr->GetModelMatrix() * _model_M);
    // return (_camera_ptr->GetViewMatrix() * _camera_ptr->GetModelMatrix() * _pose_modle_ref_by_world * _model_M);
    return (_camera_ptr->GetModelViewMatrix() * _pose_modle_ref_by_world * _model_M);
}
bool rmBaseModel::init_paths(std::string _path_Assets_in){
    // Fix the path
    if (!_path_Assets_in.empty() && _path_Assets_in.back() != '/'){
        _path_Assets_in += "/";
    }
    if (!_path_Assets_sub_dir.empty() && _path_Assets_sub_dir.back() != '/'){
        _path_Assets_sub_dir += "/";
    }
    if (!_path_Shaders_sub_dir.empty() && _path_Shaders_sub_dir.back() != '/'){
        _path_Shaders_sub_dir += "/";
    }
    //
    _path_Assets = _path_Assets_in + _path_Assets_sub_dir;
    _path_Shaders = _path_Assets_in + _path_Shaders_sub_dir;
    //
    return true;
}
std::string rmBaseModel::get_full_Assets_path(std::string Assets_name_in){
    std::string full_p;
    full_p = _path_Assets + Assets_name_in;
    std::cout << "Assets = <" << full_p << ">\n";
    return full_p;
}
std::string rmBaseModel::get_full_Shader_path(std::string Shader_name_in){
    std::string full_p;
    full_p = _path_Shaders + Shader_name_in;
    std::cout << "shader = <" << full_p << ">\n";
    return full_p;
}
glm::mat4 rmBaseModel::ROStf2GLMmatrix(const geometry_msgs::TransformStamped &ros_tf){
    glm::quat _rot_q(ros_tf.transform.rotation.w, ros_tf.transform.rotation.x, ros_tf.transform.rotation.y, ros_tf.transform.rotation.z);
    glm::mat4 _rot_m = glm::toMat4(_rot_q);
    glm::vec3 _trans_v(ros_tf.transform.translation.x, ros_tf.transform.translation.y, ros_tf.transform.translation.z);
    glm::mat4 _trans_m(1.0);
    _trans_m = translate(_trans_m, _trans_v);
    return (_trans_m * _rot_m);
}
