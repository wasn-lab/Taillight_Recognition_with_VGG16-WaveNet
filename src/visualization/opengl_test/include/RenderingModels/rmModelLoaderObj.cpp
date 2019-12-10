#include "rmModelLoaderObj.h"

rmModelLoaderObj::rmModelLoaderObj(std::string _path_Assets_in, std::string modelFile, std::string textFile){
    init_paths(_path_Assets_in);
    objName = modelFile;
    textName = textFile;
	Init();
}
void rmModelLoaderObj::Init(){
    //
	_program_ptr.reset( new ShaderProgram() );
    // Load shaders
    _program_ptr->AttachShader(get_full_Shader_path("ModelLoader.vs.glsl"), GL_VERTEX_SHADER);
    _program_ptr->AttachShader(get_full_Shader_path("ModelLoader.fs.glsl"), GL_FRAGMENT_SHADER);
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
void rmModelLoaderObj::LoadModel(){
    std::vector<tinyobj::shape_t> shapes;
	std::vector<tinyobj::material_t> materials;

	std::string err;

    std::cout << "Start loading <" << objName << ">\n";
	bool ret = tinyobj::LoadObj(shapes, materials, err, get_full_Assets_path(objName).c_str());
	if (err.size()>0)
	{
		printf("Load Models Fail! Please check the solution path");
		return;
	}

    std::cout << "Load models success ! Shapes size = " <<  shapes.size() << ", Maerial size = " << materials.size() << "\n";

	/*
	*Bind model data
	*/
	for (int i = 0; i < shapes.size(); i++)
	{
		glGenVertexArrays(1, &m_shape.vao);
		glBindVertexArray(m_shape.vao);

		glGenBuffers(3, &m_shape.vbo); // <-- Note: The 2nd arg. of glGenBuffers() should be the array, this operation will save buffer name to vbo, vboTex, and ebo
        // Hoever, the above usage is not recommanded...
        glGenBuffers(1, &m_shape.p_normal);
		glBindBuffer(GL_ARRAY_BUFFER, m_shape.vbo);
		glBufferData(GL_ARRAY_BUFFER, shapes[i].mesh.positions.size() * sizeof(float) + shapes[i].mesh.normals.size() * sizeof(float), NULL, GL_STATIC_DRAW);

		glBufferSubData(GL_ARRAY_BUFFER, 0, shapes[i].mesh.positions.size() * sizeof(float), &shapes[i].mesh.positions[0]);
		glBufferSubData(GL_ARRAY_BUFFER, shapes[i].mesh.positions.size() * sizeof(float), shapes[i].mesh.normals.size() * sizeof(float), &shapes[i].mesh.normals[0]);

		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
		glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 0, (void *)(shapes[i].mesh.positions.size() * sizeof(float)));

		glBindBuffer(GL_ARRAY_BUFFER, m_shape.p_normal);
		glBufferData(GL_ARRAY_BUFFER, shapes[i].mesh.normals.size() * sizeof(float), shapes[i].mesh.normals.data(), GL_STATIC_DRAW);
		glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 0, 0);

		glBindBuffer(GL_ARRAY_BUFFER, m_shape.vboTex);
		glBufferData(GL_ARRAY_BUFFER, shapes[i].mesh.texcoords.size() * sizeof(float), shapes[i].mesh.texcoords.data(), GL_STATIC_DRAW);
		glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, 0);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_shape.ebo);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, shapes[i].mesh.indices.size() * sizeof(unsigned int), shapes[i].mesh.indices.data(), GL_STATIC_DRAW);
		m_shape.materialId = shapes[i].mesh.material_ids[0];
		m_shape.indexCount = shapes[i].mesh.indices.size();

		glEnableVertexAttribArray(0);
		glEnableVertexAttribArray(1);
		glEnableVertexAttribArray(2);
	}

	/*
	*Texture setting
	*/

	//Load texture data from file
    std::cout << "start loading <" << textName << ">\n";
	TextureData tdata = Common::Load_png(get_full_Assets_path(textName).c_str());

	//Generate empty texture
	glGenTextures(1, &m_shape.m_texture);
	glBindTexture(GL_TEXTURE_2D, m_shape.m_texture);

	//Do texture setting
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, tdata.width, tdata.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, tdata.data);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    std::cout << "Load texture success!\n";

}
void rmModelLoaderObj::Update(float dt){
    // Update the data (buffer variables) here
}
void rmModelLoaderObj::Update(ROS_INTERFACE &ros_interface){
    // Update the data (buffer variables) here
}
void rmModelLoaderObj::Update(ROS_API &ros_api){
    // Update the data (buffer variables) here
}


void rmModelLoaderObj::Render(std::shared_ptr<ViewManager> &_camera_ptr){
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
