#include "Shader.h"


void Shader::LoadShader(const char* fileName, int shaderType){
    std::cout << "Start loading shader\n";
	GLuint s = glCreateShader(shaderType);
	char** source = Common::LoadShaderSource(fileName);
	glShaderSource(s, 1, source, NULL);
	Common::FreeShaderSource(source);
	glCompileShader(s);
    Common::ShaderLog(s);

	id = s;
	type = shaderType;
	loaded = true;
}
void Shader::LoadShader(std::string fileName, int shaderType){
    LoadShader(fileName.c_str(), shaderType);
}

void Shader::Delete(){
  if (!loaded)
  {
    return;
  }
  loaded = false;
	glDeleteShader(id);
}

GLuint Shader::GetID(){
	return id;
}

bool Shader::isLoaded(){
	return loaded;
}




ShaderProgram::ShaderProgram()
{
	linked = false;
    CreateProgram();
}
//
void ShaderProgram::AttachShader(GLuint shaderId){
	glAttachShader(id, shaderId);
}
void ShaderProgram::AttachShader(std::string fileName,int _shaderType){ // Method 2: Load and attached a shader
    // Load shaders
    Shader _s;
    _s.LoadShader(fileName, _shaderType);
    AttachShader(_s.GetID());
    _s.Delete();
}
void ShaderProgram::LinkProgram(){
	glLinkProgram(id);
	int iLinkStatus;
	glGetProgramiv(id, GL_LINK_STATUS, &iLinkStatus);
	linked = iLinkStatus == GL_TRUE;
}

void ShaderProgram::UseProgram(){
  if (linked)
  {
    glUseProgram(id);
  }
}
void ShaderProgram::CloseProgram(){
    glUseProgram(0); // Use the default shader program (compactble with openGL ver 1.0 commands)
}

//
GLuint ShaderProgram::GetID(){
	return id;
}

void ShaderProgram::Delete(){
  if (!linked)
  {
    return;
  }
  linked = false;
	glDeleteProgram(id);
}
// Private methods
GLuint ShaderProgram::CreateProgram(){
	id = glCreateProgram();
	return id;
}
