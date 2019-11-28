#ifndef SHADER_H
#define SHADER_H

#include "Common.h"


class Shader
{
public:
	GLuint GetID();
	bool isLoaded();
	void LoadShader(const char* fileName,int _shaderType);
    void LoadShader(std::string fileName,int _shaderType);
	void Delete();

private:
	GLuint id;
	int type;
	bool loaded;
};

//
class ShaderProgram
{

public:

	ShaderProgram(); // Create a new program
    // Step 1
    void AttachShader(GLuint shaderId); // Method 1: Attached a loaded shader
    void AttachShader(std::string fileName,int _shaderType); // Method 2: Load and attached a shader
    // Step 2
	void LinkProgram();
    // On each iteration
	void UseProgram();
    void CloseProgram();
    //
    GLuint GetID();
	void Delete();

private:
	GLuint id;
	bool linked;
    GLuint CreateProgram();
};
#endif  // SHADER_H
