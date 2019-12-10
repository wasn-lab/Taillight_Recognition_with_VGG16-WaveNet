#ifndef COMMON_H
#define COMMON_H

#ifdef _MSC_VER
    #include "GLEW/glew.h"
    #include "FreeGLUT/freeglut.h"
    #include <direct.h>
#else
    // #include <OpenGL/gl3.h>
    #include <GL/glew.h> // <-- Added for Ubuntu, by Benson
    #include <GL/glut.h> // <-- Changed from 'GLUT/glut.h' to 'GL/glut.h' for Ubuntu, by Benson
    #include <unistd.h>
#endif

#ifndef TINYOBJLOADER_IMPLEMENTATION
#define TINYOBJLOADER_IMPLEMENTATION
#include "./TinyOBJ/tiny_obj_loader.h"
#endif

#define GLM_SWIZZLE
#include "../include/GLM/glm/glm.hpp"
#include "../include/GLM/glm/gtc/matrix_transform.hpp"
#include "../include/GLM/glm/gtc/type_ptr.hpp"
#include "../include/GLM/glm/gtx/rotate_vector.hpp"
#include "../include/GLM/glm/gtx/norm.hpp"
#include "../include/GLM/glm/gtx/quaternion.hpp"

//
/*
#define STB_IMAGE_IMPLEMENTATION
#include "../include/STB/stb_image.h"
*/
//

#ifdef _MSC_VER
#define __FILENAME__ (strrchr(__FILE__, '\\') ? strrchr(__FILE__, '\\') + 1 : __FILE__)
#define __FILEPATH__(x) ((std::string(__FILE__).substr(0, std::string(__FILE__).rfind('\\'))+(x)).c_str())
#else
#define __FILENAME__ (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define __FILEPATH__(x) ((std::string(__FILE__).substr(0, std::string(__FILE__).rfind('/'))+(x)).c_str())
#endif

#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <algorithm>
#include <fstream>

//
#include <memory> // <-- this is for std::shared_ptr
//

#define deg2rad(x) ((x)*((3.1415926f)/(180.0f)))
#define rad2deg(x) ((180.0f) / ((x)*(3.1415926f)))

typedef struct _TextureData
{
	_TextureData() : width(0), height(0), data(0) {}
	int width;
	int height;
	unsigned char* data;
} TextureData;

struct _Shape
{
	_Shape() : vao(0), vbo(0), vboTex(0), ebo(0), p_normal(0), materialId(0), indexCount(0), m_texture(0),
		position(glm::vec3(0, 0, 0)), rotation(glm::vec3(0, 0, 0)), scale(glm::vec3(1, 1, 1)) {}

	GLuint vao;
	GLuint vbo;
	GLuint vboTex;
	GLuint ebo;

	GLuint p_normal;
	int materialId;
	int indexCount;
	GLuint m_texture;

	glm::vec3 position;
	glm::vec3 rotation;
	glm::vec3 scale;

	glm::mat4 getTransformationMatrix();

};

class Common
{
public:
	static void DumpInfo(void);
	static void ShaderLog(GLuint shader);
	static void PrintGLError();
	static TextureData Load_png(const char* path);
	static char** LoadShaderSource(const char* file);
	static void FreeShaderSource(char** srcp);
    static void print_out_mat4(glm::mat4 & m_in); // test
};



#endif  // COMMON_H
