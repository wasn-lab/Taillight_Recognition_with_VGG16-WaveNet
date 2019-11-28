#ifndef RM_TEXT_3D_H
#define RM_TEXT_3D_H

#include "rmBaseModel.h"


#include <map> // std::map
// FreeType
#include <ft2build.h>
#include FT_FREETYPE_H



class rmText3D : public rmBaseModel
{
public:
    rmText3D(std::string _path_Assets_in, int _ROS_topic_id_in);
    //
	void Update(float dt);
    void Update(ROS_INTERFACE &ros_interface);
    void Update(ROS_API &ros_api);
	void Render(std::shared_ptr<ViewManager> &_camera_ptr);

protected:
    void Init();
    virtual void LoadModel();
    //
    int _ROS_topic_id;
    // std::shared_ptr< msgs::LidRoi > box3d_out_ptr;
    // ros::Time msg_time;


    void RenderText(const std::string &text, GLfloat x, GLfloat y, GLfloat scale, glm::vec3 color);

private:
    // model info
    struct Shape{
        GLuint vao;
        GLuint vbo;
        GLuint ebo;
        GLuint m_texture;
        //
        int indexCount;

        glm::mat4 model;
    };
    Shape m_shape;

    // The structure for point
    struct vertex_p_c
	{
		glm::vec3     position;
		glm::vec3     color;
	};

    struct vertex_p_tc
	{
		glm::vec2     xyposition;
		glm::vec2     texcor;
	};

    //uniform id
	struct
	{
		GLint  mv_matrix;
		GLint  proj_matrix;
        GLint  textColor;
	} uniforms;


    // Text
    /// Holds all state information relevant to a character as loaded using FreeType
    struct Character {
        GLuint TextureID;   // ID handle of the glyph texture
        glm::ivec2 Size;    // Size of glyph
        glm::ivec2 Bearing;  // Offset from baseline to left/top of glyph
        GLuint Advance;    // Horizontal offset to advance to next glyph
    };
    std::map<GLchar, Character> Characters;

    int _font_size;

    std::string text_current;


};

#endif // RM_TEXT_3D_H
