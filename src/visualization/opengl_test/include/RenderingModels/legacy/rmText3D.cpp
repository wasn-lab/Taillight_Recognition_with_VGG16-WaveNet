#include "rmText3D.h"





rmText3D::rmText3D(std::string _path_Assets_in, int _ROS_topic_id_in):
    _ROS_topic_id(_ROS_topic_id_in)
{
    init_paths(_path_Assets_in);
	Init();
}
void rmText3D::Init(){
    //
	_program_ptr.reset( new ShaderProgram() );
    // Load shaders
    _program_ptr->AttachShader(get_full_Shader_path("Text3D.vs.glsl"), GL_VERTEX_SHADER);
    _program_ptr->AttachShader(get_full_Shader_path("Text3D.fs.glsl"), GL_FRAGMENT_SHADER);
    // Link _program_ptr
	_program_ptr->LinkProgram();
    //

    // Cache uniform variable id
	uniforms.proj_matrix = glGetUniformLocation(_program_ptr->GetID(), "proj_matrix");
	uniforms.mv_matrix = glGetUniformLocation(_program_ptr->GetID(), "mv_matrix");
    uniforms.textColor = glGetUniformLocation(_program_ptr->GetID(), "textColor");


    // Init model matrices
	m_shape.model = glm::mat4(1.0);
    attach_pose_model_by_model_ref_ptr(m_shape.model); // For adjusting the model pose by public methods

    // Current text
    text_current = "";

    //Load model to shader _program_ptr
	LoadModel();

}
void rmText3D::LoadModel(){


    // FreeType
    FT_Library ft;
    // All functions return a value different than 0 whenever an error occurred
    if (FT_Init_FreeType(&ft))
        std::cout << "ERROR::FREETYPE: Could not init FreeType Library" << std::endl;
    // Load font as face
    FT_Face face;
    // if (FT_New_Face(ft, "fonts/arial.ttf", 0, &face))
    if (FT_New_Face(ft, "/usr/share/fonts/truetype/freefont/FreeSans.ttf", 0, &face))
    // if (FT_New_Face(ft, "FreeSans.ttf", 0, &face))
        std::cout << "ERROR::FREETYPE: Failed to load font" << std::endl;



    // Set size to load glyphs as
    _font_size = 48;
    FT_Set_Pixel_Sizes(face, 0, _font_size);



    // Load first 128 characters of ASCII set
    //--------------------------------------//
    for (GLubyte c = 0; c < 128; c++)
    {
        // Load character glyph
        if (FT_Load_Char(face, c, FT_LOAD_RENDER))
        {
            std::cout << "ERROR::FREETYTPE: Failed to load Glyph" << std::endl;
            continue;
        }
        // Generate texture
        GLuint texture;
        glGenTextures(1, &texture);
        glBindTexture(GL_TEXTURE_2D, texture);

        // Note: normal it's 4-byte alignment
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glTexImage2D(
            GL_TEXTURE_2D,
            0,
            GL_ALPHA,
            face->glyph->bitmap.width,
            face->glyph->bitmap.rows,
            0,
            GL_ALPHA,
            GL_UNSIGNED_BYTE,
            face->glyph->bitmap.buffer
        );
        // Set texture options
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        // Now store character for later use
        Character character = {
            texture,
            glm::ivec2(face->glyph->bitmap.width, face->glyph->bitmap.rows),
            glm::ivec2(face->glyph->bitmap_left, face->glyph->bitmap_top),
            GLuint(face->glyph->advance.x)
        };
        Characters.insert(std::pair<GLchar, Character>(c, character));
    }
    glBindTexture(GL_TEXTURE_2D, 0);
    //--------------------------------------//


    // Destroy FreeType once we're finished
    FT_Done_Face(face);
    FT_Done_FreeType(ft);





    glGenVertexArrays(1, &m_shape.vao);
	glBindVertexArray(m_shape.vao);

    glGenBuffers(1, &m_shape.vbo);
	glBindBuffer(GL_ARRAY_BUFFER, m_shape.vbo);
    // glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * 6 * 4, NULL, GL_DYNAMIC_DRAW);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertex_p_tc) * 6, NULL, GL_DYNAMIC_DRAW);

    // glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(GLfloat), NULL);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, sizeof(vertex_p_tc), NULL);
    glEnableVertexAttribArray(0);

    // Close
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);


}
void rmText3D::Update(float dt){
    // Update the data (buffer variables) here
}
void rmText3D::Update(ROS_INTERFACE &ros_interface){
    // Update the data (buffer variables) here
}
void rmText3D::Update(ROS_API &ros_api){
    // Update the data (buffer variables) here
}
void rmText3D::Render(std::shared_ptr<ViewManager> &_camera_ptr){

    glBindVertexArray(m_shape.vao);

	_program_ptr->UseProgram();
    // m_shape.model = translateMatrix * rotateMatrix * scaleMatrix;
    // The transformation matrices and projection matrices
    glUniformMatrix4fv(uniforms.mv_matrix, 1, GL_FALSE, value_ptr( get_mv_matrix(_camera_ptr, m_shape.model) ));
    glUniformMatrix4fv(uniforms.proj_matrix, 1, GL_FALSE, value_ptr(_camera_ptr->GetProjectionMatrix()));

    RenderText("Hello world", 0.0, 0.0, 1.0, glm::vec3(1.0f, 1.0f, 0.0f));


    // Draw the element according to ebo
    // glDrawElements(GL_TRIANGLES, m_shape.indexCount, GL_UNSIGNED_INT, 0);
    // glDrawArrays(GL_TRIANGLES, 0, 3*5); // draw part of points
    //--------------------------------//
    _program_ptr->CloseProgram();
}




//-----------------------------------------------//
void rmText3D::RenderText(const std::string &text, GLfloat x, GLfloat y, GLfloat scale_in, glm::vec3 color)
{

    // Activate corresponding render state
    glUniform3f( uniforms.textColor, color.x, color.y, color.z);

    /*
    if (text_current == text){
        // Render quad
        glDrawArrays(GL_TRIANGLES, 0, 6);
        return;
    }
    text_current = text;
    */

    //--------------------------------------------------------//
    glActiveTexture(GL_TEXTURE0);

    //
    GLfloat scale = scale_in/GLfloat(_font_size);

    // Iterate through all characters
    std::string::const_iterator c;
    for (c = text.begin(); c != text.end(); c++)
    {
        Character ch = Characters[*c];

        GLfloat xpos = x + ch.Bearing.x * scale;
        GLfloat ypos = y - (ch.Size.y - ch.Bearing.y) * scale;

        GLfloat w = ch.Size.x * scale;
        GLfloat h = ch.Size.y * scale;
        // Update VBO for each character
        GLfloat vertices[6][4] = {
            { xpos,     ypos + h,   0.0, 0.0 },
            { xpos,     ypos,       0.0, 1.0 },
            { xpos + w, ypos,       1.0, 1.0 },

            { xpos,     ypos + h,   0.0, 0.0 },
            { xpos + w, ypos,       1.0, 1.0 },
            { xpos + w, ypos + h,   1.0, 0.0 }
        };

        // Update content of VBO memory
        glBindBuffer(GL_ARRAY_BUFFER, m_shape.vbo);
        // glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vertices), vertices); // Be sure to use glBufferSubData and not glBufferData


        // Directly assign data to memory of GPU
        //--------------------------------------------//
    	vertex_p_tc * vertex_ptr = (vertex_p_tc *)glMapBufferRange(GL_ARRAY_BUFFER, 0, 6 * sizeof(vertex_p_tc), GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT);
    	for (size_t i = 0; i < 6; i++)
    	{
    		vertex_ptr[i].xyposition[0] = vertices[i][0];
    		vertex_ptr[i].xyposition[1] = vertices[i][1];
    		vertex_ptr[i].texcor[0] = vertices[i][2]; //
    		vertex_ptr[i].texcor[1] = vertices[i][3]; //
    	}
    	glUnmapBuffer(GL_ARRAY_BUFFER);
        //--------------------------------------------//


        // Render glyph texture over quad
        glBindTexture(GL_TEXTURE_2D, ch.TextureID);

        // Close
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        // Render quad
        glDrawArrays(GL_TRIANGLES, 0, 6);
        // Now advance cursors for next glyph (note that advance is number of 1/64 pixels)
        x += (ch.Advance >> 6) * scale; // Bitshift by 6 to get value in pixels (2^6 = 64 (divide amount of 1/64th pixels by 64 to get amount of pixels))
    }
    glBindTexture(GL_TEXTURE_2D, 0);
}
