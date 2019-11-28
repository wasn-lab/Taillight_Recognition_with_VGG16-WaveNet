#include "rmText2D.h"


void *font2D = GLUT_BITMAP_TIMES_ROMAN_24;
void *fonts2D[] =
{
    GLUT_BITMAP_8_BY_13,
    GLUT_BITMAP_9_BY_15,
    GLUT_BITMAP_TIMES_ROMAN_10,
    GLUT_BITMAP_TIMES_ROMAN_24,
    GLUT_BITMAP_HELVETICA_10,
    GLUT_BITMAP_HELVETICA_12,
    GLUT_BITMAP_HELVETICA_18
};


rmText2D::rmText2D()
{
    // init_paths(_path_Assets_in);
	Init();
}
void rmText2D::Init(){
    //

    //Load model to shader _program_ptr
	LoadModel();

}
void rmText2D::LoadModel(){

}
void rmText2D::Update(float dt){
    // Update the data (buffer variables) here
}
void rmText2D::Update(ROS_INTERFACE &ros_interface){
    // Update the data (buffer variables) here
}
void rmText2D::Update(ROS_API &ros_api){
    // Update the data (buffer variables) here

    // test
    static int _count = 0;

    insert_text(
        text2D_data(
            "This is 2D text: " + std::to_string(_count),
            glm::vec2(0.5, 0.8),
            0.2,
            glm::vec3(1.0f, 0.5f, 0.0f)
        )
    );
    _count++;
}
void rmText2D::Render(std::shared_ptr<ViewManager> &_camera_ptr){
    // test
    // static int _count = 0;
    //
    glUseProgram(0); // Program 0: OpenGL ver1.0
    // selectFont2D(2);
    // text2D_output(0.5,0.8, "This is 2D text: " + std::to_string(_count++) );

    for (size_t i=0; i < text2D_buffer.size(); ++i){
        _draw_one_text2D(_camera_ptr, text2D_buffer.front() );
        text2D_buffer.pop();
    }
}


// Different draw methods
//--------------------------------------------------------//
void rmText2D::_draw_one_text2D(std::shared_ptr<ViewManager> &_camera_ptr, text2D_data &_data_in){
    selectFont2D(3);

    /*
    int _line_count = 1;
    int _max_word_per_line = 0;
    int _word_per_line = 0;
    // Counting lines
    for (size_t i = 0; i < _data_in.text.size(); i++) {
      if (_data_in.text[i] == '\n'){
          _line_count++;
          _word_per_line = 0;
      }else{
          _word_per_line++;
      }
      if (_word_per_line > _max_word_per_line){
          _max_word_per_line = _word_per_line;
      }
    }
    //
    */

    glColor3f(_data_in.color.x, _data_in.color.y, _data_in.color.z);

    // glScalef(0.1,0.1,1.0);

    text2D_output(_data_in.position_2D.x, _data_in.position_2D.y, _data_in.text);

    glColor3f(1,1,1);
}
//--------------------------------------------------------//

//
void rmText2D::selectFont2D(int newfont){
  font2D = fonts2D[newfont];
}
void rmText2D::text2D_output(float x, float y, std::string string_in){
  glRasterPos2f(x, y);
  for (size_t i = 0; i < string_in.size(); i++) {
    glutBitmapCharacter(font2D, string_in[i]);
  }
}
