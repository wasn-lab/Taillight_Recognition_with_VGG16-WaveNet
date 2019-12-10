#ifndef RM_TEXT_3D_v2_H
#define RM_TEXT_3D_v2_H

#include "rmBaseModel.h"
#include "GL2DShape.hpp" // GL2DShape

#include <queue>          // std::queue
#include <sstream>        // for to_string_p

// #include <map> // std::map
// FreeType
#include <ft2build.h>
#include FT_FREETYPE_H

/*
//------------------------------//
About 2D texts
Note 1:
    This module can also perform as 2D text;
    however, the current implemntation let all the 2D text share
    the same one 2D plane (the "board"), ie. it's global property.
    Warnning: you can only draw "one" kind of 2D text per Text3D_v2 object.
Note 2:
    Based on the current implemtation, the 2D text is not efficient as it could be.
//------------------------------//
*/




// atlas
struct atlas;
//


class rmText3D_v2 : public rmBaseModel
{
public:
    // // Different alignment
    // //--------------------------------------//
    // enum class ALIGN_X{
    //     LEFT,
    //     CENTER,
    //     RIGHT
    // };
    // enum class ALIGN_Y{
    //     TOP,
    //     CENTER,
    //     BUTTON
    // };
    // //--------------------------------------//

    /*
    Requirements:
    2D draw_mode:
        0 - proj_matrix, m_shape.model, set_pose_modle_ref_by_world()
        1 - m_shape.model <-- 2D
        2 - m_shape.model <-- 2D, pose2D_depth=1.0
        3 - pose2D_depth=1.0
    */


    // Different drawing method
    //--------------------------------------//
    // text2D in 3D space
    /*
    Note: position_2D can be the following
    pos_mode, size_mode:
        0 - OpenCV pixel x:[0,ncol] "right", y:[0,nrow] "down"
        1 - OpenGL normalizd coordinate x:[-1,1] "right", y:[-1,1] "up"
    ---
    Requirements:
    pos_mode, size_mode:
        0 - _viewport_size, board_width, board_height
        1 - m_shape.shape, board_aspect_ratio
    draw_mode (is_fullviewport, is_background):
        false, false - m_shape.model, pose2D_depth=0.0
        false, true - m_shape.model, pose2D_depth=1.0
        true, false - pose2D_depth=0.0
        true, true - pose2D_depth=1.0
    */
    struct text2Dflat_data{
        std::string text;
        glm::vec2   position_2D; // (x,y), the unit depends on pos_mode
        float       size; // The "hight of line", the unit depends on pos_mode
        glm::vec3   color;
        ALIGN_X     align_x;
        ALIGN_Y     align_y;
        int         pos_mode;
        int         size_mode;
        bool        is_fullviewport;
        bool        is_background;
        text2Dflat_data(
            const std::string &text_in,
            const glm::vec2 &position_2D_in=glm::vec2(0.0f),
            float size_in=1.0f,
            const glm::vec3 &color_in=glm::vec3(1.0f),
            ALIGN_X align_x_in=ALIGN_X::LEFT,
            ALIGN_Y align_y_in=ALIGN_Y::TOP,
            int pos_mode_in=1,
            int size_mode_in=1,
            bool is_fullviewport_in=false,
            bool is_background_in=false
        ):
            text(text_in),
            position_2D(position_2D_in),
            size(size_in),
            color(color_in),
            align_x(align_x_in),
            align_y(align_y_in),
            pos_mode(pos_mode_in),
            size_mode(size_mode_in),
            is_fullviewport(is_fullviewport_in),
            is_background(is_background_in)
        {
        }
    };
    // text2D in 3D space
    /*
    Note: position_2D can be the following
    pos_mode:
        0 - meter in 3D perspective view (3D text2D only)
        1 - OpenGL normalizd coordinate x:[-1,1] "right", y:[-1,1] "up"
    ---
    Requirements:
    pos_mode:
        0 - (none)
        1 - m_shape.shape, board_aspect_ratio
    */
    struct text2Din3D_data{
        std::string text;
        glm::vec2   position_2D; // (x,y), the unit depends on pos_mode
        float       size; // The "hight of line", the unit depends on pos_mode
        glm::vec3   color;
        ALIGN_X     align_x;
        ALIGN_Y     align_y;
        int         pos_mode;
        text2Din3D_data(
            const std::string &text_in,
            const glm::vec2 &position_2D_in=glm::vec2(0.0f),
            float size_in=1.0f,
            const glm::vec3 &color_in=glm::vec3(1.0f),
            ALIGN_X align_x_in=ALIGN_X::LEFT,
            ALIGN_Y align_y_in=ALIGN_Y::TOP,
            int pos_mode_in=0
        ):
            text(text_in),
            position_2D(position_2D_in),
            size(size_in),
            color(color_in),
            align_x(align_x_in),
            align_y(align_y_in),
            pos_mode(pos_mode_in)
        {
        }
    };
    // text3D as an object in space
    struct text3D_data{// Different drawing method
        std::string text;
        glm::mat4   pose_ref_point;
        glm::vec2   offset_ref_point_2D;
        float       size; // The "hight of line" in meter
        glm::vec3   color;
        ALIGN_X     align_x;
        ALIGN_Y     align_y;
        text3D_data(
            const std::string &text_in,
            const glm::mat4 &pose_ref_point_in=glm::mat4(1.0f),
            const glm::vec2 &offset_ref_point_2D_in=glm::vec2(0.0f),
            float size_in=1.0f,
            const glm::vec3 &color_in=glm::vec3(1.0f),
            ALIGN_X align_x_in=ALIGN_X::LEFT,
            ALIGN_Y align_y_in=ALIGN_Y::TOP
        ):
            text(text_in),
            pose_ref_point(pose_ref_point_in),
            offset_ref_point_2D(offset_ref_point_2D_in),
            size(size_in),
            color(color_in),
            align_x(align_x_in),
            align_y(align_y_in)
        {}
    };
    // text3D as a billboard attached to a 3D point
    struct text_billboard_data{
        std::string text;
        glm::vec3   position_ref_point; // (x,y,z) in meter
        glm::vec2   offset_ref_point_2D;
        float       size; // The "hight of line" in meter
        glm::vec3   color;
        ALIGN_X     align_x;
        ALIGN_Y     align_y;
        text_billboard_data(
            const std::string &text_in,
            const glm::vec3 &position_ref_point_in=glm::vec3(0.0f),
            const glm::vec2 &offset_ref_point_2D_in=glm::vec2(0.0f),
            float size_in=1.0f,
            const glm::vec3 &color_in=glm::vec3(1.0f),
            ALIGN_X align_x_in=ALIGN_X::LEFT,
            ALIGN_Y align_y_in=ALIGN_Y::TOP
        ):
            text(text_in),
            position_ref_point(position_ref_point_in),
            offset_ref_point_2D(offset_ref_point_2D_in),
            size(size_in),
            color(color_in),
            align_x(align_x_in),
            align_y(align_y_in)
        {}
    };
    // text3D as a billboard attached to a 3D point with fixed size
    struct text_freeze_board_data{
        std::string text;
        glm::vec3   position_ref_point; // (x,y,z) in meter
        glm::vec2   offset_ref_point_2D;
        float       size; // The "hight of line" in "pixel"
        glm::vec3   color;
        ALIGN_X     align_x;
        ALIGN_Y     align_y;
        text_freeze_board_data(
            const std::string &text_in,
            const glm::vec3 &position_ref_point_in=glm::vec3(0.0f),
            const glm::vec2 &offset_ref_point_2D_in=glm::vec2(0.0f),
            float size_in=48,
            const glm::vec3 &color_in=glm::vec3(1.0f),
            ALIGN_X align_x_in=ALIGN_X::LEFT,
            ALIGN_Y align_y_in=ALIGN_Y::TOP
        ):
            text(text_in),
            position_ref_point(position_ref_point_in),
            offset_ref_point_2D(offset_ref_point_2D_in),
            size(size_in),
            color(color_in),
            align_x(align_x_in),
            align_y(align_y_in)
        {}
    };
    //--------------------------------------//
    // end Different drawing method







    rmText3D_v2(std::string _path_Assets_in, std::string frame_id_in="");
    rmText3D_v2(std::string _path_Assets_in, int _ROS_topic_id_in);
    //
	void Update(float dt);
    void Update(ROS_INTERFACE &ros_interface);
    void Update(ROS_API &ros_api);
	void Render(std::shared_ptr<ViewManager> &_camera_ptr);
    void Reshape(const glm::ivec2 & viewport_size_in);
    //
    inline glm::mat4 * get_model_m_ptr(){ return &(m_shape.model); }

    void setup_params(int im_width_in, int im_height_in){
        im_pixel_width = im_width_in;
        im_pixel_height = im_height_in;
        im_aspect = float(im_pixel_width) / float(im_pixel_height);
        updateBoardGeo();
    }

    // Insert method for texts
    // queues - draw once
    //-------------------------------------//
    inline void insert_text(const text2Dflat_data & data_in ){
        text2Dflat_queue.push( data_in );
    }
    inline void insert_text(const text2Din3D_data & data_in ){
        text2Din3D_queue.push( data_in );
    }
    inline void insert_text(const text3D_data & data_in ){
        text3D_queue.push( data_in );
    }
    inline void insert_text(const text_billboard_data & data_in ){
        text_billboard_queue.push( data_in );
    }
    inline void insert_text(const text_freeze_board_data & data_in ){
        text_freeze_board_queue.push( data_in );
    }
    //-------------------------------------//

    // buffers - draw each time until update
    //-------------------------------------//
    inline void insert_text(const std::vector<text2Dflat_data> & data_list_in ){
        text2Dflat_buffer = data_list_in;
    }
    inline void insert_text(const std::vector<text2Din3D_data> & data_list_in ){
        text2Din3D_buffer = data_list_in;
    }
    inline void insert_text(const std::vector<text3D_data> & data_list_in ){
        text3D_buffer = data_list_in;
    }
    inline void insert_text(const std::vector<text_billboard_data> & data_list_in ){
        text_billboard_buffer = data_list_in;
    }
    inline void insert_text(const std::vector<text_freeze_board_data> & data_list_in ){
        text_freeze_board_buffer = data_list_in;
    }
    //-------------------------------------//


    // Shape
    //-------------------------------------------//
    // For usage, please refer to the GL2DShape
    GL2DShape shape;
    void updateBoardGeo(){
        shape.updateBoardGeo(_viewport_size, float(im_pixel_width)/float(im_pixel_height));
        shape.get_shape(m_shape.shape);
        if ( shape.get_tranlate(translateMatrix) ){
            update_pose_model_by_model_ref();
        }
    }
    //-------------------------------------------//


protected:
    void Init();
    virtual void LoadModel();
    //
    int _ROS_topic_id;
    // std::shared_ptr< msgs::LidRoi > box3d_out_ptr;
    // ros::Time msg_time;
    std::string _frame_id;

    // void RenderText(const std::string &text, atlas *_atlas_ptr, float x, float y, float scale_x_in, float scale_y_in, glm::vec3 color);
    void RenderText(
        const std::string &text,
        std::shared_ptr<atlas> &_atlas_ptr,
        float x_in,
        float y_in,
        float scale_x_in,
        float scale_y_in,
        glm::vec3 color,
        ALIGN_X align_x=ALIGN_X::LEFT,
        ALIGN_Y align_y=ALIGN_Y::TOP
    );

    // Different draw methods
    //--------------------------------------------------------//
    void _draw_one_text2Dflat(std::shared_ptr<ViewManager> &_camera_ptr, text2Dflat_data &_data_in);
    void _draw_one_text2Din3D(std::shared_ptr<ViewManager> &_camera_ptr, text2Din3D_data &_data_in);
    void _draw_one_text3D(std::shared_ptr<ViewManager> &_camera_ptr, text3D_data &_data_in);
    void _draw_one_text_billboard(std::shared_ptr<ViewManager> &_camera_ptr, text_billboard_data &_data_in);
    void _draw_one_text_freeze_board(std::shared_ptr<ViewManager> &_camera_ptr, text_freeze_board_data &_data_in);
    //--------------------------------------------------------//



    // The image params
    // Note: The origin of the image is at its center.
    int im_pixel_width;
    int im_pixel_height;
    float im_aspect; // w / h
    //



private:
    // model info
    struct Shape{
        GLuint vao;
        GLuint vbo;
        GLuint ebo;
        GLuint m_texture;
        //
        int indexCount;
        //
        glm::mat4 shape;
        glm::mat4 model;
    };
    Shape m_shape;


    struct point {
    	GLfloat x;
    	GLfloat y;
    	GLfloat s;
    	GLfloat t;
    };
    // std::vector<point> vector_list;

    //uniform id
	struct
	{
		GLint  mv_matrix;
		GLint  proj_matrix;
        GLint  textColor;
        GLint  ref_point;
        GLint  pose2D_depth; // default/foreground: 0.0, background: 1.0, floating: greater than 0.0
	} uniforms;

    glm::vec2 ref_point;

    //
    int _num_vertex_idx_per_box;
    long long _max_num_vertex_idx;
    int _num_vertex_per_box;
    long long _max_num_vertex;
    long long _max_num_box;
    // long long _max_string_length;


    //
    //------------------------------------------//
    // queues - only draw on time
    std::queue<text2Dflat_data> text2Dflat_queue;
    std::queue<text2Din3D_data> text2Din3D_queue;
    std::queue<text3D_data> text3D_queue;
    std::queue<text_billboard_data> text_billboard_queue;
    std::queue<text_freeze_board_data> text_freeze_board_queue;
    // buffers - draw each time
    std::vector<text2Dflat_data> text2Dflat_buffer;
    std::vector<text2Din3D_data> text2Din3D_buffer;
    std::vector<text3D_data> text3D_buffer;
    std::vector<text_billboard_data> text_billboard_buffer;
    std::vector<text_freeze_board_data> text_freeze_board_buffer;
    //------------------------------------------//



    // Pointers of atlas
    /*
    atlas *a48_ptr;
    atlas *a24_ptr;
    atlas *a12_ptr;
    */
    /*
    std::shared_ptr<atlas> a48_ptr;
    std::shared_ptr<atlas> a24_ptr;
    std::shared_ptr<atlas> a12_ptr;
    */



};

#endif // RM_TEXT_3D_v2_H
