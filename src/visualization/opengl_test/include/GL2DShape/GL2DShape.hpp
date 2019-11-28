#ifndef GL_2D_SHAPE_H
#define GL_2D_SHAPE_H

#include "Common.h"

// Different alignment
//--------------------------------------//
enum class ALIGN_X{
    LEFT,
    CENTER,
    RIGHT
};
enum class ALIGN_Y{
    TOP,
    CENTER,
    BUTTON
};
//--------------------------------------//


// Size mode:
// 0 - fixed size
// 1 - fixed width
// 2 - fixed height
// 3 - (2D) fixed width ratio relative to viewport
// 4 - (2D) fixed height ratio ralative to viewport
// 5 - (2D) fixed pixel size
// 6 - (2D) fixed pixel width
// 7 - (2D) fixed pixel height

// ref_point_mode:
// (the position of the origin of the viewport coordinate to describe the position of the shape)
// 0: upper-left corner
// 1: upper-right corner
// 2: lower-left corner
// 3: lower-right corner

class GL2DShape{
public:


    GL2DShape(glm::vec2 original_board_size_in=glm::vec2(2.0f, 2.0f) );

    // Set board size
    void setBoardSize(float width_in, float height_in); // 3D space
    void setBoardSize(float size_in, bool is_width); // 3D space / Using the aspect ratio from pixel data
    void setBoardSizeRatio(float width_ratio_in, float height_ratio_in); // Only use when is_perspected==false is_moveable==true
    void setBoardSizeRatio(float ratio_in, bool is_width); // Only use when is_perspected==false is_moveable==true
    void setBoardSizePixel(int px_width_in, int px_heighth_in);
    void setBoardSizePixel(int pixel_in, bool is_width);
    // Set 2D image (moveable) position
    void setBoardPositionCVPixel(
        int cv_x,
        int cv_y,
        int ref_point_mode_in=0,
        ALIGN_X     align_x_in=ALIGN_X::CENTER,
        ALIGN_Y     align_y_in=ALIGN_Y::CENTER
    );
    // ref_point_mode:
    // (the position of the origin of the viewport coordinate to describe the position of the shape)
    // 0: upper-left corner
    // 1: upper-right corner
    // 2: lower-left corner
    // 3: lower-right corner

    // Update method
    void updateBoardGeo(const glm::ivec2 &viewportsize_in, float aspect_ratio_in=-1.0f);

    // Getting methods
    inline bool get_shape(glm::mat4 &shape_out){ shape_out = _shape;    return true; }
    inline bool get_tranlate(glm::mat4 & translation_m_out){
        if (is_using_cv_pose){
            translation_m_out = _translation_m;
            return true;
        }
        return false;
    }


    // Note: The origin of the image is at its center.
    // int im_pixel_width;
    // int im_pixel_height;
    // float im_aspect; // w / h
    // Params
    float board_width; // meter or pixel
    float board_height; // meter or pixel
    float board_aspect_ratio; // w/h
    float board_size_ratio_w; // Only for mode 3 or 34
    float board_size_ratio_h; // Only for mode 4 or 34
    int board_shape_mode;
    glm::ivec2 _viewport_size; // (w,h)
    // Size mode:
    // 0 - fixed size
    // 1 - fixed width
    // 2 - fixed height
    // 3 - (2D) fixed width ratio relative to viewport
    // 4 - (2D) fixed height ratio ralative to viewport
    // 5 - (2D) fixed pixel size
    // 6 - (2D) fixed pixel width
    // 7 - (2D) fixed pixel height

    // Board position
    bool is_using_cv_pose;
    glm::ivec2 cv_pose;
    int ref_point_mode;
    ALIGN_X board_align_x;
    ALIGN_Y board_align_y;
    // ref_point_mode:
    // (the position of the origin of the viewport coordinate to describe the position of the shape)
    // 0: upper-left corner
    // 1: upper-right corner
    // 2: lower-left corner
    // 3: lower-right corner





    void setup_params(int im_width_in, int im_height_in, int image_offset_in_box_cv_x_in, int image_offset_in_box_cv_y_in){
        im_pixel_width = im_width_in;
        im_pixel_height = im_height_in;
        im_aspect = float(im_pixel_width) / float(im_pixel_height);
        image_offset_in_box_cv_x = image_offset_in_box_cv_x_in;
        image_offset_in_box_cv_y = image_offset_in_box_cv_y_in;
        updateBoardGeo(_viewport_size, im_aspect);
    }

    // OpenCV --> OpenGL
    //-------------------------------------------------------//
    // The box param in openCV style
    struct box_param_cv{
        glm::vec2 xy;
        int width;
        int height;
        int obj_class;

        box_param_cv(int x, int y, int w, int h, int obj_class_in):
            xy(x, y), width(w), height(h), obj_class(obj_class_in)
        {}
    };
    // The box param in openGL style: 4 points in normalized coordinate: x,y belongs to [-1, 1]
    struct box_param_gl{
        glm::vec2 xy_list[4];
        int obj_class;
    };
    // The image params
    // Note: The origin of the image is at its center.
    int im_pixel_width;
    int im_pixel_height;
    float im_aspect; // w / h
    // The box coordinate relative to image, normally (0,0)
    int image_offset_in_box_cv_x;
    int image_offset_in_box_cv_y;
    //
    inline void toNormGL(int cv_x, int cv_y, float &gl_x, float &gl_y){
        // Convert CV coordinate to normalized GL coordinate x:[-1,1], y:[-1,1]
        gl_x = (cv_x - image_offset_in_box_cv_x)/float(im_pixel_width) * 2.0 - 1.0;
        gl_y = (cv_y - image_offset_in_box_cv_y)/float(im_pixel_height) * -2.0 + 1.0;
    }
    void convert_cv_to_normalized_gl(const box_param_cv &box_cv_in, box_param_gl & box_gl_out){
        box_gl_out.obj_class = box_cv_in.obj_class;
        float gl_x, gl_y;
        // float gl_w, gl_h;
        // gl_w = box_cv_in.width/float(im_pixel_width);
        // gl_h = box_cv_in.height/float(im_pixel_height);
        int _w = box_cv_in.width;
        int _h = box_cv_in.height;
        int _i = 0;
        // P1
        toNormGL(box_cv_in.xy[0], box_cv_in.xy[1], gl_x, gl_y);
        box_gl_out.xy_list[_i++] = glm::vec2(gl_x, gl_y);
        // P2
        toNormGL(box_cv_in.xy[0]+_w, box_cv_in.xy[1], gl_x, gl_y);
        box_gl_out.xy_list[_i++] = glm::vec2(gl_x, gl_y);
        // P3
        toNormGL(box_cv_in.xy[0]+_w, box_cv_in.xy[1]+_h, gl_x, gl_y);
        box_gl_out.xy_list[_i++] = glm::vec2(gl_x, gl_y);
        // P4
        toNormGL(box_cv_in.xy[0], box_cv_in.xy[1]+_h, gl_x, gl_y);
        box_gl_out.xy_list[_i++] = glm::vec2(gl_x, gl_y);
    }
    bool is_gl_box_valid(const box_param_gl & box_gl){
        // At least one point valid is valid
        bool is_valid = false;
        for (size_t i=0; i < 4; ++i){
            if (box_gl.xy_list[i][0] >= -1 && box_gl.xy_list[i][0] <= 1){
                if (box_gl.xy_list[i][1] >= -1 && box_gl.xy_list[i][1] <= 1){
                    is_valid |= true;
                }
            }
        }
        return is_valid;
    }
    //-------------------------------------------------------//
    // end OpenCV --> OpenGL

private:

    glm::mat4 _shape;
    glm::mat4 _translation_m;

    void updateBoardSize();
    void updateBoardPosition();

    //
    glm::vec2 original_board_size;



};

#endif // GL_2D_SHAPE_H
