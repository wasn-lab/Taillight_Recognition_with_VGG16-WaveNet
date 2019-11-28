#ifndef RM_LV2_TAG_BOUNDINGBOX_2D_H
#define RM_LV2_TAG_BOUNDINGBOX_2D_H

#include "rmBaseModel.h"

//
#include "rmBoundingBox2D.h"
#include "rmText3D_v2.h"


class rmlv2TagBoundingBox2D : public rmBaseModel
{
public:
    rmlv2TagBoundingBox2D(
        std::string _path_Assets_in,
        int _ROS_topic_id_in,
        bool is_perspected_in=true,
        bool is_moveable_in=true
    );
    //
	void Update(float dt);
    void Update(ROS_INTERFACE &ros_interface);
    void Update(ROS_API &ros_api);
	void Render(std::shared_ptr<ViewManager> &_camera_ptr);
    void Reshape(const glm::ivec2 & viewport_size_in);

    void setup_params(int im_width_in, int im_height_in, int image_offset_in_box_cv_x_in, int image_offset_in_box_cv_y_in){
        im_pixel_width = im_width_in;
        im_pixel_height = im_height_in;
        im_aspect = float(im_pixel_width) / float(im_pixel_height);
        image_offset_in_box_cv_x = image_offset_in_box_cv_x_in;
        image_offset_in_box_cv_y = image_offset_in_box_cv_y_in;
        // updateBoardSize();
        rm_box.setup_params(im_width_in, im_height_in, image_offset_in_box_cv_x_in, image_offset_in_box_cv_y_in);
        rm_text.setup_params(im_width_in, im_height_in);
    }



    GL2DShape shape;
    void updateBoardGeo(){
        shape.updateBoardGeo(_viewport_size, float(im_pixel_width)/float(im_pixel_height));
        rm_box.shape = shape;
        rm_text.shape = shape;
        rm_box.updateBoardGeo();
        rm_text.updateBoardGeo();
    }


protected:
    void Init();
    //
    int _ROS_topic_id;
    // std::shared_ptr< msgs::CamObj > msg_out_ptr;
    std::shared_ptr< msgs::DetectedObjectArray > msg_out_ptr;
    // ros::Time msg_time;

    // Settings
    bool is_perspected;
    bool is_moveable;

    //
    rmBoundingBox2D rm_box;
    rmText3D_v2 rm_text;

    void update_GL_data();

private:

    std::vector<rmText3D_v2::text2Din3D_data> text2Din3D_list;
    std::vector<rmText3D_v2::text2Dflat_data> text2Dflat_list;

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




};

#endif // RM_LV2_TAG_BOUNDINGBOX_2D_H
