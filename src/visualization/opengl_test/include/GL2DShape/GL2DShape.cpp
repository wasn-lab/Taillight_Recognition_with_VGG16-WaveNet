#include "GL2DShape.hpp"


GL2DShape::GL2DShape(glm::vec2 original_board_size_in):
    original_board_size(original_board_size_in),
    _shape(1.0f), _translation_m(1.0f),
    board_width(1.0), board_height(1.0), board_aspect_ratio(1.0),
    board_shape_mode(0),
    is_using_cv_pose(false)
{

}

// Shape
//------------------------------------------------------//
void GL2DShape::setBoardSize(float width_in, float height_in){
    board_shape_mode = 0;
    board_width = width_in;
    board_height = height_in;
    board_aspect_ratio = board_width/board_height;
    //
    updateBoardSize();
}
void GL2DShape::setBoardSize(float size_in, bool is_width){ // Using the aspect ratio from pixel data
    // board_aspect_ratio = float(im_pixel_width)/float(im_pixel_height);
    if (is_width){
        board_shape_mode = 1;
        board_width = size_in;
        board_height = board_width / board_aspect_ratio;
    }else{
        board_shape_mode = 2;
        board_height = size_in;
        board_width = board_height * board_aspect_ratio;
    }
    //
    updateBoardSize();
}
void GL2DShape::setBoardSizeRatio(float width_ratio_in, float height_ratio_in){ // Only use when is_perspected==false is_moveable==true
    board_shape_mode = 34; // 3 and 4
    board_size_ratio_w = width_ratio_in;
    board_size_ratio_h = height_ratio_in;
    board_width     = _viewport_size[0] * board_size_ratio_w;
    board_height    = _viewport_size[1] * board_size_ratio_h;
    //
    updateBoardSize();
}
void GL2DShape::setBoardSizeRatio(float ratio_in, bool is_width){ // Only use when is_perspected==false is_moveable==true
    // board_aspect_ratio = float(im_pixel_width)/float(im_pixel_height);

    if (is_width){
        board_shape_mode = 3;
        board_size_ratio_w = ratio_in;
        board_width = _viewport_size[0] * board_size_ratio_w;
        board_height = board_width / board_aspect_ratio;
    }else{
        board_shape_mode = 4;
        board_size_ratio_h = ratio_in;
        board_height = _viewport_size[1] * board_size_ratio_h;
        board_width = board_height * board_aspect_ratio;
    }
    //
    updateBoardSize();
}
void GL2DShape::setBoardSizePixel(int px_width_in, int px_heighth_in){
    board_shape_mode = 5;
    board_width = float(px_width_in);
    board_height = float(px_heighth_in);
    board_aspect_ratio = board_width/board_height;
    updateBoardSize();
}
void GL2DShape::setBoardSizePixel(int pixel_in, bool is_width){
    // board_aspect_ratio = float(im_pixel_width)/float(im_pixel_height);
    if (is_width){
        board_shape_mode = 6;
        board_width = pixel_in;
        board_height = board_width / board_aspect_ratio;
    }else{
        board_shape_mode = 7;
        board_height = pixel_in;
        board_width = board_height * board_aspect_ratio;
    }
    updateBoardSize();
}
void GL2DShape::updateBoardSize(){
    // board_aspect_ratio = float(im_pixel_width)/float(im_pixel_height);
    switch(board_shape_mode){
        case 0: // fixed size
            board_aspect_ratio = board_width/board_height;
            _shape = glm::scale(glm::mat4(1.0f), glm::vec3( board_width/(original_board_size.x), board_height/(original_board_size.y), 1.0f) );
            break;
        case 1: // fixed width
            board_height = board_width / board_aspect_ratio;
            _shape = glm::scale(glm::mat4(1.0f), glm::vec3( board_width/(original_board_size.x), board_height/(original_board_size.y), 1.0f) );
            break;
        case 2: // fixed height
            board_width = board_height * board_aspect_ratio;
            _shape = glm::scale(glm::mat4(1.0f), glm::vec3( board_width/(original_board_size.x), board_height/(original_board_size.y), 1.0f) );
            break;
        case 34: // fixed width and height ratio relative to viewport
            board_width = _viewport_size[0] * board_size_ratio_w;
            board_height = _viewport_size[1] * board_size_ratio_h;
            _shape = glm::scale(glm::mat4(1.0f), glm::vec3( (board_width/_viewport_size[0])*(2.0f/original_board_size.x), (board_height/_viewport_size[1])*(2.0f/original_board_size.y), 1.0f) );
            break;
        case 3:  // fixed width ratio relative to viewport
            board_width = _viewport_size[0] * board_size_ratio_w;
            board_height = board_width / board_aspect_ratio;
            _shape = glm::scale(glm::mat4(1.0f), glm::vec3( (board_width/_viewport_size[0])*(2.0f/original_board_size.x), (board_height/_viewport_size[1])*(2.0f/original_board_size.y), 1.0f) );
            break;
        case 4: // fixed height ratio relative to viewport
            board_height = _viewport_size[1] * board_size_ratio_h;
            board_width = board_height * board_aspect_ratio;
            _shape = glm::scale(glm::mat4(1.0f), glm::vec3( (board_width/_viewport_size[0])*(2.0f/original_board_size.x), (board_height/_viewport_size[1])*(2.0f/original_board_size.y), 1.0f) );
            break;
        case 5:
            board_aspect_ratio = board_width/board_height;
            _shape = glm::scale(glm::mat4(1.0f), glm::vec3( (board_width/_viewport_size[0])*(2.0f/original_board_size.x), (board_height/_viewport_size[1])*(2.0f/original_board_size.y), 1.0f) );
            break;
        case 6:  // fixed width pixel
            board_height = board_width / board_aspect_ratio;
            _shape = glm::scale(glm::mat4(1.0f), glm::vec3( (board_width/_viewport_size[0])*(2.0f/original_board_size.x), (board_height/_viewport_size[1])*(2.0f/original_board_size.y), 1.0f) );
            break;
        case 7: // fixed height pixel
            board_width = board_height * board_aspect_ratio;
            _shape = glm::scale(glm::mat4(1.0f), glm::vec3( (board_width/_viewport_size[0])*(2.0f/original_board_size.x), (board_height/_viewport_size[1])*(2.0f/original_board_size.y), 1.0f) );
            break;
        default:
            break;
    }
}
//------------------------------------------------------//


// Position
//------------------------------------------------------//
void GL2DShape::setBoardPositionCVPixel(
    int cv_x,
    int cv_y,
    int ref_point_mode_in,
    ALIGN_X     align_x_in,
    ALIGN_Y     align_y_in
){
    is_using_cv_pose = true;

    // ref_point_mode:
    // (the position of the origin of the viewport coordinate to describe the position of the shape)
    // 0: upper-left corner
    // 1: upper-right corner
    // 2: lower-left corner
    // 3: lower-right corner

    cv_pose = glm::ivec2(cv_x, cv_y);
    ref_point_mode = ref_point_mode_in;
    board_align_x = align_x_in;
    board_align_y = align_y_in;
    updateBoardPosition();
}
void GL2DShape::updateBoardPosition(){
    glm::ivec2 cv_ref(0,0);
    switch(ref_point_mode){
        case 0:
            break;
        case 1:
            cv_ref = glm::ivec2(_viewport_size[0], 0);
            break;
        case 2:
            cv_ref = glm::ivec2(0, _viewport_size[1]);
            break;
        case 3:
            cv_ref = glm::ivec2(_viewport_size[0], _viewport_size[1]);
            break;
        default:
            break;
    }
    //
    switch(board_align_x){
        case ALIGN_X::LEFT:
            cv_ref += glm::ivec2(board_width/2, 0);
            break;
        case ALIGN_X::CENTER:
            break;
        case ALIGN_X::RIGHT:
            cv_ref -= glm::ivec2(board_width/2, 0);
            break;
        default:
            break;
    }
    switch(board_align_y){
        case ALIGN_Y::TOP:
            cv_ref += glm::ivec2(0, board_height/2);
            break;
        case ALIGN_Y::CENTER:
            break;
        case ALIGN_Y::BUTTON:
            cv_ref -= glm::ivec2(0, board_height/2);
            break;
        default:
            break;
    }
    glm::vec2 gl_pose;
    gl_pose.x = float(cv_pose.x + cv_ref.x)/float(_viewport_size.x) * 2.0 - 1.0;
    gl_pose.y = float(cv_pose.y + cv_ref.y)/float(_viewport_size.y) * -2.0 + 1.0;
    // std::cout << "gl_pose = (" << gl_pose.x << ", " << gl_pose.y << ")\n";
    //
    _translation_m = translate(glm::mat4(1.0), glm::vec3(gl_pose, 0.0f));
}
//------------------------------------------------------//



// Update
//------------------------------------------------------//
void GL2DShape::updateBoardGeo(const glm::ivec2 &viewportsize_in, float aspect_ratio_in){
    _viewport_size = viewportsize_in;
    if (aspect_ratio_in >= 0.0f)
        board_aspect_ratio = aspect_ratio_in;
    updateBoardSize(); // Do this first
    if (is_using_cv_pose){
        updateBoardPosition();
    }
}
