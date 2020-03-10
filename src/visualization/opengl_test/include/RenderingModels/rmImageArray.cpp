#include "rmImageArray.h"

// Maximum texture width
#define MAXWIDTH 4096




// Atlas
//--------------------------------------//
struct ATLAS_IMAGE {
	GLuint TextureID;		// texture object
    int font_size;
    std::vector<std::string> image_path_list;

	int _tex_w;			// width of texture in pixels
	int _tex_h;			// height of texture in pixels

    struct CHARACTER_PARAM{
		int ax;	// advance.x
		int ay;	// advance.y

		int bw;	// bitmap.width;
		int bh;	// bitmap.height;

		int bl;	// bitmap_left;
		int bt;	// bitmap_top;

		float tx;	// x offset of glyph in texture coordinates
		float ty;	// y offset of glyph in texture coordinates
        //
        CHARACTER_PARAM():
            ax(0),ay(0),bw(0),bh(0),bl(0),bt(0),tx(0),ty(0)
        {}
	};
    // character information
    vector<CHARACTER_PARAM> _ch;
    //

    ATLAS_IMAGE(std::vector<std::string> &image_path_list_in){
        image_path_list = image_path_list_in;
        Init();
    }

    // Init, load textures
    //--------------------------------//
    void Init() {

        //
        // FT_GlyphSlot _glyph = face->glyph;

        int roww = 0;
        int rowh = 0;
        _tex_w = 0;
        _tex_h = 0;

        // memset(_ch, 0, sizeof _ch);
        _ch.resize( image_path_list.size() + IMAGE_CH_START_CHAR );


        int im_pixel_width = 1;
        int im_pixel_height = 1;
        int rowh_max = 0;
        /* Find minimum size for a texture holding all visible ASCII characters */
        for (int i = 0; i < image_path_list.size(); i++) {

            // Scoped region: load the texture, 1st time
            {
                TextureData tdata = Common::Load_png(image_path_list[i].c_str());
                im_pixel_width = tdata.width;
                im_pixel_height = tdata.height;
            }
            //

            // Change to next row
        	if (roww + im_pixel_width + 1 >= MAXWIDTH) {
        		_tex_w = std::max(_tex_w, roww);
        		_tex_h += rowh + 1;
        		roww = 0;
        		rowh = 0;
        	}

            // Advance
        	roww += im_pixel_width + 1;
        	rowh = std::max(rowh, im_pixel_height);

            // Find the maximum height of texture images
            rowh_max = std::max(rowh_max, rowh);
        }

        _tex_w = std::max(_tex_w, roww);
        _tex_h += rowh + 1;
        //-------------------------------------//

        //
        font_size = rowh_max;
        //

        /* Create a texture that will be used to hold all ASCII glyphs */
        glActiveTexture(GL_TEXTURE0);
        glGenTextures(1, &TextureID);
        glBindTexture(GL_TEXTURE_2D, TextureID);

        //
        // glTexImage2D(GL_TEXTURE_2D, 0, GL_ALPHA, _tex_w, _tex_h, 0, GL_ALPHA, GL_UNSIGNED_BYTE, 0);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, _tex_w, _tex_h, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);
        //

        /* We require 1 byte alignment when uploading texture data */
        // glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

        //
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        //


        /* Paste all glyph bitmaps into the texture, remembering the offset */
        int _offset_x = 0;
        int _offset_y = 0;
        //
        rowh = 0;
        im_pixel_width = 1;
        im_pixel_height = 1;
        for (int i = 0; i < image_path_list.size(); i++) {

            // Scoped region: load the texture, 1st time
            {
                TextureData tdata = Common::Load_png(image_path_list[i].c_str());
                im_pixel_width = tdata.width;
                im_pixel_height = tdata.height;
                //
                if (_offset_x + im_pixel_width + 1 >= MAXWIDTH) {
            		_offset_y += rowh + 1;
            		rowh = 0;
            		_offset_x = 0;
            	}
                // Upload the texture data
            	glTexSubImage2D(GL_TEXTURE_2D, 0, _offset_x, _offset_y, im_pixel_width, im_pixel_height, GL_RGBA, GL_UNSIGNED_BYTE, tdata.data);
            }
            //

            int ch_idx = i + IMAGE_CH_START_CHAR;
        	_ch[ch_idx].ax = im_pixel_width; // devide by 64 to translate to pixel // width
        	_ch[ch_idx].ay = 0; // devide by 64 to translate to pixel // 0

        	_ch[ch_idx].bw = im_pixel_width; // width
        	_ch[ch_idx].bh = im_pixel_height;  // height

        	_ch[ch_idx].bl = 0; // _glyph->bitmap_left;  // 0
        	_ch[ch_idx].bt = 0; // _glyph->bitmap_top;   // 0

        	_ch[ch_idx].tx = _offset_x / double(_tex_w);
        	_ch[ch_idx].ty = _offset_y / double(_tex_h);

        	rowh = std::max(rowh, im_pixel_height);
        	_offset_x += im_pixel_width + 1;
        }
        // Add " " (space) character
        //----------------------------------//
        // (None)
        //----------------------------------//
        fprintf(stderr, "Generated a %d x %d (%d kb) texture atlas\n", _tex_w, _tex_h, _tex_w * _tex_h / 1024);
    }
    //--------------------------------//
    // end Init, load textures

	~ATLAS_IMAGE() {
		glDeleteTextures(1, &TextureID);
	}
};
//--------------------------------------//
// end Atlas






// atlas_ptr
// std::shared_ptr<ATLAS_IMAGE> atlas_ptr;
//



// Box vertex index
namespace rmLidarBoundingBox_ns{
    const GLuint box_idx_data[] = {
        0,1,2,
        1,2,3
    };
}



rmImageArray::rmImageArray(std::string _path_Assets_in, std::string frame_id_in):
    _frame_id(frame_id_in)
{
    _path_Shaders_sub_dir += "ImageArray/";
    init_paths(_path_Assets_in);
    _num_vertex_idx_per_box = (3*2);
    _num_vertex_per_box = 4; // 8;
    _max_num_box = 500; // 100000;
    _max_num_vertex_idx = _max_num_box*(long long)(_num_vertex_idx_per_box);
    _max_num_vertex = _max_num_box*(long long)(_num_vertex_per_box);
    //
	Init();
}
rmImageArray::rmImageArray(std::string _path_Assets_in, int _ROS_topic_id_in):
    _ROS_topic_id(_ROS_topic_id_in)
{
    _path_Shaders_sub_dir += "ImageArray/";
    init_paths(_path_Assets_in);
    _num_vertex_idx_per_box = (3*2);
    _num_vertex_per_box = 4; // 8;
    _max_num_box = 500; // 100000;
    _max_num_vertex_idx = _max_num_box*(long long)(_num_vertex_idx_per_box);
    _max_num_vertex = _max_num_box*(long long)(_num_vertex_per_box);
    //
	Init();
}
void rmImageArray::Init(){
    //
	_program_ptr.reset( new ShaderProgram() );
    // Load shaders
    _program_ptr->AttachShader(get_full_Shader_path("ImageArray.vs.glsl"), GL_VERTEX_SHADER);
    _program_ptr->AttachShader(get_full_Shader_path("ImageArray.fs.glsl"), GL_FRAGMENT_SHADER);
    // Link _program_ptr
	_program_ptr->LinkProgram();
    //

    // Cache uniform variable id
	uniforms.proj_matrix = glGetUniformLocation(_program_ptr->GetID(), "proj_matrix");
	uniforms.mv_matrix = glGetUniformLocation(_program_ptr->GetID(), "mv_matrix");
    uniforms.textColor = glGetUniformLocation(_program_ptr->GetID(), "textColor");
    uniforms.ref_point = glGetUniformLocation(_program_ptr->GetID(), "ref_point");
    uniforms.pose2D_depth = glGetUniformLocation(_program_ptr->GetID(), "pose2D_depth");

    // Init model matrices
    m_shape.shape = glm::mat4(1.0);
	m_shape.model = glm::mat4(1.0);
    attach_pose_model_by_model_ref_ptr(m_shape.model); // For adjusting the model pose by public methods
    // The reference point of a text
    ref_point = glm::vec2(0.0f);

    //Load model to shader _program_ptr
	LoadModel();

}
void rmImageArray::LoadModel(){

    /*
    //-------------------------------//
    // std::vector<std::string> image_name_list;
    std::vector<std::string> image_path_list;
    // Enter the image file name
    image_name_list.push_back("TFL_red_off.png");
    image_name_list.push_back("TFL_red_on.png");
    image_name_list.push_back("TFL_yello_off.png");
    image_name_list.push_back("TFL_yello_on.png");
    image_name_list.push_back("TFL_green_off.png");
    image_name_list.push_back("TFL_green_on.png");
    // Generate full paths
    for (size_t i=0; i < image_name_list.size(); ++i){
        image_path_list.push_back( get_full_Assets_path(image_name_list[i]) );
    }
    //-------------------------------//


    if (!atlas_ptr){
        atlas_ptr.reset(new ATLAS_IMAGE(image_path_list));
    }else{
        std::cout << "The atlas<" << atlas_ptr->font_size << "> is already created.\n";
    }
    */



    // GL things
    //----------------------------------------//
    glGenVertexArrays(1, &m_shape.vao);
	glBindVertexArray(m_shape.vao);

    glGenBuffers(1, &m_shape.vbo);
	glBindBuffer(GL_ARRAY_BUFFER, m_shape.vbo);
    // glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * 6 * 4, NULL, GL_DYNAMIC_DRAW);
    glBufferData(GL_ARRAY_BUFFER, _max_num_vertex * sizeof(point), NULL, GL_DYNAMIC_DRAW);

    // glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(GLfloat), NULL);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, sizeof(point), NULL);
    glEnableVertexAttribArray(0);


    // ebo
    glGenBuffers(1, &m_shape.ebo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_shape.ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, _max_num_vertex_idx * sizeof(GLuint), NULL, GL_STATIC_DRAW);
    // Directly assign data to memory of GPU
    //--------------------------------------------//
	GLuint * _idx_ptr = (GLuint *)glMapBufferRange(GL_ELEMENT_ARRAY_BUFFER, 0, _max_num_vertex_idx * sizeof(GLuint), GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT);
	size_t _idx_idx = 0;
    long long _offset_box = 0;
	for (size_t i = 0; i < _max_num_vertex_idx; i++)
	{
        _idx_ptr[i] = rmLidarBoundingBox_ns::box_idx_data[_idx_idx] + _offset_box;
        // std::cout << "_idx_ptr[" << i << "] = " << _idx_ptr[i] << "\n";
        _idx_idx++;
        if (_idx_idx >= _num_vertex_idx_per_box){
            _idx_idx = 0;
            _offset_box += _num_vertex_per_box;
        }
	}
	glUnmapBuffer(GL_ELEMENT_ARRAY_BUFFER);
    // m_shape.indexCount = _max_num_vertex_idx; //  1 * _num_vertex_idx_per_box; // ;
    //--------------------------------------------//



    // Close
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
    //----------------------------------------//




    /*
    // test
    vector<text_billboard_data> data_test;
    for (size_t _k=0; _k < 1000; ++_k){
        data_test.push_back(
            text_billboard_data(
                "billboard #" + std::to_string(_k),
                glm::vec3( float(_k)*2.0f, 0.0f, 2.0f),
                glm::vec2(0.0f, 0.0f),
                1.5f,
                glm::vec3(0.6f)
            )
        );
        insert_text(data_test);
    }
    //
    */


}
void rmImageArray::Update(float dt){
    // Update the data (buffer variables) here
}
void rmImageArray::Update(ROS_INTERFACE &ros_interface){
    // Update the data (buffer variables) here
}
void rmImageArray::Update(ROS_API &ros_api){
    // Update the data (buffer variables) here

    // Update transform
    //--------------------------------//
    if (_frame_id.size() > 0){
        // Get tf
        bool tf_successed = false;
        glm::mat4 _model_tf = ROStf2GLMmatrix(ros_api.get_tf(_frame_id, tf_successed));
        set_pose_modle_ref_by_world(_model_tf);
        // end Get tf
    }else{
        if ( ros_api.ros_interface.is_topic_got_frame(_ROS_topic_id) ){
            // Get tf
            bool tf_successed = false;
            glm::mat4 _model_tf = ROStf2GLMmatrix(ros_api.get_tf(_ROS_topic_id, tf_successed));
            set_pose_modle_ref_by_world(_model_tf);
            // end Get tf
        }
    }
    //--------------------------------//
    // end Update transform


    // // test
    // //----------------------//
    // static int _count = 0;
    // insert_text( text2Din3D_data( "Hello world: " + std::to_string(_count) + std::string("\nSecond line\n\tThird line\nABCDEFGabcdefg"), glm::vec2(0.0f), 1.0f, glm::vec3(1.0f,1.0f,0.0f) ) );
    // //
    // insert_text( text2Din3D_data( "Text2Din3D pos_mode=0: " + std::to_string(_count), glm::vec2(10.0f), 1.0f, glm::vec3(0.0f,1.0f,0.0f) , ALIGN_X::LEFT, ALIGN_Y::TOP, 0) );
    // insert_text( text2Din3D_data( "Text2Din3D pos_mode=1: " + std::to_string(_count), glm::vec2(1.0f), 1.0f, glm::vec3(0.0f,1.0f,0.0f) , ALIGN_X::LEFT, ALIGN_Y::TOP, 1) );
    // /*
    // for (size_t _k=0; _k < 1000; ++_k){
    //     insert_text( text2Din3D_data("Text #" + std::to_string(_k) + ": " + std::to_string(_count), glm::vec2( 0.0f, float(_k))) );
    // }
    // */
    //
    // 2D text
    // insert_text( text2Dflat_data( "text2Dflat\n\tpos_mode=1\n\tsize_mode=0\n\tis_fullviewport=true\n\tis_background=false\nseq: " + std::to_string(_count), glm::vec2(0.5f), 24, glm::vec3(0.0f,0.0f,1.0f) , ALIGN_X::LEFT, ALIGN_Y::TOP, 1, 0, true, false) );
    // insert_text( text2Dflat_data( "text2Dflat\n\tpos_mode=1\n\tsize_mode=0\n\tis_fullviewport=true\n\tis_background=true\nseq: " + std::to_string(_count), glm::vec2(-0.5f), 24, glm::vec3(0.0f,0.0f,0.5f) , ALIGN_X::LEFT, ALIGN_Y::TOP, 1, 0, true, true) );
    //
    // insert_text( text2Dflat_data( "--text2Dflat\n\tpos_mode=1\n\tsize_mode=0\n\tis_fullviewport=false\n\tis_background=false\nseq: " + std::to_string(_count), glm::vec2(-1.0f, 1.0f), 24, glm::vec3(1.0f,0.0f,0.0f) , ALIGN_X::LEFT, ALIGN_Y::TOP, 0, 0, false, false) );
    //
    // //
    // insert_text( text3D_data("Text3D") );
    // //
    // // insert_text( text_billboard_data("The billboard" ));
    // insert_text(
    //     text_billboard_data(
    //         "The billboard\nSecond line\nSeq: " + std::to_string(_count),
    //         glm::vec3( 0.0f, 0.0f, 0.0f),
    //         glm::vec2(0.0f, 0.0f),
    //         1.0f,
    //         glm::vec3(1.0f, 0.8f, 0.2f),
    //         ALIGN_X::RIGHT,
    //         ALIGN_Y::CENTER
    //     )
    // );
    // /*
    // for (size_t _k=0; _k < 1000; ++_k){
    //     insert_text(
    //         text_billboard_data(
    //             "billboard #" + std::to_string(_k) + ": " + std::to_string(_count),
    //             glm::vec3( float(_k)*2.0f, 0.0f, 2.0f),
    //             glm::vec2(0.0f, 0.0f)
    //         )
    //     );
    // }
    // */
    //
    //
    // //
    // for (size_t _k=0; _k < 3; ++_k){
    //     insert_text(
    //         text_freeze_board_data(
    //             "I am freezed board #" + std::to_string(_k) + ": " + std::to_string(_count),
    //             glm::vec3( float(_k)*20.0f, 0.0f, 10.0f),
    //             glm::vec2(0.0f, 0.0f),
    //             24,
    //             glm::vec3(1.0f, 0.0f, 1.0f),
    //             ALIGN_X::CENTER,
    //             ALIGN_Y::CENTER
    //         )
    //     );
    // }
    //
    // //
    // _count++;
    // //
    // //----------------------//
    // // end test
}
void rmImageArray::Render(std::shared_ptr<ViewManager> &_camera_ptr){
    glBindVertexArray(m_shape.vao);
	_program_ptr->UseProgram();
    // queues
    //--------------------------------//
    while (!text2Dflat_queue.empty()){
        _draw_one_text2Dflat(_camera_ptr, text2Dflat_queue.front() );
        text2Dflat_queue.pop();
    }
    while (!text2Din3D_queue.empty()){
        _draw_one_text2Din3D(_camera_ptr, text2Din3D_queue.front() );
        text2Din3D_queue.pop();
    }
    while (!text3D_queue.empty()){
        _draw_one_text3D(_camera_ptr, text3D_queue.front() );
        text3D_queue.pop();
    }
    while (!text_billboard_queue.empty()){
        _draw_one_text_billboard(_camera_ptr, text_billboard_queue.front() );
        text_billboard_queue.pop();
    }
    while (!text_freeze_board_queue.empty()){
        _draw_one_text_freeze_board(_camera_ptr, text_freeze_board_queue.front() );
        text_freeze_board_queue.pop();
    }
    //--------------------------------//

    // buffers
    //--------------------------------//
    for (size_t i=0; i < text2Dflat_buffer.size(); ++i){
        _draw_one_text2Dflat(_camera_ptr, text2Dflat_buffer[i] );
    }
    for (size_t i=0; i < text2Din3D_buffer.size(); ++i){
        _draw_one_text2Din3D(_camera_ptr, text2Din3D_buffer[i] );
    }
    for (size_t i=0; i < text3D_buffer.size(); ++i){
        _draw_one_text3D(_camera_ptr, text3D_buffer[i] );
    }
    for (size_t i=0; i < text_billboard_buffer.size(); ++i){
        _draw_one_text_billboard(_camera_ptr, text_billboard_buffer[i] );
    }
    for (size_t i=0; i < text_freeze_board_buffer.size(); ++i){
        _draw_one_text_freeze_board(_camera_ptr, text_freeze_board_buffer[i] );
    }
    //--------------------------------//

    _program_ptr->CloseProgram();
}
void rmImageArray::Reshape(const glm::ivec2 & viewport_size_in){
    _viewport_size = viewport_size_in;
    // _viewport_size = _camera_ptr->GetViewportSize();
    updateBoardGeo();
}

void rmImageArray::setup_image_dictionary(std::vector<std::string> &image_name_list_in){
    image_name_list = image_name_list_in;
    std::vector<std::string> image_path_list;
    // Generate full paths
    for (size_t i=0; i < image_name_list.size(); ++i){
        image_path_list.push_back( get_full_Assets_path(image_name_list[i]) );
    }
    //
    atlas_ptr.reset(new ATLAS_IMAGE(image_path_list));
    std::cout << "The atlas<" << atlas_ptr->font_size << "> is created.\n";
    //
}


// Different draw methods
//--------------------------------------------------------//
void rmImageArray::_draw_one_text2Dflat(std::shared_ptr<ViewManager> &_camera_ptr, text2Dflat_data &_data_in){
    // static int _count = 0;
    /*
    pos_mode:
        0 - OpenCV pixel x:[0,ncol] "right", y:[0,nrow] "down"
        1 - OpenGL normalizd coordinate x:[-1,1] "right", y:[-1,1] "up"
    */
    glm::vec2 nl_position_2D;   // normalized
    glm::vec2 nl_size;          // normalized
    if (_data_in.is_fullviewport){
        //
        if (_data_in.pos_mode == 0){
            glm::vec2 _factor(2.0/float(_viewport_size[0]), 2.0/float(_viewport_size[1]) );
            nl_position_2D = _data_in.position_2D * _factor * glm::vec2(1.0f, -1.0f) + glm::vec2(-1.0f, 1.0f);
        }else{ // 1
            nl_position_2D = _data_in.position_2D;       // [-1, 1]
        }
        //
        if (_data_in.size_mode == 0){
            glm::vec2 _factor(2.0/float(_viewport_size[0]), 2.0/float(_viewport_size[1]) );
            nl_size = glm::vec2(_data_in.size) * _factor; // the size in ratio of the whole board --> OpenGL coordinate: [-1, 1]
        }else{
            nl_size[0] = _data_in.size * 2.0 * ( float(_viewport_size[1]) / float(_viewport_size[0]) ); //  the size is based i=on "height"
            nl_size[1] = _data_in.size * 2.0; // the size in ratio of the whole board --> OpenGL coordinate: [-1, 1]
        }
        glUniformMatrix4fv(uniforms.mv_matrix, 1, GL_FALSE, value_ptr( glm::mat4(1.0f) ));
    }else{ // Note full viewport
        //
        if (_data_in.pos_mode == 0){
            glm::vec2 _factor(2.0/float(shape.board_width), 2.0/float(shape.board_height) );
            nl_position_2D = (_data_in.position_2D * _factor) * glm::vec2(1.0f, -1.0f) + glm::vec2(-1.0f, 1.0f);
        }else{ // 1
            nl_position_2D = _data_in.position_2D;       // [-1, 1]
        }
        //
        if (_data_in.size_mode == 0){
            glm::vec2 _factor(2.0/float(shape.board_width), 2.0/float(shape.board_height) );
            nl_size = glm::vec2(_data_in.size) * _factor; // the size in ratio of the whole board --> OpenGL coordinate: [-1, 1]
        }else{
            nl_size[0] = _data_in.size * 2.0 / shape.board_aspect_ratio; //  the size is based i=on "height"
            nl_size[1] = _data_in.size * 2.0; // the size in ratio of the whole board --> OpenGL coordinate: [-1, 1]
        }
        glUniformMatrix4fv(uniforms.mv_matrix, 1, GL_FALSE, value_ptr( m_shape.model * m_shape.shape));
    }

    // The transformation matrices and projection matrices
    glUniformMatrix4fv(uniforms.proj_matrix, 1, GL_FALSE, value_ptr(glm::mat4(1.0)));
    if (_data_in.is_background){
        glUniform1f(uniforms.pose2D_depth, 1.0);
    }else{
        glUniform1f(uniforms.pose2D_depth, 0.0);
    }
    // RenderText("Hello world: " + std::to_string(++_count) + std::string("\nSecond line\n\tThird line\nABCDEFGabcdefg"), atlas_ptr, 0.0, 0.0, 1.0, 1.0, glm::vec3(1.0f, 1.0f, 0.0f));
    RenderText(
        _data_in.text,
        atlas_ptr,
        nl_position_2D[0],
        nl_position_2D[1],
        nl_size[0],
        nl_size[1],
        _data_in.color,
        _data_in.align_x,
        _data_in.align_y
    );
    //--------------------------------//
}
void rmImageArray::_draw_one_text2Din3D(std::shared_ptr<ViewManager> &_camera_ptr, text2Din3D_data &_data_in){
    // static int _count = 0;
    /*
    pos_mode:
        0 - meter in 3D perspective view (3D text2D only)
        1 - OpenGL normalizd coordinate x:[-1,1] "right", y:[-1,1] "up"
    */
    glm::vec2 nl_position_2D;   // normalized
    glm::vec2 nl_size;          // normalized
    if (_data_in.pos_mode == 0){
        nl_position_2D = _data_in.position_2D;
        nl_size = glm::vec2(_data_in.size);
        glUniformMatrix4fv(uniforms.mv_matrix, 1, GL_FALSE, value_ptr( get_mv_matrix(_camera_ptr, m_shape.model) ));
    }else{ // 1
        nl_position_2D = _data_in.position_2D;       // [-1, 1]
        nl_size[0] = _data_in.size * 2.0 / shape.board_aspect_ratio; //  the size is based i=on "height"
        nl_size[1] = _data_in.size * 2.0; // the size in ratio of the whole board --> OpenGL coordinate: [-1, 1]
        glUniformMatrix4fv(uniforms.mv_matrix, 1, GL_FALSE, value_ptr( get_mv_matrix(_camera_ptr, m_shape.model * m_shape.shape) ));
    }
    // m_shape.model = translateMatrix * rotateMatrix * scaleMatrix;
    // The transformation matrices and projection matrices
    // glUniformMatrix4fv(uniforms.mv_matrix, 1, GL_FALSE, value_ptr( get_mv_matrix(_camera_ptr, m_shape.model * m_shape.shape * transform_m) ));
    glUniformMatrix4fv(uniforms.proj_matrix, 1, GL_FALSE, value_ptr(_camera_ptr->GetProjectionMatrix()));
    glUniform1f(uniforms.pose2D_depth, 0.0001); // Pop out (floating) a little bit to prevent the intersection with image board
    // RenderText("Hello world: " + std::to_string(++_count) + std::string("\nSecond line\n\tThird line\nABCDEFGabcdefg"), atlas_ptr, 0.0, 0.0, 1.0, 1.0, glm::vec3(1.0f, 1.0f, 0.0f));
    RenderText(
        _data_in.text,
        atlas_ptr,
        nl_position_2D[0],
        nl_position_2D[1],
        nl_size[0],
        nl_size[1],
        _data_in.color,
        _data_in.align_x,
        _data_in.align_y
    );
    //--------------------------------//
}
void rmImageArray::_draw_one_text3D(std::shared_ptr<ViewManager> &_camera_ptr, text3D_data &_data_in){

    // The transformation matrices and projection matrices
    glUniformMatrix4fv(uniforms.mv_matrix, 1, GL_FALSE, value_ptr( get_mv_matrix(_camera_ptr, _data_in.pose_ref_point) ));
    glUniformMatrix4fv(uniforms.proj_matrix, 1, GL_FALSE, value_ptr(_camera_ptr->GetProjectionMatrix()));
    glUniform1f(uniforms.pose2D_depth, 0.0);
    //
    RenderText(
        _data_in.text,
        atlas_ptr,
        -1*(_data_in.offset_ref_point_2D[0]),
        -1*(_data_in.offset_ref_point_2D[1]),
        _data_in.size,
        _data_in.size,
        _data_in.color,
        _data_in.align_x,
        _data_in.align_y
    );
    //--------------------------------//
}
void rmImageArray::_draw_one_text_billboard(std::shared_ptr<ViewManager> &_camera_ptr, text_billboard_data &_data_in){
    // Calculate model matrix
    glm::mat4 view_m = get_mv_matrix(_camera_ptr, glm::mat4(1.0)); // _camera_ptr->GetModelViewMatrix();
    view_m[3] = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
    glm::mat4 _model_m = glm::transpose(view_m);
    _model_m[3] = glm::vec4(_data_in.position_ref_point, 1.0f);
    //

    // The transformation matrices and projection matrices
    glUniformMatrix4fv(uniforms.mv_matrix, 1, GL_FALSE, value_ptr( get_mv_matrix(_camera_ptr, _model_m) ));
    glUniformMatrix4fv(uniforms.proj_matrix, 1, GL_FALSE, value_ptr(_camera_ptr->GetProjectionMatrix()));
    glUniform1f(uniforms.pose2D_depth, 0.0);
    //
    RenderText(
        _data_in.text,
        atlas_ptr,
        -1*(_data_in.offset_ref_point_2D[0]),
        -1*(_data_in.offset_ref_point_2D[1]),
        _data_in.size,
        _data_in.size,
        _data_in.color,
        _data_in.align_x,
        _data_in.align_y
    );
    //--------------------------------//
}
void rmImageArray::_draw_one_text_freeze_board(std::shared_ptr<ViewManager> &_camera_ptr, text_freeze_board_data &_data_in){
    // Calculate model matrix
    glm::mat4 view_m = get_mv_matrix(_camera_ptr, glm::mat4(1.0)); // _camera_ptr->GetModelViewMatrix();
    //
    // glm::vec4 _ref_in_canonical = ( _camera_ptr->GetProjectionMatrix()*view_m*glm::vec4(_data_in.position_ref_point, 1.0f) );
    // GLfloat _z_ref = (_ref_in_canonical.z);
    GLfloat _z_ref = ( view_m*glm::vec4(_data_in.position_ref_point, 1.0f) ).z;
    // GLfloat _scale = _z_ref / (_camera_ptr->GetProjectionMatrix() )[2][3];
    GLfloat _scale = _z_ref / (_camera_ptr->GetProjectionMatrix() )[2][3] / ( (_camera_ptr->GetViewportSize())[1]/2 );
    //
    view_m[3] = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
    glm::mat4 _model_m = glm::transpose(view_m);
    _model_m[3] = glm::vec4(_data_in.position_ref_point, 1.0f);
    //

    // The transformation matrices and projection matrices
    glUniformMatrix4fv(uniforms.mv_matrix, 1, GL_FALSE, value_ptr( get_mv_matrix(_camera_ptr, _model_m) ));
    glUniformMatrix4fv(uniforms.proj_matrix, 1, GL_FALSE, value_ptr(_camera_ptr->GetProjectionMatrix()));
    glUniform1f(uniforms.pose2D_depth, 0.0);
    //
    RenderText(
        _data_in.text,
        atlas_ptr,
        -1*(_data_in.offset_ref_point_2D[0]),
        -1*(_data_in.offset_ref_point_2D[1]),
        1.0*_scale*_data_in.size,
        1.0*_scale*_data_in.size,
        _data_in.color,
        _data_in.align_x,
        _data_in.align_y
    );
    //--------------------------------//
}
//--------------------------------------------------------//




//-----------------------------------------------//
// void rmImageArray::RenderText(const std::string &text, ATLAS_IMAGE *_atlas_ptr, float x, float y, float scale_x_in, float scale_y_in, glm::vec3 color) {
void rmImageArray::RenderText(
    const std::string &text,
    std::shared_ptr<ATLAS_IMAGE> &_atlas_ptr,
    float x_in,
    float y_in,
    float scale_x_in,
    float scale_y_in,
    glm::vec3 color,
    ALIGN_X align_x,
    ALIGN_Y align_y
){
    // Check
    if (!atlas_ptr){
        std::cout << "Atlas is not created.\n";
        return;
    }
    //

    GLfloat scale_x = scale_x_in/GLfloat(_atlas_ptr->font_size);
    GLfloat scale_y = scale_y_in/GLfloat(_atlas_ptr->font_size);
    //
    glUniform3f( uniforms.textColor, color.x, color.y, color.z);

    // Calculate the bound
    float x_min = x_in;
    float x_max = x_in;
    float y_min = y_in;
    float y_max = y_in;


    //
    float x = x_in;
    float y = y_in;

	// point vector_list[ 4 * text.size() ];
    std::vector<point> vector_list( 4 * text.size() );
    // vector_list.resize( 4 * text.size() );
	int _idx_count = 0;
    int _valid_word_count = 0;

    //
    int _word_per_line = 0;
    int _max_word_per_line = 0;
    int _line_count = 1;

    // Iterate through all characters
    std::string::const_iterator p;
    for (p = text.begin(); p != text.end(); p++) {
        if (*p == '\n'){
            x = x_in;
            y -= _atlas_ptr->font_size * scale_y;
            //
            _line_count++;
            _word_per_line = 0;
            continue;
        }
        //
        _word_per_line++;
        if (_max_word_per_line < _word_per_line){
            _max_word_per_line = _word_per_line;
        }
        //
        if (*p == '\t'){
            x += (_atlas_ptr->_ch[' '].ax * scale_x)*8;
            continue;
        }
        //
        if (size_t(*p) >= _atlas_ptr->_ch.size()){
            std::cout << "[imageArray] Warnning!! The character is out of range.\n";
            continue;
        }
        //
        // for (uint8_t * p = (const uint8_t *)text; *p; p++) {
		/* Calculate the vertex and texture coordinates */
		float x2 = x + _atlas_ptr->_ch[*p].bl * scale_x;
		float y2 = -y - _atlas_ptr->_ch[*p].bt * scale_y;
		float w = _atlas_ptr->_ch[*p].bw * scale_x;
		float h = _atlas_ptr->_ch[*p].bh * scale_y;

		/* Advance the cursor to the start of the next character */
		x += _atlas_ptr->_ch[*p].ax * scale_x;
		y += _atlas_ptr->_ch[*p].ay * scale_y;


        // Find out the boundaries
        //----------------------------//
        if (x_max < (x2+w)){
            x_max = (x2+w);
        }
        if (x_min > (x2)){
            x_min = (x2);
        }
        if (y_max < -(y2)){
            y_max = -(y2);
        }
        if (y_min > -(y2+h)){
            y_min = -(y2+h);
        }
        //----------------------------//


		/* Skip glyphs that have no pixels */
        if (!w || !h)
        {  // For example: space --> " "
          continue;
        }

        /*
        Sequence of index
        1 - 2
        | / |
        3 - 4
        */
        // float _x_left = x2;
        // float _x_right = x2+w;
        // float _y_up = -y2;
        // float _y_down = -y2-h;
        //
        float s_left = _atlas_ptr->_ch[*p].tx;
        float s_right = _atlas_ptr->_ch[*p].tx + _atlas_ptr->_ch[*p].bw / float(_atlas_ptr->_tex_w);
        float t_up = _atlas_ptr->_ch[*p].ty;
        float t_down = _atlas_ptr->_ch[*p].ty + _atlas_ptr->_ch[*p].bh / float(_atlas_ptr->_tex_h);
		vector_list[_idx_count++] = (point) {
    		x2, -y2,
            s_left, t_up};
		vector_list[_idx_count++] = (point) {
    		x2+w, -y2,
            s_right, t_up};
		vector_list[_idx_count++] = (point) {
    		x2, -y2-h,
            s_left, t_down};
		vector_list[_idx_count++] = (point) {
    		x2+w, -y2-h,
            s_right, t_down};

        // Check if the definition of point is correct
        // if ( (-y2) >= (-y2-h)){
        //     std::cout << "right\n";
        // }else{
        //     std::cout << "wrong\n";
        // }
        //


        _valid_word_count++;

	}

    float _max_w = x_max - x_min;
    float _max_h = y_max - y_min;
    // std::cout << "(_max_w, _max_h) = " << _max_w << ", "<<  _max_h << "\n";

    // ref_point
    // Method 1: too precise that it will vibrate when word change
    // ref_point = glm::vec2(_max_w/2.0f, -1*_max_h/2.0f);
    // Method 2: less precise, using word (per line) count and line count
    float _space_x = 2*(_atlas_ptr->_ch[' '].ax * scale_x);
    float _space_y = _atlas_ptr->font_size * scale_y;
    //
    ref_point = glm::vec2(x_min - x_in, y_max - y_in);
    //
    switch (align_x){
        case ALIGN_X::LEFT:
            ref_point[0] += 0.0f;
            break;
        case ALIGN_X::CENTER:
            ref_point[0] += _max_word_per_line*_space_x/2.0f;
            break;
        case ALIGN_X::RIGHT:
            ref_point[0] += _max_w;
            break;
        default:
            // std::cout << "default align_x\n";
            ref_point[0] += 0.0f;
            break;
    }
    switch (align_y){
        case ALIGN_Y::TOP:
            ref_point[1] += 0.0f;
            break;
        case ALIGN_Y::CENTER:
            // ref_point[1] = -1*_line_count*_space_y/2.0f;
            ref_point[1] += -1*(_max_h/2.0f);
            // std::cout << "ref_point = " << ref_point.x << ", " << ref_point.y << "\n";
            break;
        case ALIGN_Y::BUTTON:
            ref_point[1] += -1*_max_h;
            break;
        default:
            // std::cout << "default align_y\n";
            ref_point[1] += 0.0f;
            break;
    }
    // ref_point = glm::vec2(_max_word_per_line*_space_x/2.0f, -1*_line_count*_space_y/2.0f);
    glUniform2f( uniforms.ref_point, ref_point.x, ref_point.y);



    // Update content of VBO memory
    glBindBuffer(GL_ARRAY_BUFFER, m_shape.vbo);


	// Draw all the character on the screen in one go
	// glBufferData(GL_ARRAY_BUFFER, sizeof vector_list, vector_list, GL_DYNAMIC_DRAW);


    // Directly assign data to memory of GPU
    //--------------------------------------------//
    point * vertex_ptr = (point *)glMapBufferRange(GL_ARRAY_BUFFER, 0, _idx_count* sizeof(point), GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT);
    for (size_t i = 0; i < _idx_count; i++)
    {
        vertex_ptr[i] = vector_list[i];
    }
    glUnmapBuffer(GL_ARRAY_BUFFER);
    //--------------------------------------------//

    // Activate corresponding render state
    glActiveTexture(GL_TEXTURE0);
	// Use the texture containing the atlas
	glBindTexture(GL_TEXTURE_2D, _atlas_ptr->TextureID);

    // Draw
	// glDrawArrays(GL_TRIANGLES, 0, _idx_count);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_shape.ebo);
    glDrawElements(GL_TRIANGLES, _valid_word_count*_num_vertex_idx_per_box, GL_UNSIGNED_INT, 0); // Using ebo

    glBindTexture(GL_TEXTURE_2D, 0);

}
