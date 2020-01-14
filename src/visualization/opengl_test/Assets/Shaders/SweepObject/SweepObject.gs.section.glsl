#version 410

#define _max_num_vertex_of_curve 100
#define _max_num_vertex_of_shape 50
#define _max_verteces_out 100 // 2*_max_num_vertex_of_shape
layout(lines, invocations = 1) in;
layout(line_strip, max_vertices = _max_verteces_out) out;
// layout(line_strip, max_vertices = 100) out;

in int gl_PrimitiveIDIn[];
out vec4 gsOutColor;

uniform mat4 mv_matrix;
uniform mat4 proj_matrix;
// The list of rotation matrices for direction
uniform mat4 lookat_matrix[_max_num_vertex_of_curve]; // The size corespond with _max_num_vertex_of_curve
// The list of vertex on a cross section
uniform vec3 section_vertexes[_max_num_vertex_of_shape];
uniform int _num_vertex_of_shape;
uniform int shape_mode; // 0 - open-loop, 1 - close-loop
//
const float PI = 3.1415926;

in VS_OUT
{
	vec4 color;
} gs_in[];

void main()
{
	mat4 mvp_matrix =  proj_matrix * mv_matrix;

    int _j = 0;
    int _num_vertex_draw = _num_vertex_of_shape;
    if (shape_mode == 1){
        _num_vertex_draw += 1;
    }
    for (int i=0; i < _num_vertex_draw; ++i){
        // vec4 _v_pre  = lookat_matrix[gl_PrimitiveIDIn[0] ]    * vec4(section_vertexes[_j], 0.0);
        vec4 _v_post = lookat_matrix[gl_PrimitiveIDIn[0] + 1] *  vec4(section_vertexes[_j], 0.0);
        //
        // gl_Position = mvp_matrix * (gl_in[0].gl_Position + _v_pre);
		// gsOutColor = gs_in[0].color;
		// EmitVertex();
		gl_Position = mvp_matrix * (gl_in[1].gl_Position + _v_post);
		gsOutColor = gs_in[1].color;
		EmitVertex();
        //
        _j++;
        _j %= _num_vertex_of_shape;
    }

	EndPrimitive();
}
