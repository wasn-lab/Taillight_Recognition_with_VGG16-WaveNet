#version 410

layout(points, invocations = 1) in;
layout(line_strip, max_vertices = 100) out;
// layout(line_strip, max_vertices = 100) out;

// in int gl_PrimitiveIDIn[];
out vec4 gsOutColor;

uniform mat4 mv_matrix;
uniform mat4 proj_matrix;
//
const float PI = 3.1415926;
const int   point_on_circle = 30;

in VS_OUT
{
    float radious;
	vec4 color;
} vs_out[];

void main()
{
	mat4 mvp_matrix =  proj_matrix * mv_matrix;
    //
    vec4 circle_point_rel_0 = vec4(vs_out[0].radious, 0, 0, 0);
    vec4 circle_point_rel_i = vec4(0, 0, 0, 0);
    //
    float angle_devide = (2.0*PI)/point_on_circle;
    for (int i=0; i < point_on_circle; ++i){
        angle_devide*i;
        circle_point_rel_i = vec4(vs_out[0].radious*cos(angle_devide*i), vs_out[0].radious*sin(angle_devide*i), 0, 0);
        gl_Position = mvp_matrix * (gl_in[0].gl_Position + circle_point_rel_i);
        gsOutColor = vs_out[0].color;
        EmitVertex();
    }
    gl_Position = mvp_matrix * (gl_in[0].gl_Position + circle_point_rel_0);
    gsOutColor = vs_out[0].color;
    EmitVertex();
    //

	EndPrimitive();
}
