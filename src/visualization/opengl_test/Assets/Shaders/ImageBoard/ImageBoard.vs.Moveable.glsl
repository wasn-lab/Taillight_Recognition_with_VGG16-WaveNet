#version 410 core

layout(location = 0) in vec2 position;
layout(location = 1) in vec2 texcoord;

uniform mat4 mv_matrix;
// uniform mat4 proj_matrix;


out VS_OUT
{
	vec2 texcoord;
} vs_out;


void main(void)
{
	// gl_Position = vec4(position, 1.0, 1.0); // Note: z = 1.0 is the farest plane
    gl_Position = mv_matrix * vec4(position, 0.0, 1.0); // Note: z = 0, the nearest
    // gl_Position = proj_matrix * mv_matrix * vec4(position, 0.0, 1.0);
	vs_out.texcoord = texcoord;
}
