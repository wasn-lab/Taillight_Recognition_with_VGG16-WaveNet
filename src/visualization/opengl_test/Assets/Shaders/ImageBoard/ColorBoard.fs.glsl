#version 410 core

uniform vec4 color_vec4;

out vec4 color;

in VS_OUT{
	vec2 texcoord;
} fs_in;

void main(void)
{
    color = color_vec4;
}
