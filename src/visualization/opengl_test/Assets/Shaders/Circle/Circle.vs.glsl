#version 410 core

layout(location = 0) in vec4 position;
layout(location = 1) in float radious;
layout(location = 2) in vec4 color;

out VS_OUT
{
    float radious;
	vec4 color;
} vs_out;

void main()
{
	gl_Position = position;
    //
    vs_out.radious = radious;
    vs_out.color = color;
}
