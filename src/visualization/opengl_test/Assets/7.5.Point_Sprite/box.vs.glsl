#version 410

in vec4 position;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;


out VS_OUT
{
	vec4 color;
} vs_out;

void main()
{
	gl_Position = projection * view * model * position;
    vs_out.color = vec4(0.0, 1.0, 0.0, 0.3);
}
