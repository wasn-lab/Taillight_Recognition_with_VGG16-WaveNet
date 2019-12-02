#version 410 core

out vec4 color;

uniform float alpha;

in VS_OUT
{
	vec4 color;
} fs_in;

void main(void)
{
	vec4 texture_color = fs_in.color;
	if (alpha >= 0.0){
        texture_color.a = alpha;
    }
    color = texture_color;
}
