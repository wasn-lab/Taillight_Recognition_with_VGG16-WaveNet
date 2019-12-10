#version 410 core

uniform sampler2D tex;

out vec4 color;

in VS_OUT
{
vec2 texcoord;
} fs_in;

void main(void)
{
	vec4 texture_color = texture(tex, fs_in.texcoord);
	float grayscale_color = 0.299*texture_color.r + 0.587*texture_color.g + 0.114*texture_color.b;
    // color = vec4(grayscale_color, grayscale_color, grayscale_color, 1.0);
    // color = vec4(grayscale_color, 0.0, 0.0, 1.0);

    if (texture_color.a < 0.01)
		color = vec4(0,0,0,0);
	else
		color = texture_color;
    
}
