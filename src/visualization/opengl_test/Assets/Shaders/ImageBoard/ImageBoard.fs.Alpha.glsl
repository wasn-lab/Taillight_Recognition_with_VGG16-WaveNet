#version 410 core

uniform sampler2D tex;
//
uniform vec4 color_transform;
uniform float alpha;

out vec4 color;

in VS_OUT{
	vec2 texcoord;
} fs_in;

void main(void)
{
	vec4 texture_color = texture(tex, fs_in.texcoord);
    //
    // texture_color = color_transform * texture_color;
    if (alpha >= 0.0){
        texture_color.a = alpha;
    }
    color = texture_color;

}
