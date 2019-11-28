#version 330 core
in vec2 TexCoords;
out vec4 color;

uniform sampler2D text;
uniform vec3 textColor;

void main()
{
    vec4 sampled = vec4(1.0, 1.0, 1.0, texture(text, TexCoords).a);
    // Remove the artifact caused by zooming or rotation
    if (sampled.a >=0.5){
        sampled.a = 1.0;
    }else{
        sampled.a = 0.0;
    }
    color = vec4(textColor, 1.0) * sampled;
}
