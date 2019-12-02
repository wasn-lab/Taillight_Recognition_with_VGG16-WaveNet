#version 410 core

layout(location = 0) in vec4 position;
layout(location = 1) in vec4 color;

// uniform float time;
uniform mat4 mv_matrix;
uniform mat4 proj_matrix;

flat out vec4 starColor;

void main(void)
{
    vec4 newVertex = mv_matrix * position;
    // float size = 2.0;
    float size = 20.0/( abs(newVertex.z) );
    // float size = 20.0/( length(newVertex.xyz) ); //20.0/( abs(newVertex.z) );
    // size = smoothstep(1.0, 800.0, size);

    vec4 color_ = color;
    // vec4 color_ = smoothstep(0.2, 1.0, abs(newVertex.z) ) * color;
    // vec4 color_ = size * color * 1.0;
    // color_.r = 0.0;
    // color_.g = 0.0;
    color_.a = 0.6;
    // color_.a = 0.8;
    starColor = color_;

    newVertex = proj_matrix * newVertex;


    gl_Position = newVertex;
	gl_PointSize = size;
}
