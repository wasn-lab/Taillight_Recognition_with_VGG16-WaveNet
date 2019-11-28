#version 410

layout(location = 0) in vec3 iv3vertex;
layout(location = 1) in vec2 iv2tex_coord;
layout(location = 2) in vec3 iv3normal;

uniform mat4 mv_matrix;
uniform mat4 proj_matrix;

out VertexData
{
    vec3 N; // eye space normal
    vec3 L; // eye space light vector
    vec3 H; // eye space halfway vector
    vec2 texcoord;
} vertexData;

void main()
{
	gl_Position = proj_matrix * mv_matrix * vec4(iv3vertex, 1.0);
    vertexData.texcoord = iv2tex_coord;
}
