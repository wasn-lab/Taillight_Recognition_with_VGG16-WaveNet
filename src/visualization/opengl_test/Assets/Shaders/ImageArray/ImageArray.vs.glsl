#version 330 core


layout (location = 0) in vec4 vertex; // <vec2 pos, vec2 tex>
out vec2 TexCoords;

uniform mat4 mv_matrix;
uniform mat4 proj_matrix;
uniform vec2 ref_point;
uniform float pose2D_depth; // default/foreground: 0.0, background: 1.0, floating: greater than 0.0

void main()
{
    // gl_Position = proj_matrix * mv_matrix * vec4(vertex.xy - ref_point, 0.0, 1.0);
    gl_Position = proj_matrix * mv_matrix * vec4(vertex.xy - ref_point, pose2D_depth, 1.0);
    TexCoords = vertex.zw;
}
