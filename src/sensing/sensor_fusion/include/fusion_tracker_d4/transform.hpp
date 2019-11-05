#ifndef TRANSFORM_HPP
#define TRANSFORM_HPP


extern float front_t[3];
extern float front_R[9];
extern float front_K[9];
extern float inv_front_R[9];
extern float inv_front_K[9];

extern float left_t[3];
extern float left_R[9];
extern float left_K[9];
extern float inv_left_R[9];
extern float inv_left_K[9];

extern float right_t[3];
extern float right_R[9];
extern float right_K[9];
extern float inv_right_R[9];
extern float inv_right_K[9];

int initMatrix(const char* path, float t[3], float R[9], float K[9], float invK[9], float invR[9]);
int matrix_inverse_3x3d (const float m[9], float inverse[9]);
float matrix_determinant_3x3d (const float m[9]);

void matrix_vector_multiply_3x3_3d (const float m[9], const float v[3], float result[3]);
void vector_add_3d (const float v1[3], const float v2[3], float result[3]);
void vector_subtract_3d (const float v1[3], const float v2[3], float result[3]);


#endif

