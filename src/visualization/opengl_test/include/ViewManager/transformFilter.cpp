#include "transformFilter.hpp"

TRANSFORM_FILTER::TRANSFORM_FILTER():
    tf_input(1.0f),
    R_input(0.0f, 0.0f, 0.0f, 1.0f),
    p_input(0.0f),
    tf_filtered(1.0f),
    R_filtered(0.0f, 0.0f, 0.0f, 1.0f),
    p_filtered(0.0f)
{

}


void TRANSFORM_FILTER::reset(glm::mat4 tf_in){
    tf_input = tf_in;
    R_input = glm::quat_cast(tf_input);
    p_input = tf_input[3].xyz();
    //
    tf_filtered = tf_input;
    R_filtered = glm::quat_cast(tf_filtered);
    p_filtered = tf_filtered[3].xyz();
}

void TRANSFORM_FILTER::setInput(const glm::mat4 & tf_in){
    tf_input = tf_in;
    R_input = glm::quat_cast(tf_input);
    p_input = tf_input[3].xyz();
}
glm::mat4 TRANSFORM_FILTER::iterateOnce(){
    // tf_filtered = tf_input;
    //
    float mixFactor = 0.2f;
    R_filtered = glm::mix(R_filtered, R_input, mixFactor);
    // R_filtered = RotateTowards(R_filtered, R_input, 0.017);
    p_filtered += mixFactor*(p_input - p_filtered);
    //
    tf_filtered = glm::toMat4(R_filtered);
    tf_filtered[3] = glm::vec4(p_filtered, 1.0f);
    return tf_filtered;
}
glm::mat4 TRANSFORM_FILTER::getOutput(){
    return tf_filtered;
}




//
glm::mat3 TRANSFORM_FILTER::getRotationMatrix(const glm::mat4 & tf_in){
    glm::mat3 _R;
    for (size_t i=0; i<3; ++i){
        _R[i] = tf_in[i].xyz();
    }
    return _R;
}
glm::mat3 TRANSFORM_FILTER::getScewSymmetricMatreix(const glm::vec3 & w_in){
    glm::mat3 _W(0.0f);
    _W[1][0] = -w_in.z;
    _W[2][0] = w_in.y;
    _W[2][1] = -w_in.x;
    _W[0][1] = w_in.z;
    _W[0][2] = -w_in.y;
    _W[1][2] = w_in.x;
    return _W;
}
glm::vec3 TRANSFORM_FILTER::getVectorFromScewSymetry(const glm::mat3 & W_in){
    return glm::vec3( W_in[1][2], W_in[2][0], W_in[0][1]);
}



//
glm::quat TRANSFORM_FILTER::RotateTowards(glm::quat q1, glm::quat q2, float maxAngle){

	if( maxAngle < 0.001f ){
		// No rotation allowed. Prevent dividing by 0 later.
		return q1;
	}

	float cosTheta = glm::dot(q1, q2);

	// q1 and q2 are already equal.
	// Force q2 just to be sure
	if(cosTheta > 0.9999f){
		return q2;
	}

	// Avoid taking the long path around the sphere
	if (cosTheta < 0){
	    q1 = q1*-1.0f;
	    cosTheta *= -1.0f;
	}

	float angle = glm::acos(cosTheta);

	// If there is only a 2&deg; difference, and we are allowed 5&deg;,
	// then we arrived.
	if (angle < maxAngle){
		return q2;
	}

	float fT = maxAngle / angle;
	angle = maxAngle;

	glm::quat res = (glm::sin((1.0f - fT) * angle) * q1 + glm::sin(fT * angle) * q2) / glm::sin(angle);
	res = glm::normalize(res);
	return res;

}
