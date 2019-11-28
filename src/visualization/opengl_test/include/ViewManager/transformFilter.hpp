#ifndef TRANSFORM_FILTER_H
#define TRANSFORM_FILTER_H

#include "Common.h"

class TRANSFORM_FILTER{
public:
    TRANSFORM_FILTER();
    void reset(glm::mat4 tf_in=glm::mat4(1.0f) );
    //
    void setInput(const glm::mat4 & tf_in);
    glm::mat4 iterateOnce();
    glm::mat4 getOutput();


    //
    glm::mat3 getRotationMatrix(const glm::mat4 & tf_in);
    void putRotationMatrix(glm::mat3 R_in, glm::mat4 &tf_in_out);
    glm::mat3 getScewSymmetricMatreix(const glm::vec3 & w_in);
    glm::vec3 getVectorFromScewSymetry(const glm::mat3 & W_in);
    //
    glm::quat RotateTowards(glm::quat q1, glm::quat q2, float maxAngle);

protected:
    glm::mat4 tf_input;
    glm::quat R_input;
    glm::vec3 p_input;

    //
    glm::mat4 tf_filtered;
    glm::quat R_filtered;
    glm::vec3 p_filtered;






private:

};

#endif // TRANSFORM_FILTER_H
