#include "CIRCULAR_BUFFER_VECTOR.h"

// First type of constructer
CIRCULAR_BUFFER_VECTOR::CIRCULAR_BUFFER_VECTOR(void):
        buffer(0,std::vector<float>(0))
{
    //
    buffer_size = 0;
    vector_size = 0;
    idx = 0;
}
// Second type of constructer
CIRCULAR_BUFFER_VECTOR::CIRCULAR_BUFFER_VECTOR(size_t buffer_size_in, size_t vector_size_in):
        buffer_size(buffer_size_in),
        vector_size(vector_size_in),
        buffer(buffer_size_in,std::vector<float>(vector_size_in,0.0))
{
    idx = 0;
}
//
void CIRCULAR_BUFFER_VECTOR::Init(size_t buffer_size_in, const std::vector<float> &initial_value){ // If using the first type of constructer, this function helps
    //
    buffer_size = buffer_size_in;
    vector_size = initial_value.size();
    idx = 0;
    //
    buffer.assign(buffer_size_in, initial_value);
}
void CIRCULAR_BUFFER_VECTOR::Reset(const std::vector<float> &value){ // Reset all the element without changeing the buffer size
    //
    buffer.assign(buffer_size, value);
    idx = 0;
}
// Element-wise operation
std::vector<float> CIRCULAR_BUFFER_VECTOR::Get(int i){ // Get the element that is i samples ago
    int idx_e = idx - i;
    while (idx_e < 0){
        idx_e += buffer_size;
    }
    return buffer[idx_e];
}
void CIRCULAR_BUFFER_VECTOR::Set(int i, const std::vector<float> &value){ // Set the element that is i samples ago to the "value"
    //
    int idx_e = idx - i;
    while (idx_e < 0){
        idx_e += buffer_size;
    }
    //
    buffer[idx_e] = value;
}
void CIRCULAR_BUFFER_VECTOR::Increase(int i, const std::vector<float> &increase_value, bool is_minus){ // vi += (or -=) increase_value
    //
    int idx_e = idx - i;
    while (idx_e < 0){
        idx_e += buffer_size;
    }
    //
    Get_VectorIncrement(buffer[idx_e], increase_value, is_minus);
}
// Iterate the buffer
void CIRCULAR_BUFFER_VECTOR::Insert(const std::vector<float> &x_new){ // Pop the oldest element and push a new element
    //
    idx++;
    while (idx >= buffer_size){
        idx -= (int)buffer_size;
    }
    buffer[idx] = x_new;
}
// Utilities
// Increment
void CIRCULAR_BUFFER_VECTOR::Get_VectorIncrement(std::vector<float> &v_a, const std::vector<float> &v_b, bool is_minus){ // v_a += (or -=) v_b
    // Size check
    if (v_a.size() != v_b.size()){
        v_a.resize(v_b.size());
    }
    //
    if (is_minus){ // -=
        for (size_t i = 0; i < v_b.size(); ++i){
            v_a[i] -= v_b[i];
        }
    }else{ // +=
        for (size_t i = 0; i < v_b.size(); ++i){
            v_a[i] += v_b[i];
        }
    }

}
