#include "IIR.h"
/*-----------------------------*/
//   Order of the numerator: m
//   Orser of the denominator: n
//--------------------------------//
//
//    y     gain*( sum(i from 0 to m) b[i]*x[k-i] )
//   --- = --------------------------------------------
//    u     1 + sum( i from 1 to n) a[i]*y[k-i]
//
//   y[k] = gain*( sum(i from 0 to m) b[i]*x[k-i] ) - sum( i from 1 to n) a[i]*y[k-i]
//
//   Compatible with the coefficient generated by Matlab fdatool
//
//   by C.F. Huang from LDSC
/*-----------------------------*/

Circular_buffer::Circular_buffer(){
    n = 0;
    ind = 0;
}
Circular_buffer::Circular_buffer(int n_in){
    n = n_in;
    A.resize(n);
    for (size_t i=0; i< A.size(); i++)
        A[i] = 0;

    ind = 0;
}
//
void Circular_buffer::Init(int n_in){ // Input: buffer size
    n = n_in;
    A.resize(n);
    for (size_t i=0; i< A.size(); i++)
        A[i] = 0;
    ind = 0;
}
//
float Circular_buffer::Get(int i){ // Get the value of the (t-i)th element, where t stands for the current sample
    int ind_e = ind-i;
    while (ind_e < 0){
        ind_e += A.size();
    }
    return A[ind_e];
}
void Circular_buffer::Insert(float x_new){ // Pop the oldest element and push a new element
    ind++;
    while (ind >= A.size()){
        ind -= A.size();
    }
    A[ind] = x_new;
}
void Circular_buffer::reset(float val){ // Reset all elements to val
    for (size_t i = 0; i< A.size(); i++)
        A[i] = val;
    ind = 0;
}

// IIR
//---------------------------------------//
IIR::IIR(int m, int n): m(m),n(n) {
    x.Init(m+1);
    y.Init(n);
}
//
void IIR::Assign_parameters(float* b_in, float* a_in, float gain_in){
    //b.assign(b_in,b_in+m+1);
    // a.assign(a_in,a_in+n);
    b.resize(m+1);
    a.resize(n);
    for (size_t i = 0; i < n; i++){
        a[i] = a_in[i+1];
    }
    gain = gain_in;
    for (size_t i = 0; i < b.size(); i++){
        b[i] = gain*b_in[i];
    }
}
//
float IIR::Iterate_once(float x_in){
    x.Insert(x_in);
    float y_new = 0;
    // Numerator
    for (int i = 0; i <= m; i++ ){
        y_new += b[i]*x.Get(i);
    }

    // Denominator
    for (int i = 0; i < n; i++){
        y_new -= a[i]*y.Get(i);
    }
    y.Insert(y_new);
    return y_new;
}
void IIR::Reset(float val){ // Rest all elements to val
    x.reset(val);
    y.reset(val);
}