//
#ifndef PI
#define PI 3.1415926
#endif
//
#ifndef FILTER_LIB_H
#define FILTER_LIB_H
//
#include "IIR.h"
#include <cmath>
#include <vector>

// using std::vector;

//--------------------LPF---------------------//
class LPF{ // Low-pass filter
public:
    float output;

    LPF(float samplingTime, float cutOff_freq_Hz_in); // cutOff_freq_Hz_in is in "Hz"
    float filter(float input);
    void reset(float input);

private:
    float Ts;
    float cutOff_freq_Hz; // Hz
    float alpha_Ts;
    float One_alpha_Ts;

    // Flag
    bool Flag_Init;
};

//--------------------LPF_vector---------------------//
class LPF_vector{ // Vectorized low-pass filter
public:
    std::vector<float> output;

    LPF_vector(size_t dimension, float samplingTime, float cutOff_freq_Hz_in); // cutOff_freq_Hz_in is in "Hz"
    std::vector<float> filter(const std::vector<float> &input);
    void reset(const std::vector<float> &input);

private:
    size_t n;
    float Ts;
    float cutOff_freq_Hz; // Hz
    float alpha_Ts;
    float One_alpha_Ts;

    // Flag
    bool Flag_Init;

    //
    std::vector<float> zeros; // Zero std::vector [0;0;0]
};

//--------------------LPF_nthOrderCritical---------------------//
class LPF_nthOrderCritical{ // nth-order critical-damped Low-pass filter (all the poles are at the same place)
public:
    float output;

    LPF_nthOrderCritical(float samplingTime, float cutOff_freq_Hz_in, size_t order_in); // cutOff_freq_Hz_in is in "Hz"
    float filter(float input);
    void reset(float input);

private:
    float Ts;
    size_t order;
    float cutOff_freq_Hz; // Hz

    // Layers of 1st-order LPF
    std::vector<LPF> filter_layers;

    // Flag
    bool Flag_Init;
};

//--------------------LPF_vector_nthOrderCritical---------------------//
class LPF_vector_nthOrderCritical{ // Vectorized nth-order critical-damped Low-pass filter (all the poles are at the same place)
public:
    std::vector<float> output;

    LPF_vector_nthOrderCritical(size_t dimension, float samplingTime, float cutOff_freq_Hz_in, size_t order_in); // cutOff_freq_Hz_in is in "Hz"
    std::vector<float> filter(const std::vector<float> &input);
    void reset(const std::vector<float> &input);

private:
    size_t n;
    float Ts;
    size_t order;

    float cutOff_freq_Hz; // Hz

    // Flag
    bool Flag_Init;

    // Layers of vectorized 1st-order LPF
    std::vector<LPF_vector> filter_layers;

    //
    std::vector<float> zeros; // Zero std::vector [0;0;0]
};


//--------------------HPF---------------------//
class HPF{ // High-pass filter
public:
    float output;

    HPF(float samplingTime, float cutOff_freq_Hz_in); // cutOff_freq_Hz_in is in "Hz"
    float filter(float input);
    void reset(float input);

private:
    float Ts;
    float cutOff_freq_Hz; // Hz
    // float alpha_Ts;
    // float One_alpha_Ts;

    // Flag
    bool Flag_Init;

    //
    LPF lpf;
};

//--------------------HPF_vector---------------------//
class HPF_vector{ // Vectorized high-pass filter
public:
    std::vector<float> output;

    HPF_vector(size_t dimension, float samplingTime, float cutOff_freq_Hz_in); // cutOff_freq_Hz_in is in "Hz"
    std::vector<float> filter(const std::vector<float> &input);
    void reset(const std::vector<float> &input);

private:
    size_t n;
    float Ts;
    float cutOff_freq_Hz; // Hz
    // float alpha_Ts;
    // float One_alpha_Ts;

    // Flag
    bool Flag_Init;

    //
    LPF_vector lpf_v;
};

//--------------------HPF_nthOrderCritical---------------------//
class HPF_nthOrderCritical{ // nth-order critical-damped High-pass filter (all the poles are at the same place)
public:
    float output;

    HPF_nthOrderCritical(float samplingTime, float cutOff_freq_Hz_in, size_t order_in); // cutOff_freq_Hz_in is in "Hz"
    float filter(float input);
    void reset(float input);

private:
    float Ts;
    size_t order;
    float cutOff_freq_Hz; // Hz

    // Layers of 1st-order HPF
    std::vector<HPF> filter_layers;

    // Flag
    bool Flag_Init;
};


//--------------------HPF_vector_nthOrderCritical---------------------//
class HPF_vector_nthOrderCritical{ // Vectorized nth-order critical-damped High-pass filter (all the poles are at the same place)
public:
    std::vector<float> output;

    HPF_vector_nthOrderCritical(size_t dimension, float samplingTime, float cutOff_freq_Hz_in, size_t order_in); // cutOff_freq_Hz_in is in "Hz"
    std::vector<float> filter(const std::vector<float> &input);
    void reset(const std::vector<float> &input);

private:
    size_t n;
    float Ts;
    size_t order;

    float cutOff_freq_Hz; // Hz

    // Flag
    bool Flag_Init;

    // Layers of vectorized 1st-order LPF
    std::vector<HPF_vector> filter_layers;

    //
    std::vector<float> zeros; // Zero std::vector [0;0;0]
};

//--------------------HPF_vector_1minusLPF_nthOrderCritical---------------------//
class HPF_vector_1minusLPF_nthOrderCritical{ // Vectorized nth-order critical-damped High-pass filter ( the version of (1 - nth-order LPF), all the poles are at the same place)
public:
    std::vector<float> output;

    HPF_vector_1minusLPF_nthOrderCritical(size_t dimension, float samplingTime, float cutOff_freq_Hz_in, size_t order_in); // cutOff_freq_Hz_in is in "Hz"
    std::vector<float> filter(const std::vector<float> &input);
    void reset(const std::vector<float> &input);

private:
    size_t n;
    float Ts;
    size_t order;

    float cutOff_freq_Hz; // Hz

    // Flag
    bool Flag_Init;

    // Layers of vectorized 1st-order LPF
    std::vector<LPF_vector> filter_layers;

    //
    std::vector<float> zeros; // Zero std::vector [0;0;0]
};


//--------------------Derivative_appr---------------------//
class Derivative_appr{ // Approximated Derivative, cut-off at 10% of sampling frequency
public:
    float output;

    Derivative_appr(float samplingTime);
    float filter(float input);
    void reset(float input);

private:
    float Ts;
    float cutOff_freq_Hz; // Hz

    // Flag
    bool Flag_Init;

    //
    IIR derivative_LPF2;
};

//--------------------Rate-saturation Filter---------------------//
class RateSaturation_Filter{ // Rate-saturation Filter
public:
    float output;
    float error;

    RateSaturation_Filter(float samplingTime, float limit_rate_in); // limit_rate is in the unit of "value/s"
    float filter(float input);
    void reset(float input);

private:
    float Ts;
    float limit_rate;
    float limit_increment;

    // Flag
    bool Flag_Init;
};

//--------------------Moving average Filter---------------------//
class MovingAverage{ // Moving average Filter
public:
    float output;

    MovingAverage(size_t windowSize_in); // windowSize is in the unit of "samples"
    float filter(float input);
    void reset(float input);

private:
    size_t windowSize;
    float windowSize_inv;

    // Flag
    bool Flag_Init;

    //
    std::vector<float> buffer;
    size_t idx_head;
    float buffer_sum;
};

//-----------First-Order Kalman Filter--------//
class FirstOrder_KalmanFilter{ // 1st-order Kalman filter
public:

    // Parameters
    float A;
    float B;
    float C;
    //
    float R;
    float Q;

    // States
    float mu_est;
    float Sigma_est;
    // Kalman gain
    float K;

    FirstOrder_KalmanFilter(float samplingTime, float A_in, float B_in, float C_in, float R_in, float Q_in, bool is_continuousTime); // If is_continuousTime -> continuous time system
    float filter(float u, float z);
    void reset(float z);

private:
    float Ts;


    // Flag
    bool Flag_Init;
};

//--------------Complementary Filter--------------//
class ComplementaryFilter{ // Fusion two signal, sig_high is for high-frequency, sig_low is for low frequency
public:
    float output;

    ComplementaryFilter(float samplingTime, float crossover_freq_Hz_in); // cutOff_freq_Hz_in is in "Hz"
    float filter(float sig_high, float sig_low);
    void reset(float sig_low);

private:
    float Ts;
    float crossover_freq_Hz; // Hz

    // Low-pass filter
    LPF lpf;

    // Flag
    bool Flag_Init;
};

//----------Complementary Filter with Rate Input----------//
class ComplementaryFilter_withRateInput{ // Fusion two signal, sig_rate is the rate signal for high-frequency, sig_low is for low frequency
public:
    float output;

    ComplementaryFilter_withRateInput(float samplingTime, float crossover_freq_Hz_in); // cutOff_freq_Hz_in is in "Hz"
    float filter(float sig_rate, float sig_low);
    void reset(float sig_low);

private:
    float Ts;
    float crossover_freq_Hz; // Hz
    float alpha_inv; // 1/alpha

    // Low-pass filter
    LPF lpf;

    // Flag
    bool Flag_Init;
};


//-----------------Saturation---------------//
class Saturation{ // Saturation
public:

    // States
    float output;
    float delta_out; // (original_out - limited_out)
    //
    bool is_saturated; // Indicate that if the signal is saturated.

    Saturation(float bound_up_in, float bound_low_in); // If is_continuousTime -> continuous time system
    // Set the limitations
    void set_bound(float bound_up_in, float bound_low_in);
    //
    float filter(float input);
    void reset(float input);

private:
    float Ts;

    //
    float bound_up;
    float bound_low;

    // Flag
    bool Flag_Init;
};

//-----------------Saturation_vector---------------//
class Saturation_vector{ // Saturation
public:

    // States
    std::vector<float> output;
    std::vector<float> delta_out; // (original_out - limited_out)
    //
    std::vector<bool> is_saturated; // Indicate that if the signal is saturated.

    Saturation_vector(size_t dimension, float bound_up_in, float bound_low_in); // If is_continuousTime -> continuous time system
    // Set the limitations
    void set_bound(float bound_up_in, float bound_low_in);
    //
    std::vector<float> filter(std::vector<float> input);
    void reset(std::vector<float> input);

private:
    size_t n;

    //
    float bound_up;
    float bound_low;

    // Flag
    bool Flag_Init;
};

#endif
