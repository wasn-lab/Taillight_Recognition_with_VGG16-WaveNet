#include "FILTER_LIB.h"

//--------------------LPF---------------------//
// Low-pass filter
LPF::LPF(float samplingTime, float cutOff_freq_Hz_in){
    //
    Ts = samplingTime;
    cutOff_freq_Hz = cutOff_freq_Hz_in;

    // alpha_Ts = (2.0*PI)*cutOff_freq_Hz*Ts;
    alpha_Ts = 1.0 - exp(-(2.0*PI)*cutOff_freq_Hz*Ts); // The discrete system's pole

    One_alpha_Ts = 1.0 - alpha_Ts;
    output = 0.0;

    //
    Flag_Init = false;
}
float LPF::filter(float input){
    // Initialization
    if (!Flag_Init){
        reset(input);
        Flag_Init = true;
        return output;
    }
    // output = One_alpha_Ts*output + alpha_Ts*input;
    output += alpha_Ts*(input - output);
    return output;
}
void LPF::reset(float input){
    // output = (1.0 - alpha_Ts)*output + alpha_Ts*input;
    output = input;
    return;
}

//

//--------------------LPF_vector---------------------//
// Vectorized low-pass filter
LPF_vector::LPF_vector(size_t dimension, float samplingTime, float cutOff_freq_Hz_in){
    //
    n = dimension;
    Ts = samplingTime;
    cutOff_freq_Hz = cutOff_freq_Hz_in;

    // alpha_Ts = (2.0*PI)*cutOff_freq_Hz*Ts;
    alpha_Ts = 1.0 - exp(-(2.0*PI)*cutOff_freq_Hz*Ts); // The discrete system's pole

    One_alpha_Ts = 1.0 - alpha_Ts;
    //
    zeros.assign(n, 0.0);
    //
    output = zeros;
    //
    Flag_Init = false;
}
std::vector<float> LPF_vector::filter(const std::vector<float> &input){
    // Initialization
    if (!Flag_Init){
        reset(input);
        Flag_Init = true;
        return output;
    }
    /*
    //
    static std::vector<float>::iterator it_output;
    static std::vector<float>::iterator it_input;
    //
    it_output = output.begin();
    it_input = std::vector<float>(input).begin();
    for (size_t i = 0; i < n; ++i){
        // output = One_alpha_Ts*output + alpha_Ts*input;
        *it_output += alpha_Ts*(*it_input - *it_output);
        //
        it_output++;
        it_input++;
    }
    //
    */

    for (size_t i = 0; i < n; ++i){
        // output = One_alpha_Ts*output + alpha_Ts*input;
        output[i] += alpha_Ts*(input[i] - output[i]);
    }

    return output;
}
void LPF_vector::reset(const std::vector<float> &input){
    // output = (1.0 - alpha_Ts)*output + alpha_Ts*input;
    output = input;
    return;
}

//--------------------LPF_nthOrderCritical---------------------//
// nth-order critical-damped Low-pass filter (all the poles are at the same place)
LPF_nthOrderCritical::LPF_nthOrderCritical(float samplingTime, float cutOff_freq_Hz_in, size_t order_in):
        filter_layers(order_in, LPF(samplingTime,cutOff_freq_Hz_in))
{
    //
    Ts = samplingTime;
    order = order_in;
    cutOff_freq_Hz = cutOff_freq_Hz_in;

    output = 0.0;

    //
    Flag_Init = false;
}
float LPF_nthOrderCritical::filter(float input){
    // Initialization
    if (!Flag_Init){
        reset(input);
        Flag_Init = true;
        return output;
    }

    // 0th-order, simply by-pass the input
    if (order == 0){
        output = input;
        return output;
    }

    // The first layer, i = 0
    filter_layers[0].filter(input);
    // The rest, i = 1 ~ n-1
    for (size_t i = 1; i < order; ++i){
        filter_layers[i].filter(filter_layers[i-1].output);
    }

    // Output the last layer's output
    output = filter_layers[order-1].output;
    return output;
}
void LPF_nthOrderCritical::reset(float input){
    // Reset all layers
    for (size_t i = 0; i < order; ++i){
        filter_layers[i].reset(input);
    }
    //
    output = input;
    return;
}

//--------------------LPF_vector_nthOrderCritical---------------------//
// Vectorized nth-order critical-damped Low-pass filter (all the poles are at the same place)
LPF_vector_nthOrderCritical::LPF_vector_nthOrderCritical(size_t dimension, float samplingTime, float cutOff_freq_Hz_in, size_t order_in):
        filter_layers(order_in, LPF_vector(dimension, samplingTime, cutOff_freq_Hz_in))
{
    //
    n = dimension;
    Ts = samplingTime;
    order = order_in;

    cutOff_freq_Hz = cutOff_freq_Hz_in;

    //
    zeros.assign(n, 0.0);
    //
    output = zeros;
    //
    Flag_Init = false;
}
std::vector<float> LPF_vector_nthOrderCritical::filter(const std::vector<float> &input){
    // Initialization
    if (!Flag_Init){
        reset(input);
        Flag_Init = true;
        return output;
    }

    // 0th-order, simply by-pass the input
    if (order == 0){
        output = input;
        return output;
    }

    // The first layer, i = 0
    filter_layers[0].filter(input);
    // The rest, i = 1 ~ n-1
    for (size_t i = 1; i < order; ++i){
        filter_layers[i].filter(filter_layers[i-1].output);
    }

    // Output the last layer's output
    output = filter_layers[order-1].output;
    return output;
}
void LPF_vector_nthOrderCritical::reset(const std::vector<float> &input){
    // Reset all layers
    for (size_t i = 0; i < order; ++i){
        filter_layers[i].reset(input);
    }
    //
    output = input;
    return;
}


//--------------------HPF---------------------//
// High-pass filter
HPF::HPF(float samplingTime, float cutOff_freq_Hz_in):
            lpf(samplingTime,cutOff_freq_Hz_in)
{
    //
    Ts = samplingTime;
    cutOff_freq_Hz = cutOff_freq_Hz_in;
    // alpha_Ts = (2.0*PI)*cutOff_freq_Hz*Ts;
    // One_alpha_Ts = 1.0 - alpha_Ts;
    output = 0.0;

    //
    Flag_Init = false;
}
float HPF::filter(float input){
    // Initialization
    if (!Flag_Init){
        reset(input);
        Flag_Init = true;
        return output; // output = 0.0
    }

    output = input - lpf.filter(input); // hpf = (1 - lpf)*x
    return output;
}
void HPF::reset(float input){
    lpf.reset(input);
    output = 0.0;
    return;
}
//--------------------HPF_vector---------------------//
// Vectorized how-pass filter
HPF_vector::HPF_vector(size_t dimension, float samplingTime, float cutOff_freq_Hz_in):
            lpf_v(dimension,samplingTime,cutOff_freq_Hz_in)
{
    //
    n = dimension;
    Ts = samplingTime;
    cutOff_freq_Hz = cutOff_freq_Hz_in;
    // alpha_Ts = (2.0*PI)*cutOff_freq_Hz*Ts;
    // One_alpha_Ts = 1.0 - alpha_Ts;
    output.assign(n,0.0);

    //
    Flag_Init = false;
}
std::vector<float> HPF_vector::filter(const std::vector<float> &input){
    // Initialization
    if (!Flag_Init){
        reset(input);
        Flag_Init = true;
        return output; // output = zeros_n
    }
    //
    lpf_v.filter(input);
    //
    /*
    // hpf = (1 - lpf)*x
    for (size_t i = 0; i < n; ++i){
        output[i] = input[i] - lpf_v.output[i];
    }
    */

    // hpf = (1 - lpf)*x
    static std::vector<float>::iterator it_output;
    static std::vector<float>::iterator it_input;
    static std::vector<float>::iterator it_lpf_v;
    //
    it_output = output.begin();
    it_input = std::vector<float>(input).begin();
    it_lpf_v = lpf_v.output.begin();
    for (size_t i = 0; i < n; ++i){
        *it_output = *it_input - *it_lpf_v;
        //
        it_output++;
        it_input++;
        it_lpf_v++;
    }
    return output;
}
void HPF_vector::reset(const std::vector<float> &input){
    lpf_v.reset(input);
    output.assign(n,0.0);
    return;
}

//--------------------HPF_nthOrderCritical---------------------//
// nth-order critical-damped High-pass filter (all the poles are at the same place)
HPF_nthOrderCritical::HPF_nthOrderCritical(float samplingTime, float cutOff_freq_Hz_in, size_t order_in):
        filter_layers(order_in, HPF(samplingTime,cutOff_freq_Hz_in))
{
    //
    Ts = samplingTime;
    order = order_in;
    cutOff_freq_Hz = cutOff_freq_Hz_in;

    output = 0.0;

    //
    Flag_Init = false;
}
float HPF_nthOrderCritical::filter(float input){
    // Initialization
    if (!Flag_Init){
        reset(input);
        Flag_Init = true;
        return output;
    }

    // 0th-order, simply by-pass the input
    if (order == 0){
        output = input;
        return output;
    }

    // The first layer, i = 0
    filter_layers[0].filter(input);
    // The rest, i = 1 ~ n-1
    for (size_t i = 1; i < order; ++i){
        filter_layers[i].filter(filter_layers[i-1].output);
    }

    // Output the last layer's output
    output = filter_layers[order-1].output;
    return output;
}
void HPF_nthOrderCritical::reset(float input){
    // Reset all layers
    for (size_t i = 0; i < order; ++i){
        filter_layers[i].reset(input);
    }
    //
    output = 0.0;
    return;
}

//--------------------HPF_vector_nthOrderCritical---------------------//
// Vectorized nth-order critical-damped High-pass filter (all the poles are at the same place)
HPF_vector_nthOrderCritical::HPF_vector_nthOrderCritical(size_t dimension, float samplingTime, float cutOff_freq_Hz_in, size_t order_in):
        filter_layers(order_in, HPF_vector(dimension, samplingTime, cutOff_freq_Hz_in))
{
    //
    n = dimension;
    Ts = samplingTime;
    order = order_in;

    cutOff_freq_Hz = cutOff_freq_Hz_in;

    //
    zeros.assign(n, 0.0);
    //
    output = zeros;
    //
    Flag_Init = false;
}
std::vector<float> HPF_vector_nthOrderCritical::filter(const std::vector<float> &input){
    // Initialization
    if (!Flag_Init){
        reset(input);
        Flag_Init = true;
        return output;
    }

    // 0th-order, simply by-pass the input
    if (order == 0){
        output = input;
        return output;
    }

    // The first layer, i = 0
    filter_layers[0].filter(input);
    // The rest, i = 1 ~ n-1
    for (size_t i = 1; i < order; ++i){
        filter_layers[i].filter(filter_layers[i-1].output);
    }

    // Output the last layer's output
    output = filter_layers[order-1].output;
    return output;
}
void HPF_vector_nthOrderCritical::reset(const std::vector<float> &input){
    // Reset all layers
    for (size_t i = 0; i < order; ++i){
        filter_layers[i].reset(input);
    }
    //
    output.assign(n,0.0);
    return;
}

//--------------------HPF_vector_1minusLPF_nthOrderCritical---------------------//
// Vectorized nth-order critical-damped High-pass filter ( the version of (1 - nth-order LPF), all the poles are at the same place)
HPF_vector_1minusLPF_nthOrderCritical::HPF_vector_1minusLPF_nthOrderCritical(size_t dimension, float samplingTime, float cutOff_freq_Hz_in, size_t order_in):
        filter_layers(order_in, LPF_vector(dimension, samplingTime, cutOff_freq_Hz_in))
{
    //
    n = dimension;
    Ts = samplingTime;
    order = order_in;

    cutOff_freq_Hz = cutOff_freq_Hz_in;

    //
    zeros.assign(n, 0.0);
    //
    output = zeros;
    //
    Flag_Init = false;
}
std::vector<float> HPF_vector_1minusLPF_nthOrderCritical::filter(const std::vector<float> &input){
    // Initialization
    if (!Flag_Init){
        reset(input);
        Flag_Init = true;
        return output;
    }

    // 0th-order, simply by-pass the input
    if (order == 0){
        output = input;
        return output;
    }

    // The first layer, i = 0
    filter_layers[0].filter(input);
    // The rest, i = 1 ~ n-1
    for (size_t i = 1; i < order; ++i){
        filter_layers[i].filter(filter_layers[i-1].output);
    }

    // Output the last layer's output
    // output = filter_layers[order-1].output;

    // hpf = (1 - lpf)*x
    static std::vector<float>::iterator it_output;
    static std::vector<float>::iterator it_input;
    static std::vector<float>::iterator it_lastFilter;
    //
    it_output = output.begin();
    it_input = std::vector<float>(input).begin();
    it_lastFilter = filter_layers[order-1].output.begin();
    for (size_t i = 0; i < n; ++i){
        *it_output = *it_input - *it_lastFilter;
        //
        it_output++;
        it_input++;
        it_lastFilter++;
    }
    return output;
}
void HPF_vector_1minusLPF_nthOrderCritical::reset(const std::vector<float> &input){
    // Reset all layers
    for (size_t i = 0; i < order; ++i){
        filter_layers[i].reset(input);
    }
    //
    output.assign(n,0.0);
    return;
}




//--------------------Derivative_appr---------------------//
// Approximated Derivative, cut-off at 10% of sampling frequency
Derivative_appr::Derivative_appr(float samplingTime):
        // derivative_LPF2(2,2) // 2nd-order LPF
        // derivative_LPF2(4,4) // 4th-order LPF
        derivative_LPF2(6,6) // 6th-order LPF
{
    //
    Ts = samplingTime;
    cutOff_freq_Hz = 0.1*Ts;

    /*
    // 10% of Fs
    float num[] = {114.296712320901,	0.,	-114.296712320901};
    float den[] = {1.,	-1.04377110555725,	0.272364530199049};
    float gain = 1.0;
    */
    /*
    // 20% of Fs
    float num[] = {297.790611663702,	0.,	-297.790611663702};
    float den[] = {1.,	-0.456521819619850,	0.0521030429472546};
    float gain = 1.0;
    */

    // 20% of Fs, 4th-order LPF
    float num[] = {44.3396241975210,	88.6792483950420,	0.,	-88.6792483950420,	-44.3396241975210};
    float den[] = {1.,	-0.913043639239699,	0.312618257683527,	-0.0475723519480234,	0.00271472708436336};
    float gain = 1.0;


    /*
    // 20% of Fs, 6th-order LPF
    float num[] = {6.60196190535925,	26.4078476214370,	33.0098095267962,	0.,	-33.0098095267962,	-26.4078476214370,	-6.60196190535925};
    float den[] = {1.,	-1.36956545885955,	0.781545644208822,	-0.237861759740118,	0.0407209062654518,	-0.00371799644497471,	0.000141445541866579};
    float gain = 1.0;
    */
    //
    derivative_LPF2.Assign_parameters(num,den,gain);
    //
    output = 0.0;

    //
    Flag_Init = false;
}
float Derivative_appr::filter(float input){
    // Initialization
    if (!Flag_Init){
        reset(input);
        Flag_Init = true;
        return output; // output = 0.0
    }
    //
    output = derivative_LPF2.Iterate_once(input);
    return output;
}
void Derivative_appr::reset(float input){
    //
    /*
    for (size_t i = 0; i < 3; ++i){
        derivative_LPF2.Iterate_once(0.0);
    }
    */
    derivative_LPF2.Reset(input);
    output = 0.0;
    return;
}

//--------------------Rate-saturation Filter---------------------//
// Rate-saturation Filter
//
//          _______
//         /
//        /
//       /
//  ____/
//
RateSaturation_Filter::RateSaturation_Filter(float samplingTime, float limit_rate_in){ // limit_rate is in the unit of "value/s"
    //
    Ts = samplingTime;
    limit_rate = limit_rate_in;
    limit_increment = limit_rate*Ts;
    output = 0.0;

    //
    Flag_Init = false;
}
float RateSaturation_Filter::filter(float input){
    // Initialization
    if (!Flag_Init){
        reset(input);
        Flag_Init = true;
        return output;
    }
    error = input - output;
    if (error > limit_increment){
        error = limit_increment;
    }else if(error < -limit_increment){
        error = -limit_increment;
    }
    output += error;
    return output;
}
void RateSaturation_Filter::reset(float input){
    // output = (1.0 - alpha_Ts)*output + alpha_Ts*input;
    output = input;
    return;
}

//--------------------Moving average Filter---------------------//
// Moving average Filter
//
//         __________
//        |         |
// _______|         |_________
//
//
MovingAverage::MovingAverage(size_t windowSize_in){ // windowSize is in the unit of "samples"
    //
    windowSize = windowSize_in;
    windowSize_inv = 1.0/float(windowSize);
    //
    buffer.assign(windowSize, 0.0);
    idx_head = 0;
    buffer_sum = 0;
    //
    output = 0.0;

    //
    Flag_Init = false;
}
float MovingAverage::filter(float input){
    // Initialization
    if (!Flag_Init){
        reset(input);
        Flag_Init = true;
        return output;
    }
    //
    idx_head++;
    if (idx_head >= windowSize){
        idx_head = 0;
    }
    //
    buffer_sum -= buffer[idx_head]; // Remove the oldest data
    buffer_sum += input;
    buffer[idx_head] = input; // Record the new data
    //
    output = buffer_sum*windowSize_inv;

    return output;
}
void MovingAverage::reset(float input){
    buffer.assign(windowSize, input);
    idx_head = 0;
    buffer_sum = input*windowSize;
    //
    output = input; // output = buffer_sum*windowSize_inv;
    return;
}

//-----------First-Order Kalman Filter--------//
// FirstOrder_KalmanFilter
FirstOrder_KalmanFilter::FirstOrder_KalmanFilter(float samplingTime, float A_in, float B_in, float C_in, float R_in, float Q_in, bool is_continuousTime){ // If is_continuousTime -> continuous time system
    //
    Ts = samplingTime;

    // Parameters
    if(is_continuousTime){
        A = 1.0 + Ts*A_in;
        B = Ts*B_in;
    }else{
        A = A_in;
        B = B_in;
    }
    //
    C = C_in;
    //
    R = R_in;
    Q = Q_in;


    //
    mu_est = 0.0;
    Sigma_est = 100000; // No a-priori knoledge

    // Kalman gain
    K = 0;

    //
    Flag_Init = false;
}
float FirstOrder_KalmanFilter::filter(float u, float z){
    // Initialization
    if (!Flag_Init){
        reset(z);
        Flag_Init = true;
        return mu_est;
    }

    // Prediction
    mu_est  = A*mu_est + B*u;
    Sigma_est = A*Sigma_est*A + R;
    // Update
    K = Sigma_est*C/(C*Sigma_est*C + Q);
    mu_est += K*(z - C*mu_est);
    Sigma_est -= K*C*Sigma_est;
    //

    return mu_est;
}
void FirstOrder_KalmanFilter::reset(float z){
    //
    mu_est = z;
    Sigma_est = 100000; // No a-priori knoledge

    // Kalman gain
    K = 0;
    return;
}

//--------------Complementary Filter--------------//
// Fusion two signal, sig_high is for high-frequency, sig_low is for low frequency
ComplementaryFilter::ComplementaryFilter(float samplingTime, float crossover_freq_Hz_in):
    lpf(samplingTime, crossover_freq_Hz_in)
{
    //
    Ts = samplingTime;
    crossover_freq_Hz = crossover_freq_Hz_in;

    //
    output = 0.0;

    //
    Flag_Init = false;
}
float ComplementaryFilter::filter(float sig_high, float sig_low){
    // Initialization
    if (!Flag_Init){
        reset(sig_low);
        Flag_Init = true;
        return output;
    }
    //
    output = sig_high + lpf.filter(sig_low - sig_high);

    return output;
}
void ComplementaryFilter::reset(float sig_low){
    // output = (1.0 - alpha_Ts)*output + alpha_Ts*input;
    lpf.reset(sig_low);
    output = sig_low;
    return;
}

//----------Complementary Filter with Rate Input----------//
// Fusion two signal, sig_rate is the rate signal for high-frequency, sig_low is for low frequency
ComplementaryFilter_withRateInput::ComplementaryFilter_withRateInput(float samplingTime, float crossover_freq_Hz_in):
    lpf(samplingTime, crossover_freq_Hz_in)
{
    //
    Ts = samplingTime;
    crossover_freq_Hz = crossover_freq_Hz_in;
    alpha_inv = 1.0/((2.0*PI)*crossover_freq_Hz_in); // 1/alpha

    //
    output = 0.0;

    //
    Flag_Init = false;
}
float ComplementaryFilter_withRateInput::filter(float sig_rate, float sig_low){
    // Initialization
    if (!Flag_Init){
        reset(sig_low);
        Flag_Init = true;
        return output;
    }
    //
    output = lpf.filter(sig_low + sig_rate*alpha_inv);

    return output;
}
void ComplementaryFilter_withRateInput::reset(float sig_low){
    // output = (1.0 - alpha_Ts)*output + alpha_Ts*input;
    lpf.reset(sig_low);
    output = sig_low;
    return;
}

//-----------Saturation--------//
// Saturation
Saturation::Saturation(float bound_up_in, float bound_low_in){ // If is_continuousTime -> continuous time system

    //
    bound_up = bound_up_in;
    bound_low = bound_low_in;

    //
    output = 0.0;
    delta_out = 0.0;
    is_saturated = false;

    Flag_Init = false;
}
void Saturation::set_bound(float bound_up_in, float bound_low_in){
    //
    bound_up = bound_up_in;
    bound_low = bound_low_in;
}
float Saturation::filter(float input){
    // Initialization
    if (!Flag_Init){
        reset(input);
        Flag_Init = true;
        return output;
    }

    // Saturation
    if (input > bound_up){
        output = bound_up;
        is_saturated = true;
    }else if (input < bound_low){
        output = bound_low;
        is_saturated = true;
    }else{
        output = input;
        is_saturated = false;
    }
    // Calculating the delta_out
    delta_out = input - output;
    return output;
}
void Saturation::reset(float input){
    //
    // Saturation
    if (input > bound_up){
        output = bound_up;
        is_saturated = true;
    }else if (input < bound_low){
        output = bound_low;
        is_saturated = true;
    }else{
        output = input;
        is_saturated = false;
    }
    // Calculating the delta_out
    delta_out = input - output;
    return;
}

//-----------Saturation_vector--------//
// Saturation_vector
Saturation_vector::Saturation_vector(size_t dimension, float bound_up_in, float bound_low_in){ // If is_continuousTime -> continuous time system
    //
    n = dimension;

    //
    bound_up = bound_up_in;
    bound_low = bound_low_in;

    //
    output.assign(n, 0.0);
    delta_out.assign(n, 0.0);
    is_saturated.assign(n, false);


    Flag_Init = false;
}
void Saturation_vector::set_bound(float bound_up_in, float bound_low_in){
    //
    bound_up = bound_up_in;
    bound_low = bound_low_in;
}
std::vector<float> Saturation_vector::filter(std::vector<float> input){
    // Initialization
    if (!Flag_Init){
        reset(input);
        Flag_Init = true;
        return output;
    }

    // Saturation
    for (size_t i = 0; i < n; ++i){
        //
        if (input[i] > bound_up){
            output[i] = bound_up;
            is_saturated[i] = true;
        }else if (input[i] < bound_low){
            output[i] = bound_low;
            is_saturated[i] = true;
        }else{
            output[i] = input[i];
            is_saturated[i] = false;
        }
        // Calculating the delta
        delta_out[i] = input[i] - output[i];
    }


    return output;
}
void Saturation_vector::reset(std::vector<float> input){
    //
    // Saturation
    for (size_t i = 0; i < n; ++i){
        //
        if (input[i] > bound_up){
            output[i] = bound_up;
            is_saturated[i] = true;
        }else if (input[i] < bound_low){
            output[i] = bound_low;
            is_saturated[i] = true;
        }else{
            output[i] = input[i];
            is_saturated[i] = false;
        }
        // Calculating the delta
        delta_out[i] = input[i] - output[i];
    }

    return;
}
