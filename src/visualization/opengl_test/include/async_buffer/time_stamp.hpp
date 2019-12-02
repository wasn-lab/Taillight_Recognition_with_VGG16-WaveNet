#ifndef TIME_STAMP_H
#define TIME_STAMP_H

// Determine if we are going to preint some debug information to std_out
#define __DEGUG__

/*
This is a class of TIME_STAMP::Time, which is targeted to re-implement
the ros::Time class for removing the dependency on ROS.

Result:
- The TIME_STAMP::Time::now() function is actually
  much more precise than ros::Time::now().

*/

//
#ifdef __DEGUG__
    #include <iostream> //
#endif
#include <chrono>
#include <thread> // for this_thread::sleep_for
#include <cmath>  // For floor()
#include <utility> // For std::swap with c++11

// using std::vector;

enum class TIME_PARAM{
    NOW
};

// Time, similar to ros::Time
//------------------------------------------------------//
namespace TIME_STAMP{
    const long long const_10_9 = 1000000000;
    const long double const_10_neg9 = 0.000000001;
    //
    struct Time{
        // The Time is: ( sec + nsec*10^-9 )
        long long sec;  // <-- This can be negative
        long long nsec; // <-- This one is greater or equals to zero
        // Constructors
        Time():sec(0),nsec(0)
        {}
        Time(TIME_PARAM t_param){
            if (t_param == TIME_PARAM::NOW){ set_now(); }else{ set_zero(); }
        }
        // explicit Time(long long sec_in): sec(sec_in), nsec(0)
        // {}
        Time(long long sec_in, long long nsec_in):sec(sec_in),nsec(nsec_in)
        {
            _correction();
        }
        Time(long double sec_in){
            long double f_sec = std::floor(sec_in);
            sec = (long long)(f_sec);
            nsec = (long long)( (sec_in - f_sec)*(long double)(const_10_9));
            _correction();
        }
        //
        void _correction(){
            if (nsec < 0){
                long long n_nsec = -nsec;
                sec -= (long long)( n_nsec/const_10_9 );
                //
                sec--;
                nsec = const_10_9 - (n_nsec % const_10_9);
            }else if (nsec > const_10_9){ // nsec >= 0
                sec += (long long)( nsec/const_10_9 );
                nsec %= const_10_9;

            }
        }
        //
        long double toSec(){
            return ( (long double)(sec) + (long double)(nsec)*const_10_neg9 );
        }
        long double toMiliSec(){
            return ( toSec()*1000.0 );
        }
        long double toMicroSec(){
            return ( toSec()*1000000.0 );
        }
        // Usage: Time time_A = Time::now();
        static Time now(){
            using namespace std::chrono;
            auto tp_n = high_resolution_clock::now();
            auto tp_sec = time_point_cast<seconds>(tp_n);
            Time time_B;
            time_B.sec = tp_sec.time_since_epoch().count();
            time_B.nsec = duration_cast<nanoseconds>(tp_n - tp_sec).count();
            return time_B;
        }
        // Usage: time_A.set_now(); --> time_A becomes the current time.
        void set_now(){
            using namespace std::chrono;
            auto tp_n = high_resolution_clock::now();
            auto tp_sec = time_point_cast<seconds>(tp_n);
            sec = tp_sec.time_since_epoch().count();
            nsec = duration_cast<nanoseconds>(tp_n - tp_sec).count();
        }

        // Utilities
        void set_zero(){    sec = 0;  nsec = 0;     }
        void show(){ std::cout << "Time = (" << sec << ", " << nsec << ")\n"; }
        void show_sec(){ std::cout << "Time = " << toSec() << " sec.\n"; }
        void show_msec(){ std::cout << "Time = " << toMiliSec() << " msec.\n"; }
        void show_usec(){ std::cout << "Time = " << toMicroSec() << " usec.\n"; }
        //
        void sleep(){
            // std::cout << "Sleep started.\n";
            std::this_thread::sleep_for( std::chrono::seconds( sec ) + std::chrono::nanoseconds( nsec ) );
            // std::cout << "Sleep ended.\n";
        }
        //

        // Comparison
        bool is_zero() const {
            return ( (sec == 0) && (nsec == 0) );
        }
        bool is_negative() const {
            return (sec < 0);
        }
        bool equal(const Time &time_B) const {
            return ( (sec == time_B.sec) && (nsec == time_B.nsec) );
        }
        bool greater_than(const Time &time_B) const {
            if (sec != time_B.sec)
                return (sec > time_B.sec);
            else // ==
                return (nsec > time_B.nsec);
        }
        bool greater_or_equal(const Time &time_B) const {
            if (sec != time_B.sec)
                return (sec > time_B.sec);
            else // ==
                return (nsec >= time_B.nsec);
        }
        void swap(Time & time_B){
            std::swap(this->sec, time_B.sec);
            std::swap(this->nsec, time_B.nsec);
        }
        //

        // Math
        Time add(const Time &time_B) const {
            return Time(sec+time_B.sec, nsec+time_B.nsec);
        }
        Time minus(const Time &time_B) const {
            return Time(sec-time_B.sec, nsec-time_B.nsec);
        }
        void increase(const Time &time_B){
            sec += time_B.sec; nsec += time_B.nsec;
            _correction();
        }
        void decrease(const Time &time_B){
            sec -= time_B.sec; nsec -= time_B.nsec;
            _correction();
        }
        Time abs() const {
            if (sec < 0){
                return Time(-sec, -nsec);
            }
            return *this;
        }
        //




        // Operators
        bool operator ==(Time const& time_B){
            return equal(time_B);
        }
        bool operator !=(Time const& time_B) const {
            return !equal(time_B);
        }
        bool operator >(Time const& time_B) const {
            return greater_than(time_B);
        }
        bool operator >=(Time const& time_B) const {
            return greater_or_equal(time_B);
        }
        bool operator <(Time const& time_B) const {
            return !greater_or_equal(time_B);
        }
        bool operator <=(Time const& time_B) const {
            return !greater_than(time_B);
        }
        Time operator +(Time const& time_B) const {
            return add(time_B);
        }
        Time operator -(Time const& time_B) const {
            return minus(time_B);
        }
        Time& operator +=(const Time & time_B){
            increase(time_B);
            return *this;
        }
        Time& operator -=(const Time & time_B){
            decrease(time_B);
            return *this;
        }

    }; // end struct Time


    class Period{
    public:

        Time stamped_t;
        Time duration;
        Time last_duration;
        Time jitter;
        double jitter_us_avg;
        //
        std::string name;
        long long seq;

        Period(): stamped_t(TIME_PARAM::NOW), duration(), seq(0), jitter_us_avg(0.0)
        {
        }
        Period(std::string name_in): stamped_t(TIME_PARAM::NOW), duration(), name(name_in), seq(0), jitter_us_avg(0.0)
        {
        }

        void set_name(std::string name_in){ name = name_in; }

        Time stamp(){
            Time _current(TIME_PARAM::NOW);
            Time new_duration = _current - stamped_t;
            // duration
            last_duration.swap(duration);
            duration.swap(new_duration);
            // jitter
            jitter = (duration - last_duration);
            jitter_us_avg += 0.1*( jitter.abs().toMicroSec() - jitter_us_avg);
            // stamped_t
            stamped_t.swap(_current);
            // seq
            seq++;
            return duration;
        }
        void show(){        std::cout << "Period [" << name << " seq:" << seq << "] "; duration.show();    }
        void show_sec(){    std::cout << "Period [" << name << " seq:" << seq << "] "; duration.show_sec();    }
        void show_msec(){   std::cout << "Period [" << name << " seq:" << seq << "] "; duration.show_msec();    }
        void show_usec(){   std::cout << "Period [" << name << " seq:" << seq << "] "; duration.show_usec();    }
        //
        void show_jitter_usec(){   std::cout << "Period [" << name << " seq:" << seq << "] Jitter (avg.) = " << jitter_us_avg << "usec.\n";    }

    private:
    }; // end class Period

    class FPS{
    public:
        Period period;
        Time   stamp_next_th;
        double T_raw; // The latest period
        double T_filtered; // filtered
        double T_filtered_2; // 2nd filtered
        double fps; // 1.0/T_filtered
        //
        double a_Ts;
        //
        long long seq;
        //
        std::string name;

        FPS(): seq(0), stamp_next_th(Time::now().toSec() + 20.0f)
        {
            init();
        }
        FPS(std::string name_in): seq(0), stamp_next_th(Time::now().toSec() + 20.0f), name(name_in)
        {
            init();
        }
        //
        void init(){
            a_Ts = 0.1; // 0.1;
            T_filtered = 1000000.0;
            //
            T_filtered_2 = T_filtered;
            //
            fps = 0.0;
        }
        //
        void set_name(std::string name_in){ name = name_in; }

        double stamp(){
            // Note: This is a legacy method for backward compatibility
            //       Please use update() for new applications.
            period.stamp();
            // Filter, 1st order
            T_raw = period.duration.toSec();
            T_filtered += a_Ts*(T_raw - T_filtered);

            // fps
            fps = 1.0/T_filtered;
            // seq
            seq++;
            return fps;
        }

        double update(bool is_updated){
            if (is_updated){
                period.stamp();
                T_raw = period.duration.toSec();
                // seq
                seq++;
                // Predict next stamp
                stamp_next_th = period.stamped_t + Time(T_raw*1.2);
            }else{
                Time _current(TIME_PARAM::NOW);
                if (_current > stamp_next_th){
                    // Just use the duration from the last update to now
                    T_raw = (_current - period.stamped_t).toSec();
                }else{
                    // No update
                    return fps;
                }
            }
            // Filter, 1st order
            T_filtered += a_Ts*(T_raw - T_filtered);
            T_filtered_2 += a_Ts*(T_filtered - T_filtered_2);
            // fps
            // fps = 1.0/T_filtered;
            fps = 1.0/T_filtered_2;
            return fps;
        }

        void show(){        std::cout << "FPS [" << name << " seq:" << seq << "] = " << fps << "\n";    }

    private:
    };
} // end namespace TIME_STAMP
//------------------------------------------------------//




#endif // TIME_STAMP_H
