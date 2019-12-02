#ifndef ASYNC_BUFFER_H
#define ASYNC_BUFFER_H

// Determine if we are going to preint some debug information to std_out
// #define __DEGUG__

/*
This is a class that implement a single-producer-single-consumer (SPSC) queue.
The implementation of this queue is based on a circular buffer using fixed-length array.

*/

//
#ifdef __DEGUG__
    #include <iostream> //
#endif
#include <memory> // <-- this is for std::shared_ptr
#include <vector>
#include <utility> // std::pair, std::make_pair
#include <mutex>
#include <chrono>
#include <cmath>  // For floor()

// using std::vector;



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
        // Comparison
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
        //
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

    };
}
//------------------------------------------------------//







// The async_buffer class
//------------------------------------------------------//
template <class _T>
class async_buffer{
public:



    // Methods
    async_buffer(size_t buffer_length_in);
    async_buffer(size_t buffer_length_in, _T place_holder_element);



    // Asign _copy_func
    //------------------------------------------------------------------//
    // Important: The following function is important for opencv "Mat"
    //            and other class that use costomized copy function
    //------------------------------------------------------------------//
    /*
    Using the syntex to attach the costomized costomized copy_func:
    async_buffer::assign_copy_func(&copy_func);

    The costomized copy_func should be defined as the following:
    -----------------------------------------------
    For the case of _T being "std::string",
    bool copy_func(string & t, const string & s){
        t = s;
        return true;
    }
    -----------------------------------------------
    For the case of _T being opencv "Mat",
    bool copy_func(Mat & t, const Mat & s){
        t = s.clone();
        // or s.copyTo(t);
        return true;
    }
    -----------------------------------------------
    */
    //------------------------------------------------------------------//
    bool assign_copy_func(bool (*copy_func_in)(_T & _target, const _T & _source)){
        _copy_func = copy_func_in;
    }
    //------------------------------------------------------------------//




    // Queue operations
    bool                    put(const _T & element_in, bool is_droping=true);  // Copy the data in, slow
    std::pair<_T, bool>     front(bool is_poping=false);  // Copy the data out, slow
    bool                    front(_T & content_out, bool is_poping=false);  // Copy the data out, slow
    bool                    pop();    // Only move the index, fast

    // Advanced method
    // std::pair<_T, bool>     pop_front();                // Copy the data out, slow


    // Status of the queue
    // The following method has suttle mutex setting that may effect the result
    bool is_empty(){return _is_empty();} // This is a fast method which may return true even when there are some elements in queue (but not vise versa)
    bool is_full(){return _is_full();}  // // This is a fast method which may return true even when there are some sapces in queue (but not vise versa)
    int size_est(){return _size_est();} // The number of buffered contents. This is a fast method with no mutex but may be incorrect in both direction.
    size_t size_exact(){return _size_exact();} // The number of buffered contents. This is a blocking method with mutex, which will return a correct value.


private:
    // Parameters
    int _dl_len;

    // The container
    std::vector<double> _stamp_list; // time stamp (sec.) of each element
    std::vector<_T> _data_list;


    // The indicators
    int _idx_write;
    int _idx_read;

    // Auxiliary container
    _T _empty_element;
    _T _tmp_output;

    // Function pointer for _copy_func
    bool (*_copy_func)(_T & _target, const _T & _source);

    // mutex locks
    /*
    std::mutex * _mlock_idx_write;
    std::mutex * _mlock_idx_read;
    */
    std::shared_ptr<std::mutex> _mlock_idx_write;
    std::shared_ptr<std::mutex> _mlock_idx_read;
    std::shared_ptr<std::mutex> _mlock_write_block;
    std::shared_ptr<std::mutex> _mlock_read_block;
    //


    // Private methods
    inline void _set_index_write(int idx_write_new){
        {
            std::lock_guard<std::mutex> _lock(*_mlock_idx_write);
            _idx_write = idx_write_new;
        }
    }
    inline void _set_index_read(int idx_read_new){
        {
            std::lock_guard<std::mutex> _lock(*_mlock_idx_read);
            _idx_read = idx_read_new;
        }
    }


    // The default copy function
    // Note: if _T is the opencv Mat,
    //       you should attach acopy function using Mat::clone() or Mat.copyTo()

    // Note: static members are belong to class itself not the object
    static bool _default_copy_func(_T & _target, const _T & _source){
        _target = _source;
        return true;
    }
    //


    // utilities
    inline int _increase_idx(int idx_in){
        // Calculate the increased index, not setting the index
        return _correcting_idx(idx_in+1);
    }
    inline int _correcting_idx(int idx_in){
        // The following equation is to correct the behavior of negative value
        // -7 % 3 = -1 --> -7 mod 3 = 2
        return ( ( _dl_len + (idx_in % _dl_len) ) % _dl_len );
        // return (idx_in % _dl_len);
    }
    inline int _cal_size(int _idx_write_in, int _idx_read_in){
        // Calculate the number of buffered elements according to the indexes given.
        return _correcting_idx(_idx_write_in - _idx_read_in);
    }
    inline bool _is_empty_cal(int _idx_write_in, int _idx_read_in){
        // Determine if the given indexes indicate that the buffer is full
        return ( _cal_size(_idx_write_in, _idx_read_in) == 0 );
    }
    inline bool _is_full_cal(int _idx_write_in, int _idx_read_in){
        // Determine if the given indexes indicate that the buffer is full
        return ( _cal_size(_idx_write_in, _idx_read_in) == (_dl_len-1) );
    }

    // Status of the queue
    // The following method has suttle mutex setting that may effect the result
    bool _is_empty(); // This is a fast method which may return true even when there are some elements in queue (but not vise versa)
    bool _is_full();  // // This is a fast method which may return true even when there are some sapces in queue (but not vise versa)
    int _size_est(); // The number of buffered contents. This is a fast method with no mutex but may be incorrect in both direction.
    size_t _size_exact(); // The number of buffered contents. This is a blocking method with mutex, which will return a correct value.



};
//------------------------------------------------------//



//=====================================================================//

template <class _T>
async_buffer<_T>::async_buffer(size_t buffer_length_in):
    _mlock_idx_write(new std::mutex()),
    _mlock_idx_read(new std::mutex()),
    _mlock_write_block(new std::mutex()),
    _mlock_read_block(new std::mutex()),
    //
    _copy_func(&_default_copy_func)
{
    // The size is equals to mod(_idx_write - _idx_read, _dl_len)
    // If two indexes are identical, it means that the buffer is empty.
    // Note: we should prevent the case that the buffer is overwitten to "empty" when the buffer is full.
    _idx_read = 0;
    _idx_write = 0;

    // Note: buffer_length_in should be at least "1", or the queue will not store any thing.
    // However, specifying "0" will not cause any error, thus remove the following correcting term.
    /*
    if (buffer_length_in < 1)
        buffer_length_in = 1;
    */
    _dl_len = buffer_length_in + 1; // The _data_list should always contains an place wich is ready to be written, hence it will be larger than the buffer length.
    _stamp_list.resize(_dl_len);
    _data_list.resize(_dl_len);

}
template <class _T>
async_buffer<_T>::async_buffer(size_t buffer_length_in, _T place_holder_element):
    _mlock_idx_write(new std::mutex()),
    _mlock_idx_read(new std::mutex()),
    _mlock_write_block(new std::mutex()),
    _mlock_read_block(new std::mutex()),
    //
    _copy_func(&_default_copy_func),
    //
    _empty_element(place_holder_element),
    _tmp_output(place_holder_element)
{
    //------------------------------------------------//
    // A place_holder_element is supported at input
    // incase that the element does not has empty constructor.
    //------------------------------------------------//

    // The size is equals to mod(_idx_write - _idx_read, _dl_len)
    // If two indexes are identical, it means that the buffer is empty.
    // Note: we should prevent the case that the buffer is overwitten to "empty" when the buffer is full.
    _idx_read = 0;
    _idx_write = 0;

    // Note: buffer_length_in should be at least "1", or the queue will not store any thing.
    // However, specifying "0" will not cause any error
    _dl_len = buffer_length_in + 1; // The _data_list should always contains an place wich is ready to be written, hence it will be larger than the buffer length.
    _stamp_list.resize(_dl_len);
    _data_list.resize(_dl_len, place_holder_element);

}


//
template <class _T> bool async_buffer<_T>::put(const _T & element_in, bool is_droping){

    // To lock the write for ensuring only one producer a time
    //-------------------------------------------------------//
    std::lock_guard<std::mutex> _lock(*_mlock_write_block);
    //-------------------------------------------------------//


    // To put an element into the buffer
    bool _all_is_well = true;
    if (_is_full()){
        //
        if (is_droping){
            // Keep dropping until the buffer is not full
            while(_is_full()){
                if (!pop())
                    return false;
            }
            _all_is_well = false;
        }else{
            // No dropping, cancel the put.
            return false;
        }
        //
    }
    // else
    int _idx_write_tmp;
    {
        std::lock_guard<std::mutex> _lock(*_mlock_idx_write);
        _idx_write_tmp = _idx_write;
    }

    // Note: the copy method may not sussess if _T is "Mat" from opencv
    //       be sure to use IMG.clone() mwthod for putting an image in.
    // The following operation might be time consumming
    // _data_list[_idx_write_tmp] = element_in;
    _copy_func(_data_list[_idx_write_tmp], element_in);

    // Note: the following function already got a lock,
    // don't use the same lock recursively
    _set_index_write( _increase_idx(_idx_write_tmp) );
    return _all_is_well;
}

template <class _T> std::pair<_T, bool> async_buffer<_T>::front(bool is_poping){
    // To get an element from the buffer
    // Return false if it's empty

    // To lock the read for ensuring only one consumer a time
    //-------------------------------------------------------//
    std::lock_guard<std::mutex> _lock(*_mlock_read_block);
    //-------------------------------------------------------//

    // If the buffer is empty, we
    if (_is_empty()){
        return std::pair<_T, bool>(this->_empty_element, false);
    }
    // else
    int _idx_read_tmp;
    {
        std::lock_guard<std::mutex> _lock(*_mlock_idx_read);
        _idx_read_tmp = _idx_read;
    }

    // pop?
    if(!is_poping){
        // test
        // _copy_func(_tmp_output, _data_list[_idx_read_tmp]);

        // Note: the copy method may not sussess if _T is "Mat" from opencv
        //       be sure to use IMG.clone() mwthod outside this function.
        // The following operation might be time consumming
        return std::pair<_T, bool>(_data_list[_idx_read_tmp], true);

        // test
        // return std::pair<_T, bool>(_tmp_output, true);
    }else{
        // We need to copy the data first before we move the index (delete)
        // Note: if _T is opencv Mat, the following operation won't really copy the data
        _copy_func(_tmp_output, _data_list[_idx_read_tmp]);

        // Note: the following function already got a lock,
        // don't use the same lock recursively
        _set_index_read( _increase_idx(_idx_read_tmp) );

        // Note: the copy method may not sussess if _T is "Mat" from opencv
        //       be sure to use IMG.clone() mwthod outside this function.
        // The following operation might be time consumming
        return std::pair<_T, bool>(_tmp_output, true);
    }
    //
}
template <class _T> bool async_buffer<_T>::front(_T & content_out, bool is_poping){
    // To get an element from the buffer
    // Return false if it's empty

    // To lock the read for ensuring only one consumer a time
    //-------------------------------------------------------//
    std::lock_guard<std::mutex> _lock(*_mlock_read_block);
    //-------------------------------------------------------//

    // If the buffer is empty, we
    if (_is_empty()){
        return false;
    }
    // else
    int _idx_read_tmp;
    {
        std::lock_guard<std::mutex> _lock(*_mlock_idx_read);
        _idx_read_tmp = _idx_read;
    }

    // pop?
    if(!is_poping){
        _copy_func(content_out, _data_list[_idx_read_tmp]);
        return true;
    }else{
        // We need to copy the data first before we move the index (delete)
        // Note: if _T is opencv Mat, the following operation won't really copy the data

        _copy_func(content_out, _data_list[_idx_read_tmp]);
        // content_out = std::move(_data_list[_idx_read_tmp]); // The content in the buffer will disappear.

        // Note: the following function already got a lock,
        // don't use the same lock recursively
        _set_index_read( _increase_idx(_idx_read_tmp) );

        // Note: the copy method may not sussess if _T is "Mat" from opencv
        //       be sure to use IMG.clone() mwthod outside this function.
        // The following operation might be time consumming
        return true;
    }
    //
}
template <class _T> bool async_buffer<_T>::pop(){
    // To remove an element from the buffer
    // Return false if it's empty

    // To lock the read for ensuring only one consumer a time
    //-------------------------------------------------------//
    std::lock_guard<std::mutex> _lock(*_mlock_read_block);
    //-------------------------------------------------------//


    if (_is_empty()){
        return false;
    }
    // else
    int _idx_read_tmp;
    {
        std::lock_guard<std::mutex> _lock(*_mlock_idx_read);
        _idx_read_tmp = _idx_read;
    }

    // Note: the following function already got a lock,
    // don't use the same lock recursively
    _set_index_read( _increase_idx(_idx_read_tmp) );
    return true;
}


// Advanced methods
/*
template <class _T> std::pair<_T, bool> async_buffer<_T>::pop_front(){
    // To get an element from the buffer and remove it

    // Note: There exist a risk that the _tmp_output may be overwriten by another read function
    //       However, since we only consider the case of "singl consumer", this should be safe.


    // To lock the read for ensuring only one consumer a time
    //-------------------------------------------------------//
    std::lock_guard<std::mutex> _lock(*_mlock_read_block);
    //-------------------------------------------------------//

    // If the buffer is empty, we
    if (_is_empty()){
        return std::pair<_T, bool>(this->_empty_element, false);
    }
    // else
    int _idx_read_tmp;
    {
        std::lock_guard<std::mutex> _lock(*_mlock_idx_read);
        _idx_read_tmp = _idx_read;
    }

    // We need to copy the data first before we move the index (delete)
    // Note: if _T is opencv Mat, the following operation won't really copy the data
    _copy_func(_tmp_output, _data_list[_idx_read_tmp]);

    // Note: the following function already got a lock,
    // don't use the same lock recursively
    _set_index_read( _increase_idx(_idx_read_tmp) );

    // Note: the copy method may not sussess if _T is "Mat" from opencv
    //       be sure to use IMG.clone() mwthod outside this function.
    // The following operation might be time consumming
    return std::pair<_T, bool>(_tmp_output, true);
}
*/




//
template <class _T> bool async_buffer<_T>::_is_empty(){
    // Note: This method is used by "consumer"
    // This is a fast method which may return true even when there are some elements in queue (but not vise versa)

    // Cache the "write" first
    int _idx_write_tmp, _idx_read_tmp;
    {
        std::lock_guard<std::mutex> _lock_w(*_mlock_idx_write);
        _idx_write_tmp = _idx_write;
    }
    {
        std::lock_guard<std::mutex> _lock_r(*_mlock_idx_read);
        _idx_read_tmp = _idx_read;
    }

    return _is_empty_cal(_idx_write_tmp, _idx_read_tmp);
}

template <class _T> bool async_buffer<_T>::_is_full(){
    // Note: This method is used by "producer"
    // // This is a fast method which may return true even when there are some sapces in queue (but not vise versa)

    // Cache the "read" first
    int _idx_write_tmp, _idx_read_tmp;
    {
        std::lock_guard<std::mutex> _lock_r(*_mlock_idx_read);
        _idx_read_tmp = _idx_read;
    }
    {
        std::lock_guard<std::mutex> _lock_w(*_mlock_idx_write);
        _idx_write_tmp = _idx_write;
    }


    return _is_full_cal(_idx_write_tmp, _idx_read_tmp);
}

template <class _T> int async_buffer<_T>::_size_est(){
    // This method may be used by both producer and consumer
    // The number of buffered contents. This is a fast method with no mutex but may be incorrect in both direction.

    // Cache the "read" first, since the write might change more frequently
    int _idx_write_tmp, _idx_read_tmp;
    {
        std::lock_guard<std::mutex> _lock_r(*_mlock_idx_read);
        _idx_read_tmp = _idx_read;
    }
    {
        std::lock_guard<std::mutex> _lock_w(*_mlock_idx_write);
        _idx_write_tmp = _idx_write;
    }
    //
    #ifdef __DEGUG__
        std::cout << "(_idx_write_tmp, _idx_read_tmp) = (" << _idx_write_tmp << ", " << _idx_read_tmp << ") ";
    #endif
    //
    return _cal_size(_idx_write_tmp, _idx_read_tmp);
}

template <class _T> size_t async_buffer<_T>::_size_exact(){
    // This method may be used by both producer and consumer
    // The number of buffered contents. This is a blocking method with mutex, which will return a correct value.

    // Cache both index at the same time and lock all the way to the end
    std::lock_guard<std::mutex> _lock_w(*_mlock_idx_write);
    std::lock_guard<std::mutex> _lock_r(*_mlock_idx_read);
    int _idx_read_tmp = _idx_read;
    int _idx_write_tmp = _idx_write;

    return size_t( _cal_size(_idx_write_tmp, _idx_read_tmp) );
}

#endif
