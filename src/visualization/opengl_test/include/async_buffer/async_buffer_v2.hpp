#ifndef ASYNC_BUFFER_V2_H
#define ASYNC_BUFFER_V2_H

// Determine if we are going to preint some debug information to std_out
#define __DEGUG__

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
#include <time_stamp.hpp> // The TIME_STAMP::Time class
#include <boost/any.hpp> // any type, Note: in c++17, this is also implemented in std::any

// using std::vector;


// The async_buffer base class
//------------------------------------------------------//
class async_buffer_base{
public:
    // Constructors
    async_buffer_base(){}

    // pop
    virtual bool pop() = 0;    // Only move the index, fast

    // boost::any wrappers
    //-----------------------------------------------//
    // Put in the std::shared_ptr as element_in_ptr
    virtual bool put_any(boost::any & element_in_ptr, bool is_droping=true, const TIME_STAMP::Time &stamp_in=TIME_STAMP::Time::now(), bool is_no_copying=true) = 0;  // Exchanging the data, fast
    // Put in the std::shared_ptr as content_out_ptr
    virtual bool front_any(boost::any & content_out_ptr, bool is_poping=false, const TIME_STAMP::Time &stamp_req=TIME_STAMP::Time()) = 0;  // If is_poping, exchanging the data out, fast; if not is_poping, share the content (Note: this may not be safe!!)
    //-----------------------------------------------//

    // void* wrappers
    //-----------------------------------------------//
    // Put in the &(std::shared_ptr<_T>) or &(_T) as element_in_ptr
    virtual bool put_void(const void * element_in_ptr, bool is_droping=true, const TIME_STAMP::Time &stamp_in=TIME_STAMP::Time::now(), bool is_shared_ptr=true) = 0;  // Exchanging the data, fast
    // Put in the &(std::shared_ptr<_T>) or &(_T) as content_out_ptr
    virtual bool front_void(void * content_out_ptr, bool is_poping=false, const TIME_STAMP::Time &stamp_req=TIME_STAMP::Time(), bool is_shared_ptr=true) = 0;  // If is_poping, exchanging the data out, fast; if not is_poping, share the content (Note: this may not be safe!!)
    //-----------------------------------------------//

    // Advanced operations
    //-----------------------------------------------//
    // Time stamp
    virtual TIME_STAMP::Time get_stamp(void) = 0; // Note: use this function right after using any one of the "front" method
    //-----------------------------------------------//
protected:

private:
};
//------------------------------------------------------//


// The async_buffer class
//------------------------------------------------------//
template <class _T>
class async_buffer: public async_buffer_base
{
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
        return true;
    }
    //------------------------------------------------------------------//




    // Basic queue operations
    //-----------------------------------------------//
    // Put, overloading
    bool    put(const _T & element_in, bool is_droping=true, const TIME_STAMP::Time &stamp_in=TIME_STAMP::Time::now());  // Copy the data in, slow
    bool    put(std::shared_ptr<_T> & element_in_ptr, bool is_droping=true, const TIME_STAMP::Time &stamp_in=TIME_STAMP::Time::now());  // Exchanging the data, fast

    // Front, overloading
    bool    front(_T & content_out, bool is_poping=false, const TIME_STAMP::Time &stamp_req=TIME_STAMP::Time() );  // Copy the data out, slow
    bool    front(std::shared_ptr<_T> & content_out_ptr, bool is_poping=false, const TIME_STAMP::Time &stamp_req=TIME_STAMP::Time());  // If is_poping, exchanging the data out, fast; if not is_poping, share the content (Note: this may not be safe!!)
    // pop
    bool    pop();    // Only move the index, fast
    //-----------------------------------------------//

    // Wrappers
    //-----------------------------------------------//
    inline bool    put(const _T & element_in, const TIME_STAMP::Time &stamp_in){
        return put(element_in, true, stamp_in);
    }
    inline bool    put(std::shared_ptr<_T> & element_in_ptr, const TIME_STAMP::Time &stamp_in){
        return put(element_in_ptr, true, stamp_in);
    }
    //-----------------------------------------------//


    // boost::any wrappers
    //-----------------------------------------------//
    // Put in the std::shared_ptr as element_in_ptr
    bool    put_any(boost::any & element_in_ptr, bool is_droping=true, const TIME_STAMP::Time &stamp_in=TIME_STAMP::Time::now(), bool is_no_copying=true);  // Exchanging the data, fast
    // Put in the std::shared_ptr as content_out_ptr
    bool    front_any(boost::any & content_out_ptr, bool is_poping=false, const TIME_STAMP::Time &stamp_req=TIME_STAMP::Time());  // If is_poping, exchanging the data out, fast; if not is_poping, share the content (Note: this may not be safe!!)
    //-----------------------------------------------//

    // void* wrappers
    //-----------------------------------------------//
    // Put in the &(std::shared_ptr<_T>) or &(_T) as element_in_ptr
    bool    put_void(const void * element_in_ptr, bool is_droping=true, const TIME_STAMP::Time &stamp_in=TIME_STAMP::Time::now(), bool is_shared_ptr=true);  // Exchanging the data, fast
    // Put in the &(std::shared_ptr<_T>) or &(_T) as content_out_ptr
    bool    front_void(void * content_out_ptr, bool is_poping=false, const TIME_STAMP::Time &stamp_req=TIME_STAMP::Time(), bool is_shared_ptr=true);  // If is_poping, exchanging the data out, fast; if not is_poping, share the content (Note: this may not be safe!!)
    //-----------------------------------------------//

    // Advanced operations
    //-----------------------------------------------//
    // Time stamp
    TIME_STAMP::Time get_stamp(void){return _stamp_out;} // Note: use this function right after using any one of the "front" method
    //-----------------------------------------------//


    // Status of the queue
    // The following method has suttle mutex setting that may effect the result
    bool is_empty(){return _is_empty();} // This is a fast method which may return true even when there are some elements in queue (but not vise versa)
    bool is_full(){return _is_full();}  // // This is a fast method which may return true even when there are some sapces in queue (but not vise versa)
    int size_est(){return _size_est();} // The number of buffered contents. This is a fast method with no mutex but may be incorrect in both direction.
    size_t size_exact(){return _size_exact();} // The number of buffered contents. This is a blocking method with mutex, which will return a correct value.

    // Utilities
    inline _T get_empty_element(){ return _empty_element;    }

private:
    // Parameters
    int _dl_len;

    // The containers
    std::vector< std::shared_ptr<_T> > _data_ptr_list; //
    std::vector<TIME_STAMP::Time> _stamp_list; // time stamp (sec.) of each element

    // Pointer operations
    inline void _fill_in_ptr_if_empty(std::shared_ptr<_T> & _ptr_in){
        if (!_ptr_in){  _ptr_in.reset( new _T(_empty_element) );  }
    }
    inline bool is_ptr_shared(const std::shared_ptr<_T> & _ptr_in){
        return ( _ptr_in && !_ptr_in.unique() );
    }
    inline void _reset_ptr_if_shared(std::shared_ptr<_T> & _ptr_in){
        if (is_ptr_shared(_ptr_in)){    _ptr_in.reset( new _T(_empty_element) );    }
    }


    // The indicators
    int _idx_write;
    int _idx_read;

    // Auxiliary container
    _T _empty_element;
    std::shared_ptr<_T> _tmp_output_ptr;
    TIME_STAMP::Time _stamp_out;

    // Flags
    bool _got_front_but_no_pop;

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
        std::lock_guard<std::mutex> _lock(*_mlock_idx_write);
        _idx_write = idx_write_new;
    }
    inline int _get_index_write(){
        std::lock_guard<std::mutex> _lock(*_mlock_idx_write);
        return _idx_write;
    }
    inline void _set_index_read(int idx_read_new){
        std::lock_guard<std::mutex> _lock(*_mlock_idx_read);
        _idx_read = idx_read_new;
    }
    inline int _get_index_read(){
        std::lock_guard<std::mutex> _lock(*_mlock_idx_read);
        return _idx_read;
    }


    // Time searching, find the minimum delta_t in absoulute value
    /*
    inline int get_idx_of_closest_stamp(int _idx_read_tmp, const TIME_STAMP::Time &stamp_req){
        int _idx_write_tmp = _get_index_write();
        int _closest_idx = _idx_read_tmp;
        TIME_STAMP::Time _min_delta_t;
        bool _is_first = true;
        for( int _idx_read_search = _idx_read_tmp; (_idx_read_search != _idx_write_tmp) ; _idx_read_search = _increase_idx(_idx_read_search)){
            TIME_STAMP::Time _delta_t = (stamp_req - _stamp_list[_idx_read_search]).abs();
            if (_is_first || _delta_t < _min_delta_t){
            // if (_is_first || (_delta_t < _min_delta_t && stamp_req >= _stamp_list[_idx_read_tmp]) ){
                _is_first = false;
                _min_delta_t = _delta_t;
                _closest_idx = _idx_read_search;
            }
        }
        //
        return _closest_idx;
    }
    */
    inline int get_idx_of_closest_stamp(int _idx_read_tmp, const TIME_STAMP::Time &stamp_req){
        int _idx_write_tmp = _get_index_write();
        int _closest_idx = -1;
        TIME_STAMP::Time _min_delta_t(10.0f);
        for( int _idx_read_search = _idx_read_tmp; (_idx_read_search != _idx_write_tmp) ; _idx_read_search = _increase_idx(_idx_read_search)){
            TIME_STAMP::Time _delta_t = (stamp_req - _stamp_list[_idx_read_search]).abs();
            // if (_is_first || _delta_t < _min_delta_t){
            if ( (_delta_t < _min_delta_t && stamp_req >= _stamp_list[_idx_read_tmp]) ){
                _min_delta_t = _delta_t;
                _closest_idx = _idx_read_search;
            }
        }
        //
        return _closest_idx;
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
        return _correcting_idx(++idx_in);
    }
    inline int _decrease_idx(int idx_in){
        // Calculate the decreased index, not setting the index
        return _correcting_idx(--idx_in);
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
    _copy_func(&_default_copy_func),
    _empty_element(),
    _tmp_output_ptr(nullptr)
{
    // The size is equals to mod(_idx_write - _idx_read, _dl_len)
    // If two indexes are identical, it means that the buffer is empty.
    // Note: we should prevent the case that the buffer is overwitten to "empty" when the buffer is full.
    _idx_read = 0;
    _idx_write = 0;
    _got_front_but_no_pop = false; // Reset the flag for front

    // Note: buffer_length_in should be at least "1", or the queue will not store any thing.
    // However, specifying "0" will not cause any error, thus remove the following correcting term.
    /*
    if (buffer_length_in < 1)
        buffer_length_in = 1;
    */
    _dl_len = buffer_length_in + 1; // The _data_ptr_list should always contains an place wich is ready to be written, hence it will be larger than the buffer length.
    _stamp_list.resize(_dl_len);

    // Note: Any new instance is not working here, this results in several async_buffer in a vector pointing to the same instance set!!
    // This is because the vector only construct the element once and coy it all over the place.
    // To solve this problem, we simply let the pointer be null, and assign it value at runtime
    _data_ptr_list.resize(_dl_len, nullptr); // Note: initialized with all null pointers!!
    #ifdef __DEGUG__
        // test, this should be
        std::cout << "_data_ptr_list.size() == _dl_len? " << (_data_ptr_list.size() == _dl_len) << "\n";
        // test, this should be "0"
        std::cout << "_data_ptr_list[0].use_count() = " << _data_ptr_list[0].use_count() << "\n";
        //
    #endif
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
    _tmp_output_ptr(nullptr)
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
    _got_front_but_no_pop = false; // Reset the flag for front

    // Note: buffer_length_in should be at least "1", or the queue will not store any thing.
    // However, specifying "0" will not cause any error
    _dl_len = buffer_length_in + 1; // The _data_ptr_list should always contains an place wich is ready to be written, hence it will be larger than the buffer length.
    _stamp_list.resize(_dl_len);

    // Note: Any new instance is not working here, this results in several async_buffer in a vector pointing to the same instance set!!
    // This is because the vector only construct the element once and coy it all over the place.
    // To solve this problem, we simply let the pointer be null, and assign it value at runtime
    _data_ptr_list.resize(_dl_len, nullptr); // Note: initialized with all null pointers!!
    #ifdef __DEGUG__
        // test, this should be
        std::cout << "_data_ptr_list.size() == _dl_len? " << (_data_ptr_list.size() == _dl_len) << "\n";
        // test, this should be "0"
        std::cout << "_data_ptr_list[0].use_count() = " << _data_ptr_list[0].use_count() << "\n";
        //
    #endif
}


//
//
template <class _T> bool async_buffer<_T>::put(const _T & element_in, bool is_droping, const TIME_STAMP::Time &stamp_in){

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
    int _idx_write_tmp = _get_index_write();

    // Assign time "now" (can put this line right after getting the index)
    _stamp_list[_idx_write_tmp] = stamp_in;
    //

    // Fill the pointer
    _fill_in_ptr_if_empty(_data_ptr_list[_idx_write_tmp]);
    //

    // Note: the copy method may not sussess if _T is "Mat" from opencv
    //       be sure to use IMG.clone() mwthod for putting an image in.
    _copy_func(*_data_ptr_list[_idx_write_tmp], element_in); // *ptr <-- instance


    // Note: the following function already got a lock,
    // don't use the same lock recursively
    _set_index_write( _increase_idx(_idx_write_tmp) );
    return _all_is_well;
}
template <class _T> bool async_buffer<_T>::put(std::shared_ptr<_T> & element_in_ptr, bool is_droping, const TIME_STAMP::Time &stamp_in){

    // To lock the write for ensuring only one producer a time
    //-------------------------------------------------------//
    std::lock_guard<std::mutex> _lock(*_mlock_write_block);
    //-------------------------------------------------------//



    // To put an element into the buffer
    bool _all_is_well = true;

    // The input is an empty pointer, return immediatly.
    if (!element_in_ptr){
        _all_is_well = false;
        return false;
    }

    // Check if the buffer is full
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
    int _idx_write_tmp = _get_index_write();

    // Assign time "now" (can put this line right after getting the index)
    _stamp_list[_idx_write_tmp] = stamp_in;
    //
    // std::cout << "stamp_in = " << stamp_in.sec << "\n";


    //---------------------------------------------------------//
    // Fill the pointer
    _fill_in_ptr_if_empty(_data_ptr_list[_idx_write_tmp]);
    //

    // Note: the element_in_ptr is shure not to be an empty pointer
    // Pre-check: The input pointer should be pure (unique )
    if (!element_in_ptr.unique() ){ // Not unique
        // Copy element
        // Note: the copy method may not sussess if _T is "Mat" from opencv
        //       be sure to use IMG.clone() mwthod for putting an image in.
        _copy_func(*_data_ptr_list[_idx_write_tmp], *element_in_ptr); // *ptr <-- *ptr
#ifdef __DEGUG__
        std::cout << "[put] input pointer is not pure.\n";
#endif
    }else{
        // The input pointer is pure (unique or null)
        // swapping pointers
        _data_ptr_list[_idx_write_tmp].swap(element_in_ptr);

        // Post-check: the output pointer should be pure (unique or empty)
        if (!element_in_ptr.unique() ){ // Not not unique (empty or shared)
            element_in_ptr.reset(new _T(_empty_element)); // Reset the pointer to make it clean.
#ifdef __DEGUG__
            std::cout << "[put] container pointer is not pure.\n";
#endif
        }
        //
    }
    //---------------------------------------------------------//

    // Note: the following function already got a lock,
    // don't use the same lock recursively
    _set_index_write( _increase_idx(_idx_write_tmp) );
    return _all_is_well;
}


template <class _T> bool async_buffer<_T>::front(_T & content_out, bool is_poping, const TIME_STAMP::Time &stamp_req){
    // To get an element from the buffer
    // Return false if it's empty

    // To lock the read for ensuring only one consumer a time
    //-------------------------------------------------------//
    std::lock_guard<std::mutex> _lock(*_mlock_read_block);
    //-------------------------------------------------------//

    // If the buffer is empty, we return
    if (_is_empty()){
        return false;
    }
    // else
    int _idx_read_tmp = _get_index_read();



    // Search for closest time stamp
    //---------------------------//
    // _idx_read_tmp = ?
    if (!stamp_req.is_zero()){
        // _idx_read_tmp = get_idx_of_closest_stamp(_idx_read_tmp, stamp_req);
        int _idx_op = get_idx_of_closest_stamp(_idx_read_tmp, stamp_req);
        if (_idx_op < 0){
            return false;
        }
        _idx_read_tmp = _idx_op;
    }
    // Store the stamp (can put this line right after getting the index)
    _stamp_out = _stamp_list[_idx_read_tmp];
    //
    //---------------------------//


    // Fill the pointer
    _fill_in_ptr_if_empty(_data_ptr_list[_idx_read_tmp]);
    //


    // pop?
    if(!is_poping){
        // Fill the pointer
        _fill_in_ptr_if_empty(_tmp_output_ptr);
        //
        if (_got_front_but_no_pop){
            // _got_front_but_no_pop = true;
            // --- No need to get element from the buffer again
            _copy_func(content_out, *_tmp_output_ptr); // instance <-- *ptr
        }else{
            _got_front_but_no_pop = true;
            // Swap pointer with _tmp_output_ptr
            _tmp_output_ptr.swap(_data_ptr_list[_idx_read_tmp]);
            _copy_func(content_out, *_tmp_output_ptr); // instance <-- *ptr
            _set_index_read(_idx_read_tmp); // In case that the time searching found a different index
        }
    }else{
        // Reset the flag for front
        _got_front_but_no_pop = false;
        //

        // We need to copy the data first before we move the index (delete)
        // Note: the default copy method may not sussess if _T is "Mat" from opencv
        //       be sure to use IMG.clone() method in customized _copy_func() using assign_copy_func() method.
        // The following operation might be time consumming
        _copy_func(content_out, *_data_ptr_list[_idx_read_tmp]);
        // content_out = std::move(_data_ptr_list[_idx_read_tmp]); // The content in the buffer will disappear.

        // Note: the following function already got a lock,
        // don't use the same lock recursively
        _set_index_read( _increase_idx(_idx_read_tmp) );

    }
    //
    return true;
}

template <class _T> bool async_buffer<_T>::front(std::shared_ptr<_T> & content_out_ptr, bool is_poping, const TIME_STAMP::Time &stamp_req){
    // If is_poping, exchanging the data out, fast;
    // if not is_poping, share the content (Note: this may not be safe!!)

    // To get an element from the buffer
    // Return false if it's empty

    // To lock the read for ensuring only one consumer a time
    //-------------------------------------------------------//
    std::lock_guard<std::mutex> _lock(*_mlock_read_block);
    //-------------------------------------------------------//

    // If the buffer is empty, we return
    if (_is_empty()){
        return false;
    }
    // else
    int _idx_read_tmp = _get_index_read();


    // Search for closest time stamp
    //---------------------------//
    // _idx_read_tmp = ?
    if (!stamp_req.is_zero()){
        // _idx_read_tmp = get_idx_of_closest_stamp(_idx_read_tmp, stamp_req);
        int _idx_op = get_idx_of_closest_stamp(_idx_read_tmp, stamp_req);
        if (_idx_op < 0){
            return false;
        }
        _idx_read_tmp = _idx_op;
    }
    // Store the stamp (can put this line right after getting the index)
    _stamp_out = _stamp_list[_idx_read_tmp];
    //
    //---------------------------//



    // Fill the pointer
    _fill_in_ptr_if_empty(_data_ptr_list[_idx_read_tmp]);
    //


    // pop?
    if(!is_poping){
        // Fill the pointer
        _fill_in_ptr_if_empty(_tmp_output_ptr);
        //
        if (_got_front_but_no_pop){
            // _got_front_but_no_pop = true;
            // --- No need to get element from the buffer again
            content_out_ptr = _tmp_output_ptr; // Share content with _tmp_output_ptr
        }else{
            _got_front_but_no_pop = true;
            // Swap the pointer with _tmp_output_ptr
            _tmp_output_ptr.swap(_data_ptr_list[_idx_read_tmp]);
            content_out_ptr = _tmp_output_ptr; // Share content with _tmp_output_ptr
            _set_index_read(_idx_read_tmp); // In case that the time searching found a different index
        }
    }else{
        // Reset the flag for front
        _got_front_but_no_pop = false;
        //

        // We need to exchange the data first before we move the index (delete)
        //---------------------------------------------------------//

        // Check if the content_out_ptr is null
        _fill_in_ptr_if_empty(content_out_ptr);
        //

        // Pre-check: The input pointer should be pure (unique or empty)
        if ( is_ptr_shared(_data_ptr_list[_idx_read_tmp]) ){ // Not null and not unique
        // if (_data_ptr_list[_idx_read_tmp] && !_data_ptr_list[_idx_read_tmp].unique() ){ // Not null and not unique
            // Copy element
            // Note: the copy method may not sussess if _T is "Mat" from opencv
            //       be sure to use IMG.clone() mwthod for putting an image in.
            _copy_func(*content_out_ptr, *_data_ptr_list[_idx_read_tmp]); // *ptr <-- *ptr
            #ifdef __DEGUG__
            std::cout << "[front pop] buffer pointer is not pure.\n";
            #endif
        }else{
            // The input pointer is pure (unique or null)
            // swapping
            content_out_ptr.swap(_data_ptr_list[_idx_read_tmp]);

            // Post-check: the output pointer should be unique (not empty and not shared)
            _reset_ptr_if_shared(_data_ptr_list[_idx_read_tmp]);
        }
        //---------------------------------------------------------//


        // Note: the following function already got a lock,
        // don't use the same lock recursively
        _set_index_read( _increase_idx(_idx_read_tmp) );
    }
    //
    return true;
}


template <class _T> bool async_buffer<_T>::pop(){
    // To remove an element from the buffer
    // Return false if it's empty

    // To lock the read for ensuring only one consumer a time
    //-------------------------------------------------------//
    std::lock_guard<std::mutex> _lock(*_mlock_read_block);
    //-------------------------------------------------------//

    // Reset the flag for front
    _got_front_but_no_pop = false;
    //

    if (_is_empty()){
        return false;
    }
    // else
    int _idx_read_tmp = _get_index_read();

    // Note: the following function already got a lock,
    // don't use the same lock recursively
    _set_index_read( _increase_idx(_idx_read_tmp) );
    return true;
}



// boost::any wrappers
//-----------------------------------------------//
template <class _T> bool async_buffer<_T>::put_any(boost::any & element_in_ptr, bool is_droping, const TIME_STAMP::Time &stamp_in, bool is_no_copying){  // Exchanging the data, fast
    if (element_in_ptr.empty()){
        element_in_ptr =  std::shared_ptr< _T >();
    }
    std::shared_ptr< _T > *_ptr_ptr = boost::any_cast< std::shared_ptr< _T > >( &element_in_ptr );
    // std::cout << "_ptr_ptr->use_count() = " << _ptr_ptr->use_count() << "\n";
    if (is_no_copying){
        return put(*_ptr_ptr, is_droping, stamp_in);
    }else{
        return put(*(*_ptr_ptr), is_droping, stamp_in);
    }
}
// Front, overloading
template <class _T> bool async_buffer<_T>::front_any(boost::any & content_out_ptr, bool is_poping, const TIME_STAMP::Time &stamp_req){  // If is_poping, exchanging the data out, fast; if not is_poping, share the content (Note: this may not be safe!!)
    if (content_out_ptr.empty()){
        content_out_ptr =  std::shared_ptr< _T >();
    }
    std::shared_ptr< _T > *_ptr_ptr = boost::any_cast< std::shared_ptr< _T > >( &content_out_ptr );
    return front(*_ptr_ptr, is_poping, stamp_req);
}
//-----------------------------------------------//

// void* wrappers
//-----------------------------------------------//
// Put in the std::shared_ptr as element_in_ptr
template <class _T> bool async_buffer<_T>::put_void(const void * element_in_ptr, bool is_droping, const TIME_STAMP::Time &stamp_in, bool is_shared_ptr){  // Exchanging the data, fast
    try{
        if (is_shared_ptr){
            return put(*(std::shared_ptr<_T> *)(element_in_ptr), is_droping, stamp_in);
        }else{
            return put(*(_T *)(element_in_ptr), is_droping, stamp_in);
        }
    }catch(std::exception& e){
        std::cout << "---\n";
        std::cout << "Bad usage of async_buffer::put_void(), probably type error.\n";
        std::cout << "The require type is [" << typeid(_T).name() << "]\n";
        std::cout << "is_shared_ptr = " << is_shared_ptr << "\n";
        std::cout << "Error: e = <" << e.what() << ">\n";
        std::cout << "---\n";
    }
    return false;
    // return put(*element_in_ptr, is_droping, stamp_in);
}
// Put in the std::shared_ptr as content_out_ptr
template <class _T> bool async_buffer<_T>::front_void(void * content_out_ptr, bool is_poping, const TIME_STAMP::Time &stamp_req, bool is_shared_ptr){  // If is_poping, exchanging the data out, fast; if not is_poping, share the content (Note: this may not be safe!!)
    try{
        if (is_shared_ptr){
            return front(*(std::shared_ptr<_T> *)(content_out_ptr), is_poping, stamp_req);
        }else{
            return front(*(_T *)(content_out_ptr), is_poping, stamp_req);
        }
    }catch(std::exception& e){
        std::cout << "---\n";
        std::cout << "Bad usage of async_buffer::put_void(), probably type error.\n";
        std::cout << "The require type is [" << typeid(_T).name() << "]\n";
        std::cout << "is_shared_ptr = " << is_shared_ptr << "\n";
        std::cout << "Error: e = <" << e.what() << ">\n";
        std::cout << "---\n";
    }
    return false;
    // return front(*content_out_ptr, is_poping, stamp_req);
}
//-----------------------------------------------//

//
template <class _T> bool async_buffer<_T>::_is_empty(){
    // Note: This method is used by "consumer"
    // This is a fast method which may return true even when there are some elements in queue (but not vise versa)

    // Cache the "write" first
    int _idx_write_tmp, _idx_read_tmp;
    _idx_write_tmp = _get_index_write();
    _idx_read_tmp = _get_index_read();

    return _is_empty_cal(_idx_write_tmp, _idx_read_tmp);
}

template <class _T> bool async_buffer<_T>::_is_full(){
    // Note: This method is used by "producer"
    // // This is a fast method which may return true even when there are some sapces in queue (but not vise versa)

    // Cache the "read" first
    int _idx_write_tmp, _idx_read_tmp;
    _idx_read_tmp = _get_index_read();
    _idx_write_tmp = _get_index_write();

    return _is_full_cal(_idx_write_tmp, _idx_read_tmp);
}

template <class _T> int async_buffer<_T>::_size_est(){
    // This method may be used by both producer and consumer
    // The number of buffered contents. This is a fast method with no mutex but may be incorrect in both direction.

    // Cache the "read" first, since the write might change more frequently
    int _idx_write_tmp, _idx_read_tmp;
    _idx_read_tmp = _get_index_read();
    _idx_write_tmp = _get_index_write();

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
