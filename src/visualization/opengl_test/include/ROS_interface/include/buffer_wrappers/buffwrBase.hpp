#ifndef BUFFWR_BASE_H
#define BUFFWR_BASE_H


// all_header.h
#include <all_header.h>

// The derived class for the async_buffer can be any type
// The base class goes with type "char" as example

class buffwrBase{

    typedef char _BUFFER_TYPE_;
public:

    buffwrBase(): _buffer(0)
    {
        // _buffer.assign_copy_func(&_customized_copy_func);
    }
    buffwrBase(size_t buffer_length_in): _buffer(buffer_length_in)
    {
        // _buffer.assign_copy_func(&_customized_copy_func);
    }

    // Public methods
    //---------------------------------------------------------//
    // Please refers to the doc. below for usage of put_any() and front_any().
    inline virtual bool put_any(boost::any & element_in_ptr, bool is_droping=true, const TIME_STAMP::Time &stamp_in=TIME_STAMP::Time::now(), bool is_no_copying=true){
        return _buffer.put_any(element_in_ptr, is_droping, stamp_in, is_no_copying);
    }
    inline virtual bool front_any(boost::any & content_out_ptr, bool is_poping=false, const TIME_STAMP::Time &stamp_req=TIME_STAMP::Time()){
        return _buffer.front_any(content_out_ptr, is_poping, stamp_req);
    }
    // Put in the &(std::shared_ptr<_T>) or &(_T) as element_in_ptr
    inline virtual bool put_void(const void * content_in_ptr, bool is_droping=true, const TIME_STAMP::Time &stamp_in=TIME_STAMP::Time::now(), bool is_shared_ptr=false){  // Exchanging the data, fast
        return _buffer.put_void(content_in_ptr, is_droping, stamp_in, is_shared_ptr);
    }
    // Put in the &(std::shared_ptr<_T>) or &(_T) as content_out_ptr
    inline virtual bool front_void(void * content_out_ptr, bool is_poping=false, const TIME_STAMP::Time &stamp_req=TIME_STAMP::Time(), bool is_shared_ptr=false){  // If is_poping, exchanging the data out, fast; if not is_poping, share the content (Note: this may not be safe!!)
        return _buffer.front_void(content_out_ptr, is_poping, stamp_req, is_shared_ptr);
    }
    inline virtual bool pop(){ return _buffer.pop(); }
    inline virtual TIME_STAMP::Time get_stamp(void){return _buffer.get_stamp();} // Note: use this function right after using any one of the "front" method
    //
    inline virtual void * get_tmp_in_ptr(){   return &(_tmp_in_ptr);  }
    //---------------------------------------------------------//

    // The temporary container
    std::shared_ptr< _BUFFER_TYPE_ > _tmp_in_ptr; // tmp input element

protected:
    /*
    static bool _customized_copy_func(_BUFFER_TYPE_ & _target, const _BUFFER_TYPE_ & _source){
        _target = _source.clone();
        return true;
    }
    */

private:
    // The buffer (e.g. with type "_BUFFER_TYPE_" )
    async_buffer< _BUFFER_TYPE_ >  _buffer;
};


/*
// Using put_any() with content_in: (The following syntex makes a clear poiter transfer withoud copying or sharring)
//---------------------------------------//
boost::any any_ptr;
{
    std::shared_ptr< _T > _content_ptr = std::make_shared< _T >( content_in ); // <-- If we already get the std::shared_ptr, ignore this line
    any_ptr = _content_ptr;
} // <-- Note: the _content_ptr is destroyed when leaveing the scope, thus the use_count for the _ptr in any_ptr is "1" (unique).
buffwr_obj.put_any(any_ptr, true, _time_in, true);
//---------------------------------------//

// Using put_any() with content_in_ptr: (The following syntex makes a clear poiter transfer withoud copying or sharring)
//---------------------------------------//
boost::any any_ptr;
{
    std::shared_ptr< _T > _content_ptr = std::make_shared< _T >( *content_in_ptr ); // <-- If we already get the std::shared_ptr, ignore this line
    any_ptr = _content_ptr;
} // <-- Note: the _content_ptr is destroyed when leaveing the scope, thus the use_count for the _ptr in any_ptr is "1" (unique).
buffwr_obj.put_any(any_ptr, true, _time_in, true);
//---------------------------------------//

// Using front_any() with content_out_ptr: (The following syntex makes a clear poiter transfer withoud copying or sharring)
//---------------------------------------//
std::shared_ptr< _T > content_out_ptr;
{
    boost::any any_ptr;
    bool result = buffwr_list[topic_id]->front_any(any_ptr, true, _current_slice_time);
    if (result){
        // content_out_ptr = boost::any_cast< std::shared_ptr< cv::Mat > >( any_ptr ); // <-- Not good, this makes a copy
        std::shared_ptr< cv::Mat > *_ptr_ptr = boost::any_cast< std::shared_ptr< cv::Mat > >( &any_ptr );
        content_out_ptr = *_ptr_ptr;
    }
} // <-- Note: the any_ptr is destroyed when leaving this scope, thus the use_count for content_out_ptr is "1" (unique).
//---------------------------------------//
*/

#endif // BUFFWR_BASE_H
