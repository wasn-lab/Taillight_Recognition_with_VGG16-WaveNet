#ifndef ROS_IMAGE_PUB_SUB
#define ROS_IMAGE_PUB_SUB


#include <vector>
#include <map>
#include <string>
#include <utility>      // std::pair, std::make_pair
//
#include <memory> // <-- this is for std::shared_ptr
#include <mutex>

// Boost
#include <boost/bind.hpp>

//
#include <ros/ros.h>

// MSG: Image
#include <opencv2/opencv.hpp>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>


class ImageTransfer{
public: 
    ImageTransfer();

    bool put(const cv::Mat &content_in  );
    bool put( std::shared_ptr<cv::Mat> &content_in_ptr  );
    bool get( std::shared_ptr<cv::Mat> &content_out_ptr );

    // Copy function for cv::Mat
    //-------------------------------------------//
    static bool _copy_func(cv::Mat & _target, const cv::Mat & _source){
        // _target = _source.clone();
        _source.copyTo(_target);
        return true;
    }
    //-------------------------------------------//

    inline bool is_ptr_shared(const std::shared_ptr<cv::Mat> & _ptr_in){
        return ( _ptr_in && !_ptr_in.unique() );
    }
    inline void _reset_ptr_if_shared(std::shared_ptr<cv::Mat> & _ptr_in){
        if (is_ptr_shared(_ptr_in)){    _ptr_in.reset( new cv::Mat );    }
    }

private:
    std::shared_ptr<std::mutex> _mlock_data;
    std::shared_ptr<cv::Mat> _data_ptr;

    size_t write_count;
    size_t read_count;
};


//==================================================//
//==================================================//
class RosImagePubSub{
public:

    RosImagePubSub(ros::NodeHandle &nh_in);
    // Publishers
    bool add_a_pub(size_t id_in, const std::string &topic_name);
    bool send_image(const int topic_id, const cv::Mat &content_in);
	bool send_image_rgb(const int topic_id, const cv::Mat &content_in);
    // Subscribers
    bool add_a_sub(size_t id_in, const std::string &topic_name);
    bool get_image(const int topic_id, cv::Mat &content_out);
	unsigned int sequence = 0;

    // Copy function for cv::Mat
    //-------------------------------------------//
    static bool _cv_Mat_copy_func(cv::Mat & _target, const cv::Mat & _source){
        // _target = _source.clone();
        _source.copyTo(_target);
        return true;
    }
    //-------------------------------------------//
private:


    // Handle with default namespace
    ros::NodeHandle *_nh_ptr;
    // ROS image transport (similar to  node handle, but for images)
    image_transport::ImageTransport _ros_it;

    // Image subscribers
    std::map<size_t, image_transport::Subscriber> _image_subscriber_map;
    // Image publishers
    std::map<size_t, image_transport::Publisher> _image_publisher_map;


    // ImageTransfer
    std::map<size_t, ImageTransfer>  image_trans_map;

    // callback
    void _CompressedImage_CB(const sensor_msgs::ImageConstPtr& msg, const size_t topic_id);
    std::map<size_t, std::shared_ptr<cv::Mat>> content_out_ptr_map;
};



#endif // ROS_IMAGE_PUB_SUB
