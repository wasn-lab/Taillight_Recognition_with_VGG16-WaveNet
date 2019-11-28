#ifndef GUI_ROS_INTERFACE_H
#define GUI_ROS_INTERFACE_H

// Determine if we are going to preint some debug information to std_out
// #define __DEGUG__

// #define __PUB_WITH_BUFFER__

// all_header.h
#include <all_header.h>

// Buffer wrappers
// #include <buffwrBase.hpp>
// #include <buffwrString.hpp>
// #include <buffwrtfGeoPoseStamped.hpp>
// #include <buffwrImage.hpp>
// #include <buffwrPointCloud2.hpp>
// #include <buffwrITRIPointCloud.hpp>
// #include <buffwrITRI3DBoundingBox.hpp>


// Example:
// Note: the following thing should be done in main code
// The identifier for message global id
//-----------------------------------------//
// List the topic tokens in sequence wuth the same order as the topic names in topics name list
/*
// nickname for topic_id
enum class MSG_ID{
    chatter_0,
    chatter_1,
    chatter_2,
    chatter_3,
    chatter_4,
    chatter_5
};

ROS_INTERFACE ros_interface;
// Method 1:
{
    using MSG::M_TYPE;
    ros_interface.add_a_topic("/chatter_0", M_TYPE::String, true, 10, 10);
    ros_interface.add_a_topic("/chatter_1", M_TYPE::String, true, 10, 10);
    ros_interface.add_a_topic("/chatter_2", M_TYPE::String, true, 10, 10);
}

// Method 2:
std::vector<MSG::T_PARAMS> topic_param_list;
{
    using MSG::T_PARAMS;
    using MSG::M_TYPE;
    // Start adding topics
    topic_param_list.push_back( T_PARAMS("/chatter_0", M_TYPE::String, true, 10, 10) );
    topic_param_list.push_back( T_PARAMS("/chatter_1", M_TYPE::String, true, 10, 10) );
    topic_param_list.push_back( T_PARAMS("/chatter_2", M_TYPE::String, true, 10, 10) );
}
ros_interface.load_topics(topic_param_list);
*/
//-----------------------------------------//


// library begins
//--------------------------------------------------//
namespace MSG{
    // The identifier for message types
    // Note: the type of the enum is defaultly int
    enum class M_TYPE{
        Bool,
        Int32,
        String,
        GUI2_op,
        tfGeoPoseStamped,
        Image,
        CompressedImageROSIT,
        CompressedImageJpegOnly,
        PointCloud2,
        ITRIPointCloud,
        ITRI3DBoundingBox,
        ITRICamObj,
        ITRIDetectedObjectArray,
        ITRICarInfoCarA,
        ITRICarInfo,
        ITRIDynamicPath,
        ITRIFlagInfo,
        ITRITransObj,
        NUM_MSG_TYPE
    };

    struct T_PARAMS{
        std::string name;
        int type; // According to the enum value defined in ROS_INTERFACE class
        bool is_input; // if not, it's output
        size_t ROS_queue; // The lengh of the ROS queue
        size_t buffer_length; // The buffer length setting for the SPSC buffer used for this topic
        //
        std::string frame_id; // The frame of which the data is represented at.
        bool is_transform; // Indicate that whether this topic is about a transform or not.
        std::string to_frame; // Not necessary, only needed when the topic is about a transformation.

        //
        int topic_id; // The topic_id for backward indexing. This is assigned by ROS_INTERFACE, according to the order of adding sequence
        //
        T_PARAMS():
            name(""),
            type(-1),
            is_input(false),
            ROS_queue(10),
            buffer_length(1),
            topic_id(-1),
            frame_id(""),
            is_transform(false),
            to_frame("")
        {}
        T_PARAMS(
            const std::string &name_in,
            int type_in,
            bool is_input_in,
            size_t ROS_queue_in,
            size_t buffer_length_in,
            std::string frame_id_in="",
            bool is_transform_in=false,
            std::string to_frame_in=""
        ):
            name(clearTopicName(name_in)),
            type(type_in),
            is_input(is_input_in),
            ROS_queue(ROS_queue_in),
            buffer_length(buffer_length_in),
            topic_id(-1),
            frame_id(frame_id_in),
            is_transform(is_transform_in),
            to_frame(to_frame_in)
        {}
        // The following constructor is not for user (additional topic_id to be set)
        T_PARAMS(
            const std::string &name_in,
            int type_in,
            bool is_input_in,
            size_t ROS_queue_in,
            size_t buffer_length_in,
            size_t topic_id_in,
            std::string frame_id_in="",
            bool is_transform_in=false,
            std::string to_frame_in=""
        ):
            name(clearTopicName(name_in)),
            type(type_in),
            is_input(is_input_in),
            ROS_queue(ROS_queue_in),
            buffer_length(buffer_length_in),
            topic_id(topic_id_in),
            frame_id(frame_id_in),
            is_transform(is_transform_in),
            to_frame(to_frame_in)
        {}
        //
        std::string clearTopicName(std::string name_in){
            std::string unwantedChar (" \t\f\v\n\r/");
            std::size_t found = name_in.find_last_not_of(unwantedChar);
            if (found!=std::string::npos){
                if (found != (name_in.size()-1)){
                    name_in.erase(found+1);
                    std::cout << "Fixed topic name: [" << name_in << "]\n";
                }// else, the string doesn't need to be fixed
            }// else, the topic is empty, do nothing
            return name_in;
        }
    };
}// end of the namespace MSG


//
//
//
class ROS_INTERFACE{

public:
    // Constructors
    ROS_INTERFACE();
    ROS_INTERFACE(int argc, char **argv);
    ~ROS_INTERFACE();
    //
    bool setup_node(int argc, char **argv, std::string node_name_in=std::string("ROS_interface"));
    // Setting up topics
    // Method 1: use add_a_topic to add a single topic sequentially one at a time
    // Method 2: use add_a_topic to add a single topic specific to topic_id
    // Method 3: use load_topics to load all topics
    bool add_a_topic(
        const std::string &name_in,
        int type_in,
        bool is_input_in,
        size_t ROS_queue_in,
        size_t buffer_length_in,
        std::string frame_id_in="",
        bool is_transform_in=false,
        std::string to_frame_in=""
    );
    bool add_a_topic(
        int topic_id_in,
        const std::string &name_in,
        int type_in,
        bool is_input_in,
        size_t ROS_queue_in,
        size_t buffer_length_in,
        std::string frame_id_in="",
        bool is_transform_in=false,
        std::string to_frame_in=""
    );
    bool load_topics(const std::vector<MSG::T_PARAMS> &topic_param_list_in);
    // Really start the ROS thread
    bool start();


    // Check if the ROS is started
    bool is_running();
    // Get topic imformations
    inline bool is_topic_id_valid(const int topic_id){return (_topic_param_list[topic_id].type >= 0); }
    inline bool is_topic_a_input(const int topic_id){return _topic_param_list[topic_id].is_input; }
    inline bool is_topic_got_frame(const int topic_id){return (_topic_param_list[topic_id].frame_id.size() > 0);}
    inline MSG::T_PARAMS get_topic_param(const int topic_id){ return _topic_param_list[topic_id]; }
    inline std::string get_topic_name(const int topic_id){ return _topic_param_list[topic_id].name; }
    inline size_t get_count_of_all_topics(){    return _topic_param_list.size();    }
    inline size_t get_count_of_a_topic_type(MSG::M_TYPE topic_type){    return (_msg_type_2_topic_params[int(topic_type)].size() );  }

    // Utilities
    inline ros::Time        toROStime(const TIME_STAMP::Time & time_in){return ros::Time(time_in.sec, time_in.nsec); }
    inline TIME_STAMP::Time fromROStime(const ros::Time & rostime_in){return TIME_STAMP::Time(rostime_in.sec, rostime_in.nsec);}


    // (Legacy) Getting methods for each type of message
    // The topic_id should be the "global id"
    //---------------------------------------------------------//
    bool get_String(const int topic_id, std::string & content_out);
    bool get_String(const int topic_id, std::shared_ptr< std::string> & content_out_ptr);
    bool get_Image(const int topic_id, cv::Mat & content_out);
    bool get_Image(const int topic_id, std::shared_ptr<cv::Mat> & content_out_ptr);
    bool get_PointCloud2(const int topic_id, pcl::PointCloud<pcl::PointXYZI> & content_out);
    bool get_PointCloud2(const int topic_id, std::shared_ptr< pcl::PointCloud<pcl::PointXYZI> > & content_out_ptr);
    bool get_PointCloud2(const int topic_id, std::shared_ptr< pcl::PointCloud<pcl::PointXYZI> > & content_out_ptr, ros::Time &msg_stamp);
    bool get_ITRIPointCloud(const int topic_id, pcl::PointCloud<pcl::PointXYZI> & content_out);
    bool get_ITRIPointCloud(const int topic_id, std::shared_ptr< pcl::PointCloud<pcl::PointXYZI> > & content_out_ptr);
    bool get_ITRIPointCloud(const int topic_id, std::shared_ptr< pcl::PointCloud<pcl::PointXYZI> > & content_out_ptr, ros::Time &msg_stamp);
    bool get_ITRI3DBoundingBox(const int topic_id, msgs::DetectedObjectArray & content_out);
    bool get_ITRI3DBoundingBox(const int topic_id, std::shared_ptr< msgs::DetectedObjectArray > & content_out_ptr);
    bool get_ITRI3DBoundingBox(const int topic_id, std::shared_ptr< msgs::DetectedObjectArray > & content_out_ptr, ros::Time &msg_stamp);
    //---------------------------------------------------------//

    // Sending methods for each type of message
    // The topic_id should be the "global id"
    //---------------------------------------------------------//
    bool send_string(const int topic_id, const std::string &content_in);
    bool send_GUI2_op(const int topic_id, const opengl_test::GUI2_op &content_in);
    bool send_Image(const int topic_id, const cv::Mat &content_in);
    bool send_ITRIPointCloud(const int topic_id, const pcl::PointCloud<pcl::PointXYZI> &content_in);
    bool send_ITRI3DBoundingBox(const int topic_id, const msgs::LidRoi &content_in);
    //---------------------------------------------------------//

    // New interfaces - boost::any and (void *)
    //---------------------------------------------------------//
    bool get_any_message(const int topic_id, boost::any & content_out_ptr);
    bool get_any_message(const int topic_id, boost::any & content_out_ptr, ros::Time &msg_stamp);
    bool get_void_message(const int topic_id, void * content_out_ptr, bool is_shared_ptr=true);
    bool get_void_message(const int topic_id, void * content_out_ptr, ros::Time &msg_stamp, bool is_shared_ptr=true );
    //---------------------------------------------------------//

    // (Legacy) Combined same buffer-data types
    //---------------------------------------------------------//
    // pcl::PointCloud<pcl::PointXYZI>
    bool get_any_pointcloud(const int topic_id, pcl::PointCloud<pcl::PointXYZI> & content_out);
    bool get_any_pointcloud(const int topic_id, std::shared_ptr< pcl::PointCloud<pcl::PointXYZI> > & content_out_ptr);
    bool get_any_pointcloud(const int topic_id, std::shared_ptr< pcl::PointCloud<pcl::PointXYZI> > & content_out_ptr, ros::Time &msg_stamp);
    //---------------------------------------------------------//


    // Method of time-sync for buffer outputs
    //----------------------------------------------//
    bool update_current_slice_time();
    inline ros::Time get_current_slice_time(){ return toROStime(_current_slice_time);  }
    inline void set_global_delay(const long double & global_delay_in){_global_delay = global_delay_in;}
    inline long double get_global_delay(){ return _global_delay;}
    //----------------------------------------------//


    // Advanced getting methods: get transformations
    //---------------------------------------------------------//
    bool set_ref_frame(const std::string & ref_frame_in);
    bool update_latest_tf_common_update_time(const std::string &ref_frame_in, const std::string &to_frame_in);
    ros::Time get_latest_tf_common_update_time();
    // Get tf
    bool get_tf(std::string base_fram, std::string to_frame, geometry_msgs::TransformStamped & tf_out, bool is_time_traveling=false, ros::Time lookup_stamp=ros::Time::now() );
    geometry_msgs::TransformStamped get_tf(std::string base_fram, std::string to_frame, bool & is_sucessed, bool is_time_traveling=false, ros::Time lookup_stamp=ros::Time::now());
    bool get_tf(std::string at_frame, geometry_msgs::TransformStamped & tf_out, bool is_time_traveling=false, ros::Time lookup_stamp=ros::Time::now() ); // _ref_frame --> at_frame
    geometry_msgs::TransformStamped get_tf(std::string at_frame, bool & is_sucessed, bool is_time_traveling=false, ros::Time lookup_stamp=ros::Time::now()); // _ref_frame --> at_frame
    bool get_tf(const int topic_id, geometry_msgs::TransformStamped & tf_out, bool is_time_traveling=false, ros::Time lookup_stamp=ros::Time::now());
    geometry_msgs::TransformStamped get_tf(const int topic_id, bool & is_sucessed, bool is_time_traveling=false, ros::Time lookup_stamp=ros::Time::now());
    //---------------------------------------------------------//

    // Message time stamps
    //---------------------------------------------------------//
    //---------------------------------------------------------//



private:
    bool _is_started;
    // size_t _num_topics;
    size_t _num_ros_cb_thread;

    // Although we only use "one" thread, by using this container,
    // we can start the thread later by push_back element
    std::vector<std::thread> _thread_list;
    void _ROS_worker();

    // ROS tf2
    tf2_ros::Buffer tfBuffer;
    std::shared_ptr<tf2_ros::TransformListener> tfListener_ptr;
    std::shared_ptr<tf2_ros::TransformBroadcaster> tfBrocaster_ptr;
    std::string _ref_frame; // The frame_id of the frame which all other data will be represented on. This frame_id can be dynamically changed and got a global effect on all the transformation.
    std::string _stationary_frame; // The frame that is used for time-traveling.
    // The frame time
    bool _is_using_latest_tf_common_update_time;
    ros::Time   _latest_tf_common_update_time;


    // The synchronized time for "all" buffers output at current "slice" (or sample)
    //-------------------------------------------------------------------------------//
    TIME_STAMP::Time _current_slice_time;// The newest time slice for sampling output, which is used to unify all the transform in one sampling.
    long double _global_delay; // The global delay time for calculating the _current_slice_time relative to the "current" wall time
    // Which is: _current_slice_time = TIME_STAMP::Time::now() - TIME_STAMP::Time(_global_delay);
    //-------------------------------------------------------------------------------//


    // List of topics parameters
    // TODO: make a containner to contain the following things in a single object for each topic
    //------------------------------//
    // Note: the following are setting be user
    // mapping - topic_id : topic parameters (MSG::T_PARAMS)
    std::vector<MSG::T_PARAMS> _topic_param_list;
    //------------------------------//

    // List of topic tid in each type of message
    // The tid is mainly used for indexing "SPSC buffers"
    // Note 1: this "tid" or type_id is type specified, different from global topic_id
    // Note 2: These vetor will be filled at adding/loading topics
    //------------------------------//
    std::vector<int> _topic_tid_list; // mapping - topic_id : topic_type_id
    // Topic-param list for each message type
    std::vector< std::vector<MSG::T_PARAMS> > _msg_type_2_topic_params; // _msg_type_2_topic_params[type][tid] --> T_PARAMS
    //------------------------------//

    // The subs_id and pub_id for each topic_id
    // mapping - topic_id : subs_id/pub_id
    // The subs_id/pub_id is mainly used for indexing subscribers/publishers
    // Note: these ids as well as SPSC buffers will be generated when subscribing/advertising
    //------------------------------//
    std::vector<int> _pub_subs_id_list;
    //------------------------------//


    // General subscriber/publisher
    // Subscribers
    vector<ros::Subscriber> _subscriber_list;
    // Publishers
    vector<ros::Publisher> _publisher_list;

    // ROS image transport (similar to  node handle, but for images)
    // image_transport::ImageTransport _ros_it;
    // Camera subscribers
    vector<image_transport::Subscriber> _image_subscriber_list;
    // Image publishers
    vector<image_transport::Publisher> _image_publisher_list;


    // Copy function for cv::Mat
    //-------------------------------------------//
    // Note: if _T is the opencv Mat,
    //       you should attach acopy function using Mat::clone() or Mat.copyTo()
    // Note: static members are belong to class itself not the object
    static bool _cv_Mat_copy_func(cv::Mat & _target, const cv::Mat & _source){
        // _target = _source.clone();
        _source.copyTo(_target);
        return true;
    }
    //-------------------------------------------//

    // SPSC Buffers
    //---------------------------------------------------------//
    // Note: each message type got an array of SPSC buffers with that type,
    //       each single topic use an unique SPSC buffer.
    //       The result is that we will have multiple arrays of SPSC buffers
    //---------------------------------------------------------//

    // test, buffer wrapper (the base class with virtual function)
    // std::vector< std::shared_ptr<buffwrBase> > buffwr_list;
    std::vector< std::shared_ptr<async_buffer_base> > async_buffer_list;
    // test_3, any buffer
    // std::vector< boost::any> any_buffer_list;
    // test, buffer -- any
    // std::vector< async_buffer<boost::any> > buffer_any;
    //---------------------------------------------------------//


    // Callbacks
    //---------------------------------------------------------//
    // Bool
    void _Bool_CB(const std_msgs::Bool::ConstPtr& msg, const MSG::T_PARAMS & params);
    // Int32
    void _Int32_CB(const std_msgs::Int32::ConstPtr& msg, const MSG::T_PARAMS & params);
    // String
    void _String_CB(const std_msgs::String::ConstPtr& msg, const MSG::T_PARAMS & params);
    // bool _String_pub();
    // GUI2_op
    void _GUI2_op_CB(const opengl_test::GUI2_op::ConstPtr& msg, const MSG::T_PARAMS & params);
    // tfGeoPoseStamped
    void _tfGeoPoseStamped_CB(const geometry_msgs::PoseStamped::ConstPtr& msg, const MSG::T_PARAMS & params);
    // Image
    void _Image_CB(const sensor_msgs::ImageConstPtr& msg, const MSG::T_PARAMS & params);
    // CompressedImageROSIT
    void _CompressedImageROSIT_CB(const sensor_msgs::ImageConstPtr& msg, const MSG::T_PARAMS & params);
    // CompressedImageJpegOnly
    void _CompressedImageJpegOnly_CB(const sensor_msgs::CompressedImageConstPtr& msg, const MSG::T_PARAMS & params);
    std::string _addCompressedToTopicName(std::string name_in);
    std::vector< std::shared_ptr<cv::Mat> > _cv_Mat_tmp_ptr_list;
    // PointCloud2
    // void _PointCloud2_CB(const sensor_msgs::PointCloud2::ConstPtr& msg, const MSG::T_PARAMS & params);
    void _PointCloud2_CB(const pcl::PCLPointCloud2ConstPtr& msg, const MSG::T_PARAMS & params);
    std::vector< std::shared_ptr< pcl::PointCloud<pcl::PointXYZI> > > _PointCloud2_tmp_ptr_list; // tmp cloud
    // ITRIPointCloud
    void _ITRIPointCloud_CB(const msgs::PointCloud::ConstPtr& msg, const MSG::T_PARAMS & params);
    std::vector< std::shared_ptr< pcl::PointCloud<pcl::PointXYZI> > >  _ITRIPointCloud_tmp_ptr_list; // tmp cloud
    // ITRI3DBoundingBox
    void _ITRI3DBoundingBox_CB(const msgs::LidRoi::ConstPtr& msg, const MSG::T_PARAMS & params);
    // ITRICamObj
    void _ITRICamObj_CB(const msgs::CamObj::ConstPtr& msg, const MSG::T_PARAMS & params);
    // ITRIDetectedObjectArray
    void _ITRIDetectedObjectArray_CB(const msgs::DetectedObjectArray::ConstPtr& msg, const MSG::T_PARAMS & params);
    // ITRICarInfoCarA
    void _ITRICarInfoCarA_CB(const msgs::TaichungVehInfo::ConstPtr& msg, const MSG::T_PARAMS & params);
    // ITRICarInfo
    void _ITRICarInfo_CB(const msgs::VehInfo::ConstPtr& msg, const MSG::T_PARAMS & params);
    // ITRIDynamicPath
    void _ITRIDynamicPath_CB(const msgs::DynamicPath::ConstPtr& msg, const MSG::T_PARAMS & params);
    // ITRIFlagInfo
    void _ITRIFlagInfo_CB(const msgs::Flag_Info::ConstPtr& msg, const MSG::T_PARAMS & params);
    // ITRITransObj
    void _ITRITransObj_CB(const msgs::TransfObj::ConstPtr& msg, const MSG::T_PARAMS & params);
    //---------------------------------------------------------//

}; // end of the class ROS_INTERFACE





#endif
