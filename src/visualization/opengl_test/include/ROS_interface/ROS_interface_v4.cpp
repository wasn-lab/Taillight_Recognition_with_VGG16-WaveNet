#include <ROS_interface_v4.hpp>

// using std::vector;
// using std::string;

#define MAX_NUM_THREAD_FOR_ROS_CB     8 // 6 // Use 6 threads

// Constructors
ROS_INTERFACE::ROS_INTERFACE():
    _is_started(false),
    // _num_topics(0),
    _msg_type_2_topic_params( size_t(MSG::M_TYPE::NUM_MSG_TYPE) ),
    //
    _ref_frame("GUI_map"), _stationary_frame("GUI_map"),
    _is_using_latest_tf_common_update_time(false),
    _latest_tf_common_update_time(ros::Time(0)),
    _current_slice_time(), _global_delay(0.1f)
{
    //
    _num_ros_cb_thread = MAX_NUM_THREAD_FOR_ROS_CB;
    //
    // ros::init(argc, argv, "ROS_interface", ros::init_options::AnonymousName);
    // Remember to call setup_node()
}
ROS_INTERFACE::ROS_INTERFACE(int argc, char **argv):
    _is_started(false),
    // _num_topics(0),
    _msg_type_2_topic_params( size_t(MSG::M_TYPE::NUM_MSG_TYPE) ),
    //
    _ref_frame("GUI_map"), _stationary_frame("GUI_map"),
    _is_using_latest_tf_common_update_time(false),
    _latest_tf_common_update_time(ros::Time(0)),
    _current_slice_time(), _global_delay(0.1f)
{
    //
    _num_ros_cb_thread = MAX_NUM_THREAD_FOR_ROS_CB;
    //
    ros::init(argc, argv, "ROS_interface", ros::init_options::AnonymousName);
}
// Destructor
ROS_INTERFACE::~ROS_INTERFACE(){
    for (size_t i=0; i < _image_subscriber_list.size(); ++i){
        _image_subscriber_list[i].shutdown();
    }
}




bool ROS_INTERFACE::setup_node(int argc, char **argv, std::string node_name_in){
    ros::init(argc, argv, node_name_in.c_str(), ros::init_options::AnonymousName);
    return true;
}
// Setting up topics
//----------------------------------------------------------------//
// Method 1: use add_a_topic to add a single topic sequentially one at a time
bool ROS_INTERFACE::add_a_topic(
    const std::string &name_in,
    int type_in,
    bool is_input_in,
    size_t ROS_queue_in,
    size_t buffer_length_in,
    std::string frame_id_in,
    bool is_transform_in,
    std::string to_frame_in
)
{
    // Add a topic
    size_t idx_new = _topic_param_list.size();
    _topic_param_list.push_back( MSG::T_PARAMS(name_in, type_in, is_input_in, ROS_queue_in, buffer_length_in, idx_new, frame_id_in, is_transform_in, to_frame_in) );
    // Parsing parameters
    //----------------------------//
    // _num_topics = _topic_param_list.size();
    // get topic_type_id and store them in separated arrays
    size_t i = _topic_param_list.size()-1;
    _msg_type_2_topic_params[ _topic_param_list[i].type ].push_back( _topic_param_list[i] );
    _topic_tid_list.push_back( _msg_type_2_topic_params[ _topic_param_list[i].type ].size() - 1 );
    //----------------------------//
    return true;
}
// Method 2: use add_a_topic to add a single topic specific to topic_id
bool ROS_INTERFACE::add_a_topic(
    int topic_id_in,
    const std::string &name_in,
    int type_in,
    bool is_input_in,
    size_t ROS_queue_in,
    size_t buffer_length_in,
    std::string frame_id_in,
    bool is_transform_in,
    std::string to_frame_in
)
{
    // Add a topic
    size_t idx_new = topic_id_in;
    if ( idx_new >= _topic_param_list.size() ){
        _topic_param_list.resize(idx_new+1, MSG::T_PARAMS() );
        _topic_tid_list.resize(idx_new+1, -1 );
    }
    _topic_param_list[idx_new] = ( MSG::T_PARAMS(name_in, type_in, is_input_in, ROS_queue_in, buffer_length_in, idx_new, frame_id_in, is_transform_in, to_frame_in) );
    // Parsing parameters
    //----------------------------//
    // _num_topics = _topic_param_list.size();
    // get topic_type_id and store them in separated arrays
    size_t i = idx_new; // _topic_param_list.size()-1;
    _msg_type_2_topic_params[ _topic_param_list[i].type ].push_back( _topic_param_list[i] );
    _topic_tid_list[idx_new] = _msg_type_2_topic_params[ _topic_param_list[i].type ].size() - 1;
    //----------------------------//
    return true;
}
// Method 3: use load_topics to load all topics
bool ROS_INTERFACE::load_topics(const std::vector<MSG::T_PARAMS> &topic_param_list_in){
    // Filling the dataset inside the object
    // Note: Do not subscribe/advertise topic now
    //----------------------------//
    // Saving the parameters
    _topic_param_list = topic_param_list_in;
    // Parsing parameters
    //----------------------------//
    // _num_topics = _topic_param_list.size();
    // get topic_type_id and store them in separated arrays
    for(size_t i=0; i < _topic_param_list.size(); ++i){
        // Assign the topic_id
        _topic_param_list[i].topic_id = i;
        //
        _msg_type_2_topic_params[ _topic_param_list[i].type ].push_back( _topic_param_list[i] );
        _topic_tid_list.push_back( _msg_type_2_topic_params[ _topic_param_list[i].type ].size() - 1 );
    }
    //----------------------------//
    return true;
}
//----------------------------------------------------------------//

// Really start the ROS thread
bool ROS_INTERFACE::start(){
    if (_is_started) return false; // We don't restart it again (which will result in multiple node, actually)
    // Start the ROS thread, really starting the ROS
    _thread_list.push_back( std::thread(&ROS_INTERFACE::_ROS_worker, this) );
    // _is_started = true;
    TIME_STAMP::Time _sleep_duration(0.2);
    while(!_is_started){
        // std::this_thread::sleep_for( std::chrono::milliseconds(200) );
        _sleep_duration.sleep();
    }
    return true;
}
bool ROS_INTERFACE::is_running(){
    if (!_is_started)
        return false;
    // else
    return ros::ok();
}

// Private methods
//---------------------------------------------//
void ROS_INTERFACE::_ROS_worker(){
    // This thread is called by start()


    // Handle with default namespace
    ros::NodeHandle _nh;
    // ROS image transport (similar to  node handle, but for images)
    image_transport::ImageTransport _ros_it(_nh);

    // Subscribing topics: generate SPSC buffers,  generate _pub_subs_id_list, subscribe
    // Note: the order of the above processes is important, since that the callback function should be exposed only when all the variables are set
    //----------------------------------//
    // Advertising topics: generate SPSC buffers, generated _pub_subs_id_list, advertise
    // Note: the order of the above processes is important, since that the callback function should be exposed only when all the variables are set
    //----------------------------------//
    // id: -1 means not assigned
    _pub_subs_id_list.resize(_topic_param_list.size(), -1);
    //
    int _msg_type = 0;
    //

    // Resize the SPSC buffers
    // async_buffer_list.resize( _topic_param_list.size() );
    async_buffer_list.resize( _topic_param_list.size() );
    //

    // Bool
    _msg_type = int(MSG::M_TYPE::Bool);
    for (size_t _tid=0; _tid < _msg_type_2_topic_params[_msg_type].size(); ++_tid){
        MSG::T_PARAMS _tmp_params = _msg_type_2_topic_params[_msg_type][_tid];
        // SPSC Buffer
        async_buffer_list[_tmp_params.topic_id].reset( new async_buffer< bool > (_tmp_params.buffer_length) );
        //
        // subs_id, pub_id
        if (_tmp_params.is_input){
            // Subscribe
            _pub_subs_id_list[_tmp_params.topic_id] = _subscriber_list.size();
            _subscriber_list.push_back( _nh.subscribe<std_msgs::Bool>( _tmp_params.name, _tmp_params.ROS_queue, boost::bind(&ROS_INTERFACE::_Bool_CB, this, _1, _tmp_params)  ) );
        }else{
            // Publish
            _pub_subs_id_list[_tmp_params.topic_id] = _publisher_list.size();
            _publisher_list.push_back( _nh.advertise<std_msgs::Bool>( _tmp_params.name, _tmp_params.ROS_queue) );
        }
    }
    // Int32
    _msg_type = int(MSG::M_TYPE::Int32);
    for (size_t _tid=0; _tid < _msg_type_2_topic_params[_msg_type].size(); ++_tid){
        MSG::T_PARAMS _tmp_params = _msg_type_2_topic_params[_msg_type][_tid];
        // SPSC Buffer
        async_buffer_list[_tmp_params.topic_id].reset( new async_buffer< long long > (_tmp_params.buffer_length) );
        //
        // subs_id, pub_id
        if (_tmp_params.is_input){
            // Subscribe
            _pub_subs_id_list[_tmp_params.topic_id] = _subscriber_list.size();
            _subscriber_list.push_back( _nh.subscribe<std_msgs::Int32>( _tmp_params.name, _tmp_params.ROS_queue, boost::bind(&ROS_INTERFACE::_Int32_CB, this, _1, _tmp_params)  ) );
        }else{
            // Publish
            _pub_subs_id_list[_tmp_params.topic_id] = _publisher_list.size();
            _publisher_list.push_back( _nh.advertise<std_msgs::Int32>( _tmp_params.name, _tmp_params.ROS_queue) );
        }
    }
    // String
    _msg_type = int(MSG::M_TYPE::String);
    for (size_t _tid=0; _tid < _msg_type_2_topic_params[_msg_type].size(); ++_tid){
        MSG::T_PARAMS _tmp_params = _msg_type_2_topic_params[_msg_type][_tid];
        // SPSC Buffer
        async_buffer_list[_tmp_params.topic_id].reset( new async_buffer< std::string > (_tmp_params.buffer_length) );
        //
        // subs_id, pub_id
        if (_tmp_params.is_input){
            // Subscribe
            _pub_subs_id_list[_tmp_params.topic_id] = _subscriber_list.size();
            _subscriber_list.push_back( _nh.subscribe<std_msgs::String>( _tmp_params.name, _tmp_params.ROS_queue, boost::bind(&ROS_INTERFACE::_String_CB, this, _1, _tmp_params)  ) );
        }else{
            // Publish
            _pub_subs_id_list[_tmp_params.topic_id] = _publisher_list.size();
            _publisher_list.push_back( _nh.advertise<std_msgs::String>( _tmp_params.name, _tmp_params.ROS_queue) );
        }
    }
    // GUI2_op
    _msg_type = int(MSG::M_TYPE::GUI2_op);
    for (size_t _tid=0; _tid < _msg_type_2_topic_params[_msg_type].size(); ++_tid){
        MSG::T_PARAMS _tmp_params = _msg_type_2_topic_params[_msg_type][_tid];
        // SPSC Buffer
        async_buffer_list[_tmp_params.topic_id].reset( new async_buffer< opengl_test::GUI2_op > (_tmp_params.buffer_length) );
        //
        // subs_id, pub_id
        if (_tmp_params.is_input){
            // Subscribe
            _pub_subs_id_list[_tmp_params.topic_id] = _subscriber_list.size();
            _subscriber_list.push_back( _nh.subscribe< opengl_test::GUI2_op >( _tmp_params.name, _tmp_params.ROS_queue, boost::bind(&ROS_INTERFACE::_GUI2_op_CB, this, _1, _tmp_params)  ) );
        }else{
            // Publish
            _pub_subs_id_list[_tmp_params.topic_id] = _publisher_list.size();
            _publisher_list.push_back( _nh.advertise< opengl_test::GUI2_op >( _tmp_params.name, _tmp_params.ROS_queue) );
        }
    }

    // tfGeoPoseStamped
    _msg_type = int(MSG::M_TYPE::tfGeoPoseStamped);
    for (size_t _tid=0; _tid < _msg_type_2_topic_params[_msg_type].size(); ++_tid){
        MSG::T_PARAMS _tmp_params = _msg_type_2_topic_params[_msg_type][_tid];
        // SPSC Buffer
        // Note: tf topic don't go through SPSC buffer, they go through tf2 buffer.
        async_buffer_list[_tmp_params.topic_id].reset( new async_buffer< geometry_msgs::PoseStamped > (_tmp_params.buffer_length) );
        //
        // subs_id, pub_id
        if (_tmp_params.is_input){
            // Subscribe
            _pub_subs_id_list[_tmp_params.topic_id] = _subscriber_list.size();
            _subscriber_list.push_back( _nh.subscribe<geometry_msgs::PoseStamped>( _tmp_params.name, _tmp_params.ROS_queue, boost::bind(&ROS_INTERFACE::_tfGeoPoseStamped_CB, this, _1, _tmp_params)  ) );
        }else{
            // Publish
            _pub_subs_id_list[_tmp_params.topic_id] = _publisher_list.size();
            _publisher_list.push_back( _nh.advertise<geometry_msgs::PoseStamped>( _tmp_params.name, _tmp_params.ROS_queue) );
        }
    }

    // Image
    _msg_type = int(MSG::M_TYPE::Image);
    for (size_t _tid=0; _tid < _msg_type_2_topic_params[_msg_type].size(); ++_tid){
        MSG::T_PARAMS _tmp_params = _msg_type_2_topic_params[_msg_type][_tid];
        // SPSC buffer
        {
            std::shared_ptr< async_buffer<cv::Mat> > _tmpcv_buff_ptr( new async_buffer<cv::Mat>(_tmp_params.buffer_length) );
            _tmpcv_buff_ptr->assign_copy_func(&_cv_Mat_copy_func);
            async_buffer_list[_tmp_params.topic_id] = _tmpcv_buff_ptr;
        }
        //
        // subs_id, pub_id
        if (_tmp_params.is_input){
            // Subscribe
            _pub_subs_id_list[_tmp_params.topic_id] = _image_subscriber_list.size();
            _image_subscriber_list.push_back( _ros_it.subscribe( _tmp_params.name, _tmp_params.ROS_queue, boost::bind(&ROS_INTERFACE::_Image_CB, this, _1, _tmp_params)  ) );
        }else{
            // Publish
            _pub_subs_id_list[_tmp_params.topic_id] = _image_publisher_list.size();
            _image_publisher_list.push_back( _ros_it.advertise( _tmp_params.name, _tmp_params.ROS_queue) );
        }
    }

    // CompressedImageROSIT
    _msg_type = int(MSG::M_TYPE::CompressedImageROSIT);
    for (size_t _tid=0; _tid < _msg_type_2_topic_params[_msg_type].size(); ++_tid){
        MSG::T_PARAMS _tmp_params = _msg_type_2_topic_params[_msg_type][_tid];
        // SPSC Buffer
        {
            std::shared_ptr< async_buffer<cv::Mat> > _tmpcv_buff_ptr( new async_buffer<cv::Mat>(_tmp_params.buffer_length) );
            _tmpcv_buff_ptr->assign_copy_func(&_cv_Mat_copy_func);
            async_buffer_list[_tmp_params.topic_id] = _tmpcv_buff_ptr;
        }
        //
        // subs_id, pub_id
        if (_tmp_params.is_input){
            // Subscribe
            _pub_subs_id_list[_tmp_params.topic_id] = _subscriber_list.size();
            _image_subscriber_list.push_back(
                _ros_it.subscribe( _tmp_params.name, _tmp_params.ROS_queue,
                    boost::bind(&ROS_INTERFACE::_CompressedImageROSIT_CB, this, _1, _tmp_params),
                    ros::VoidPtr(), image_transport::TransportHints("compressed")
                )
            );
        }else{
            // Publish
            _pub_subs_id_list[_tmp_params.topic_id] = _publisher_list.size();
            _image_publisher_list.push_back( _ros_it.advertise( _tmp_params.name, _tmp_params.ROS_queue) );
        }
    }

    // CompressedImageJpegOnly
    _msg_type = int(MSG::M_TYPE::CompressedImageJpegOnly);
    // Resize the input tmp buffer
    _cv_Mat_tmp_ptr_list.resize(_msg_type_2_topic_params[_msg_type].size());
    //
    for (size_t _tid=0; _tid < _msg_type_2_topic_params[_msg_type].size(); ++_tid){
        MSG::T_PARAMS _tmp_params = _msg_type_2_topic_params[_msg_type][_tid];
        // SPSC Buffer
        {
            std::shared_ptr< async_buffer<cv::Mat> > _tmpcv_buff_ptr( new async_buffer<cv::Mat>(_tmp_params.buffer_length) );
            _tmpcv_buff_ptr->assign_copy_func(&_cv_Mat_copy_func);
            async_buffer_list[_tmp_params.topic_id] = _tmpcv_buff_ptr;
        }
        //
        // subs_id, pub_id
        if (_tmp_params.is_input){
            // Subscribe
            _pub_subs_id_list[_tmp_params.topic_id] = _subscriber_list.size();
            _subscriber_list.push_back( _nh.subscribe< sensor_msgs::CompressedImage >( _addCompressedToTopicName(_tmp_params.name), _tmp_params.ROS_queue, boost::bind(&ROS_INTERFACE::_CompressedImageJpegOnly_CB, this, _1, _tmp_params)  ) );
        }else{
            // Publish
            _pub_subs_id_list[_tmp_params.topic_id] = _publisher_list.size();
            _publisher_list.push_back( _nh.advertise< sensor_msgs::CompressedImage >( _addCompressedToTopicName(_tmp_params.name), _tmp_params.ROS_queue) );
        }
    }

    // PointCloud2
    _msg_type = int(MSG::M_TYPE::PointCloud2);
    // Resize the input tmp buffer
    _PointCloud2_tmp_ptr_list.resize(_msg_type_2_topic_params[_msg_type].size());
    //
    for (size_t _tid=0; _tid < _msg_type_2_topic_params[_msg_type].size(); ++_tid){
        MSG::T_PARAMS _tmp_params = _msg_type_2_topic_params[_msg_type][_tid];
        // SPSC Buffer
        async_buffer_list[_tmp_params.topic_id].reset( new async_buffer< pcl::PointCloud<pcl::PointXYZI> > (_tmp_params.buffer_length) );
        //
        // subs_id, pub_id
        if (_tmp_params.is_input){
            // Subscribe
            _pub_subs_id_list[_tmp_params.topic_id] = _subscriber_list.size();
            _subscriber_list.push_back( _nh.subscribe<pcl::PCLPointCloud2>( _tmp_params.name, _tmp_params.ROS_queue, boost::bind(&ROS_INTERFACE::_PointCloud2_CB, this, _1, _tmp_params)  ) );
        }else{
            // Publish
            _pub_subs_id_list[_tmp_params.topic_id] = _publisher_list.size();
            _publisher_list.push_back( _nh.advertise<pcl::PCLPointCloud2>( _tmp_params.name, _tmp_params.ROS_queue) );
        }
    }

    // ITRIPointCloud
    _msg_type = int(MSG::M_TYPE::ITRIPointCloud);
    // Resize the input tmp buffer
    _ITRIPointCloud_tmp_ptr_list.resize(_msg_type_2_topic_params[_msg_type].size());
    //
    for (size_t _tid=0; _tid < _msg_type_2_topic_params[_msg_type].size(); ++_tid){
        MSG::T_PARAMS _tmp_params = _msg_type_2_topic_params[_msg_type][_tid];
        // SPSC Buffer
        async_buffer_list[_tmp_params.topic_id].reset( new async_buffer< pcl::PointCloud<pcl::PointXYZI> > (_tmp_params.buffer_length) );
        //
        // subs_id, pub_id
        if (_tmp_params.is_input){
            // Subscribe
            _pub_subs_id_list[_tmp_params.topic_id] = _subscriber_list.size();
            _subscriber_list.push_back( _nh.subscribe<msgs::PointCloud>( _tmp_params.name, _tmp_params.ROS_queue, boost::bind(&ROS_INTERFACE::_ITRIPointCloud_CB, this, _1, _tmp_params)  ) );
        }else{
            // Publish
            _pub_subs_id_list[_tmp_params.topic_id] = _publisher_list.size();
            _publisher_list.push_back( _nh.advertise<msgs::PointCloud>( _tmp_params.name, _tmp_params.ROS_queue) );
        }
    }

    // ITRI3DBoundingBox
    _msg_type = int(MSG::M_TYPE::ITRI3DBoundingBox);
    for (size_t _tid=0; _tid < _msg_type_2_topic_params[_msg_type].size(); ++_tid){
        MSG::T_PARAMS _tmp_params = _msg_type_2_topic_params[_msg_type][_tid];
        // SPSC Buffer
        // async_buffer_list[_tmp_params.topic_id].reset( new async_buffer< msgs::LidRoi > (_tmp_params.buffer_length) );
        async_buffer_list[_tmp_params.topic_id].reset( new async_buffer< msgs::DetectedObjectArray > (_tmp_params.buffer_length) );
        //
        // subs_id, pub_id
        if (_tmp_params.is_input){
            // Subscribe
            _pub_subs_id_list[_tmp_params.topic_id] = _subscriber_list.size();
            _subscriber_list.push_back( _nh.subscribe<msgs::LidRoi>( _tmp_params.name, _tmp_params.ROS_queue, boost::bind(&ROS_INTERFACE::_ITRI3DBoundingBox_CB, this, _1, _tmp_params)  ) );
        }else{
            // Publish
            _pub_subs_id_list[_tmp_params.topic_id] = _publisher_list.size();
            _publisher_list.push_back( _nh.advertise<msgs::LidRoi>( _tmp_params.name, _tmp_params.ROS_queue) );
        }
    }

    // ITRICamObj
    _msg_type = int(MSG::M_TYPE::ITRICamObj);
    for (size_t _tid=0; _tid < _msg_type_2_topic_params[_msg_type].size(); ++_tid){
        MSG::T_PARAMS _tmp_params = _msg_type_2_topic_params[_msg_type][_tid];
        // SPSC Buffer
        // async_buffer_list[_tmp_params.topic_id].reset( new async_buffer< msgs::CamObj > (_tmp_params.buffer_length) );
        async_buffer_list[_tmp_params.topic_id].reset( new async_buffer< msgs::DetectedObjectArray > (_tmp_params.buffer_length) );
        //
        // subs_id, pub_id
        if (_tmp_params.is_input){
            // Subscribe
            _pub_subs_id_list[_tmp_params.topic_id] = _subscriber_list.size();
            _subscriber_list.push_back( _nh.subscribe< msgs::CamObj >( _tmp_params.name, _tmp_params.ROS_queue, boost::bind(&ROS_INTERFACE::_ITRICamObj_CB, this, _1, _tmp_params)  ) );
        }else{
            // Publish
            _pub_subs_id_list[_tmp_params.topic_id] = _publisher_list.size();
            _publisher_list.push_back( _nh.advertise< msgs::CamObj >( _tmp_params.name, _tmp_params.ROS_queue) );
        }
    }

    // ITRIDetectedObjectArray
    _msg_type = int(MSG::M_TYPE::ITRIDetectedObjectArray);
    for (size_t _tid=0; _tid < _msg_type_2_topic_params[_msg_type].size(); ++_tid){
        MSG::T_PARAMS _tmp_params = _msg_type_2_topic_params[_msg_type][_tid];
        // SPSC Buffer
        async_buffer_list[_tmp_params.topic_id].reset( new async_buffer< msgs::DetectedObjectArray > (_tmp_params.buffer_length) );
        //
        // subs_id, pub_id
        if (_tmp_params.is_input){
            // Subscribe
            _pub_subs_id_list[_tmp_params.topic_id] = _subscriber_list.size();
            _subscriber_list.push_back( _nh.subscribe< msgs::DetectedObjectArray >( _tmp_params.name, _tmp_params.ROS_queue, boost::bind(&ROS_INTERFACE::_ITRIDetectedObjectArray_CB, this, _1, _tmp_params)  ) );
        }else{
            // Publish
            _pub_subs_id_list[_tmp_params.topic_id] = _publisher_list.size();
            _publisher_list.push_back( _nh.advertise< msgs::DetectedObjectArray >( _tmp_params.name, _tmp_params.ROS_queue) );
        }
    }

    // ITRICarInfoCarA
    _msg_type = int(MSG::M_TYPE::ITRICarInfoCarA);
    for (size_t _tid=0; _tid < _msg_type_2_topic_params[_msg_type].size(); ++_tid){
        MSG::T_PARAMS _tmp_params = _msg_type_2_topic_params[_msg_type][_tid];
        // SPSC Buffer
        // async_buffer_list[_tmp_params.topic_id].reset( new async_buffer< msgs::TaichungVehInfo > (_tmp_params.buffer_length) );
        async_buffer_list[_tmp_params.topic_id].reset( new async_buffer< msgs::VehInfo > (_tmp_params.buffer_length) );
        //
        // subs_id, pub_id
        if (_tmp_params.is_input){
            // Subscribe
            _pub_subs_id_list[_tmp_params.topic_id] = _subscriber_list.size();
            _subscriber_list.push_back( _nh.subscribe< msgs::TaichungVehInfo >( _tmp_params.name, _tmp_params.ROS_queue, boost::bind(&ROS_INTERFACE::_ITRICarInfoCarA_CB, this, _1, _tmp_params)  ) );
        }else{
            // Publish
            _pub_subs_id_list[_tmp_params.topic_id] = _publisher_list.size();
            _publisher_list.push_back( _nh.advertise< msgs::TaichungVehInfo >( _tmp_params.name, _tmp_params.ROS_queue) );
        }
    }

    // ITRICarInfo
    _msg_type = int(MSG::M_TYPE::ITRICarInfo);
    for (size_t _tid=0; _tid < _msg_type_2_topic_params[_msg_type].size(); ++_tid){
        MSG::T_PARAMS _tmp_params = _msg_type_2_topic_params[_msg_type][_tid];
        // SPSC Buffer
        async_buffer_list[_tmp_params.topic_id].reset( new async_buffer< msgs::VehInfo > (_tmp_params.buffer_length) );
        //
        // subs_id, pub_id
        if (_tmp_params.is_input){
            // Subscribe
            _pub_subs_id_list[_tmp_params.topic_id] = _subscriber_list.size();
            _subscriber_list.push_back( _nh.subscribe< msgs::VehInfo >( _tmp_params.name, _tmp_params.ROS_queue, boost::bind(&ROS_INTERFACE::_ITRICarInfo_CB, this, _1, _tmp_params)  ) );
        }else{
            // Publish
            _pub_subs_id_list[_tmp_params.topic_id] = _publisher_list.size();
            _publisher_list.push_back( _nh.advertise< msgs::VehInfo >( _tmp_params.name, _tmp_params.ROS_queue) );
        }
    }

    // ITRIDynamicPath
    _msg_type = int(MSG::M_TYPE::ITRIDynamicPath);
    for (size_t _tid=0; _tid < _msg_type_2_topic_params[_msg_type].size(); ++_tid){
        MSG::T_PARAMS _tmp_params = _msg_type_2_topic_params[_msg_type][_tid];
        // SPSC Buffer
        async_buffer_list[_tmp_params.topic_id].reset( new async_buffer< msgs::DynamicPath > (_tmp_params.buffer_length) );
        //
        // subs_id, pub_id
        if (_tmp_params.is_input){
            // Subscribe
            _pub_subs_id_list[_tmp_params.topic_id] = _subscriber_list.size();
            _subscriber_list.push_back( _nh.subscribe< msgs::DynamicPath >( _tmp_params.name, _tmp_params.ROS_queue, boost::bind(&ROS_INTERFACE::_ITRIDynamicPath_CB, this, _1, _tmp_params)  ) );
        }else{
            // Publish
            _pub_subs_id_list[_tmp_params.topic_id] = _publisher_list.size();
            _publisher_list.push_back( _nh.advertise< msgs::DynamicPath >( _tmp_params.name, _tmp_params.ROS_queue) );
        }
    }

    // ITRIFlagInfo
    _msg_type = int(MSG::M_TYPE::ITRIFlagInfo);
    for (size_t _tid=0; _tid < _msg_type_2_topic_params[_msg_type].size(); ++_tid){
        MSG::T_PARAMS _tmp_params = _msg_type_2_topic_params[_msg_type][_tid];
        // SPSC Buffer
        async_buffer_list[_tmp_params.topic_id].reset( new async_buffer< msgs::Flag_Info > (_tmp_params.buffer_length) );
        //
        // subs_id, pub_id
        if (_tmp_params.is_input){
            // Subscribe
            _pub_subs_id_list[_tmp_params.topic_id] = _subscriber_list.size();
            _subscriber_list.push_back( _nh.subscribe< msgs::Flag_Info >( _tmp_params.name, _tmp_params.ROS_queue, boost::bind(&ROS_INTERFACE::_ITRIFlagInfo_CB, this, _1, _tmp_params)  ) );
        }else{
            // Publish
            _pub_subs_id_list[_tmp_params.topic_id] = _publisher_list.size();
            _publisher_list.push_back( _nh.advertise< msgs::Flag_Info >( _tmp_params.name, _tmp_params.ROS_queue) );
        }
    }

    // ITRITransObj
    _msg_type = int(MSG::M_TYPE::ITRITransObj);
    for (size_t _tid=0; _tid < _msg_type_2_topic_params[_msg_type].size(); ++_tid){
        MSG::T_PARAMS _tmp_params = _msg_type_2_topic_params[_msg_type][_tid];
        // SPSC Buffer
        async_buffer_list[_tmp_params.topic_id].reset( new async_buffer< msgs::TransfObj > (_tmp_params.buffer_length) );
        //
        // subs_id, pub_id
        if (_tmp_params.is_input){
            // Subscribe
            _pub_subs_id_list[_tmp_params.topic_id] = _subscriber_list.size();
            _subscriber_list.push_back( _nh.subscribe< msgs::TransfObj >( _tmp_params.name, _tmp_params.ROS_queue, boost::bind(&ROS_INTERFACE::_ITRITransObj_CB, this, _1, _tmp_params)  ) );
        }else{
            // Publish
            _pub_subs_id_list[_tmp_params.topic_id] = _publisher_list.size();
            _publisher_list.push_back( _nh.advertise< msgs::TransfObj >( _tmp_params.name, _tmp_params.ROS_queue) );
        }
    }
    //----------------------------------//



    // Start spinning and loop to the end
    _num_ros_cb_thread = get_count_of_all_topics();
    if (_num_ros_cb_thread > MAX_NUM_THREAD_FOR_ROS_CB){ _num_ros_cb_thread = MAX_NUM_THREAD_FOR_ROS_CB;}
    ros::AsyncSpinner spinner(_num_ros_cb_thread); // Use ? threads
    spinner.start();
    _is_started = true; // The flag for informing the other part of system that the ROS has begun.
    std::cout << "ros_iterface started with [" << _num_ros_cb_thread <<"] threads for callbacks.\n";

    // tf2
    tfListener_ptr.reset( new tf2_ros::TransformListener(tfBuffer) );
    tfBrocaster_ptr.reset( new tf2_ros::TransformBroadcaster() );

    // Loop forever
    ros::waitForShutdown();


/*
    // Loop
    double _loop_rate = 100.0; //1.0;
    long long loop_time_ms = (long long)(1000.0/_loop_rate); // ms
    ros::Rate loop_rate_obj( 1000.0/float(loop_time_ms) ); // Hz
    //
    auto start_old = std::chrono::high_resolution_clock::now();;
    while (ros::ok()){
        //
        auto start = std::chrono::high_resolution_clock::now();

        // Evaluation
        //=============================================================//
        // pub all String
        bool is_published = false;
        is_published |= _String_pub();

        //=============================================================//
        // end Evaluation

        //
        auto elapsed = std::chrono::high_resolution_clock::now() - start;
        auto period = start - start_old;
        start_old = start;

        long long elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
        long long period_us = std::chrono::duration_cast<std::chrono::microseconds>(period).count();


        if (is_published){
            std::cout << "(pub loop) execution time (ms): " << elapsed_us*0.001 << ", ";
            std::cout << "period time error (ms): " << (period_us*0.001 - loop_time_ms) << "\n";
        }

        //
        loop_rate_obj.sleep();
    }// end of Loop
*/


    //
    // _is_started = false;
    std::cout << "End of ros_iterface\n";
}


// Method of time-sync for buffer outputs
//----------------------------------------------//
bool ROS_INTERFACE::update_current_slice_time(){
    _current_slice_time = TIME_STAMP::Time::now() - TIME_STAMP::Time(_global_delay);
    return true;
}
//----------------------------------------------//



// Set the transformation reference for all the transformation
bool ROS_INTERFACE::set_ref_frame(const std::string &ref_frame_in){
    _ref_frame = ref_frame_in;
    return true;
}
bool ROS_INTERFACE::update_latest_tf_common_update_time(const std::string &ref_frame_in, const std::string &to_frame_in){
    ros::Time _common_time;
    std::string err_str;
    tfBuffer._getLatestCommonTime(tfBuffer._lookupFrameNumber(ref_frame_in), tfBuffer._lookupFrameNumber(to_frame_in), _common_time, &err_str);
    // std::cout << "_common_time = " << _common_time.sec << ", " << _common_time.nsec << "\n";
    if (_common_time > _latest_tf_common_update_time){
        _latest_tf_common_update_time = _common_time;
        _is_using_latest_tf_common_update_time = true;
    }
    // std::cout << "err_str = <" << err_str << ">\n";
    return true;
}
ros::Time ROS_INTERFACE::get_latest_tf_common_update_time(){
    return _latest_tf_common_update_time;
}

// Get tf
//------------------------------------------//
bool ROS_INTERFACE::get_tf(std::string base_fram, std::string to_frame, geometry_msgs::TransformStamped & tf_out, bool is_time_traveling, ros::Time lookup_stamp){
    bool is_sucessed = false;
    if (!is_time_traveling){
        if ( !_current_slice_time.is_zero() ){
            try{
                tf_out = tfBuffer.lookupTransform(base_fram, to_frame, toROStime(_current_slice_time));
                is_sucessed = true;
            }catch (tf2::TransformException &ex) {
              ROS_WARN("%s",ex.what());
            }
            //
            if (!is_sucessed){
                try{
                    tf_out = tfBuffer.lookupTransform(base_fram, to_frame, ros::Time(0));
                    is_sucessed = true;
                }catch (tf2::TransformException &ex) {
                    // ROS_WARN("%s",ex.what());
                }
            }
        }else{
            try{
                tf_out = tfBuffer.lookupTransform(base_fram, to_frame, ros::Time(0));
                is_sucessed = true;
            }catch (tf2::TransformException &ex) {
                ROS_WARN("%s",ex.what());
            }
        }



    }else{
        // Using time-traveling going through _stationary_frame
        try{
            ros::Time _common_time;
            if(_is_using_latest_tf_common_update_time){
                _common_time = _latest_tf_common_update_time;
                //
                if (lookup_stamp > _common_time){
                    lookup_stamp = _common_time;
                }
            }else{
                // The representing time will use the _current_slice_time
                _common_time = toROStime(_current_slice_time);
                /*
                std::string err_str;
                tfBuffer._getLatestCommonTime(tfBuffer._lookupFrameNumber(base_fram), tfBuffer._lookupFrameNumber(_stationary_frame), _common_time, &err_str);
                */
            }
            tf_out = tfBuffer.lookupTransform(base_fram, _common_time, to_frame, lookup_stamp, _stationary_frame, ros::Duration(0.2));
            // std::cout << "Got transform\n";
            is_sucessed = true;
        }
        catch (tf2::TransformException &ex) {
            ROS_WARN("%s",ex.what());
        }
    }
    return is_sucessed;
}
geometry_msgs::TransformStamped ROS_INTERFACE::get_tf(std::string base_fram, std::string to_frame, bool & is_sucessed, bool is_time_traveling, ros::Time lookup_stamp){
    geometry_msgs::TransformStamped _tf_out;
    is_sucessed = get_tf(base_fram, to_frame, _tf_out, is_time_traveling, lookup_stamp);
    return _tf_out;
}
bool ROS_INTERFACE::get_tf(std::string at_frame, geometry_msgs::TransformStamped & tf_out, bool is_time_traveling, ros::Time lookup_stamp ){ // _ref_frame --> at_frame
    return get_tf(_ref_frame, at_frame, tf_out, is_time_traveling, lookup_stamp);
}
geometry_msgs::TransformStamped ROS_INTERFACE::get_tf(std::string at_frame, bool & is_sucessed, bool is_time_traveling, ros::Time lookup_stamp ){ // _ref_frame --> at_frame
    geometry_msgs::TransformStamped _tf_out;
    is_sucessed = get_tf(_ref_frame, at_frame, _tf_out, is_time_traveling, lookup_stamp);
    return _tf_out;
}
bool ROS_INTERFACE::get_tf(const int topic_id, geometry_msgs::TransformStamped & tf_out, bool is_time_traveling, ros::Time lookup_stamp){
    // Get the topic param by topic_id
    //------------------------------------//
    MSG::T_PARAMS _param = _topic_param_list[topic_id];
    //------------------------------------//
    return get_tf(_ref_frame, _param.frame_id, tf_out, is_time_traveling, lookup_stamp);
}
geometry_msgs::TransformStamped ROS_INTERFACE::get_tf(const int topic_id, bool & is_sucessed, bool is_time_traveling, ros::Time lookup_stamp){
    // Get the topic param by topic_id
    //------------------------------------//
    MSG::T_PARAMS _param = _topic_param_list[topic_id];
    //------------------------------------//
    return get_tf(_ref_frame, _param.frame_id, is_sucessed, is_time_traveling, lookup_stamp);
}
//------------------------------------------//
// end Get tf



// New interface: boost::any and (void *)
//---------------------------------------------------------//
bool ROS_INTERFACE::get_any_message(const int topic_id, boost::any & content_out_ptr){
    if (!is_topic_id_valid(topic_id)){
        std::cout << "Invalid topic_id at get_any_message()\n";
        return false;
    }
    // front and pop
    return ( async_buffer_list[topic_id]->front_any(content_out_ptr, true, _current_slice_time) );
}
bool ROS_INTERFACE::get_any_message(const int topic_id, boost::any & content_out_ptr, ros::Time &msg_stamp){
    if (!is_topic_id_valid(topic_id)){
        std::cout << "Invalid topic_id at get_any_message()\n";
        return false;
    }
    // front and pop
    bool result = ( async_buffer_list[topic_id]->front_any(content_out_ptr, true, _current_slice_time) );
    msg_stamp = toROStime( async_buffer_list[topic_id]->get_stamp() );
    return result;
}
bool ROS_INTERFACE::get_void_message(const int topic_id, void * content_out_ptr, bool is_shared_ptr){
    // front and pop
    return ( async_buffer_list[topic_id]->front_void( &content_out_ptr, true, _current_slice_time, is_shared_ptr) );
}
bool ROS_INTERFACE::get_void_message(const int topic_id, void * content_out_ptr, ros::Time &msg_stamp, bool is_shared_ptr ){
    // front and pop
    bool result = ( async_buffer_list[topic_id]->front_void( &content_out_ptr, true, _current_slice_time, is_shared_ptr) );
    msg_stamp = toROStime( async_buffer_list[topic_id]->get_stamp() );
    return result;
}
//---------------------------------------------------------//

// Combined same buffer-data types
//---------------------------------------------------------//
bool ROS_INTERFACE::get_any_pointcloud(const int topic_id, pcl::PointCloud<pcl::PointXYZI> & content_out){
    // front and pop
    return ( async_buffer_list[topic_id]->front_void(&content_out, true, _current_slice_time, false) );
}
bool ROS_INTERFACE::get_any_pointcloud(const int topic_id, std::shared_ptr< pcl::PointCloud<pcl::PointXYZI> > & content_out_ptr){
    // front and pop
    return ( async_buffer_list[topic_id]->front_void(&content_out_ptr, true, _current_slice_time, true) );
}
bool ROS_INTERFACE::get_any_pointcloud(const int topic_id, std::shared_ptr< pcl::PointCloud<pcl::PointXYZI> > & content_out_ptr, ros::Time &msg_stamp){
    // front and pop
    bool result = ( async_buffer_list[topic_id]->front_void(&content_out_ptr, true, _current_slice_time, true) );
    msg_stamp = toROStime( async_buffer_list[topic_id]->get_stamp() );
    return result;
}
//---------------------------------------------------------//








// Callbacks and public methods of each message type
//---------------------------------------------------------------//

// Bool
//---------------------------------------------------------------//
// input
void ROS_INTERFACE::_Bool_CB(const std_msgs::Bool::ConstPtr& msg, const MSG::T_PARAMS & params){
    // Time
    TIME_STAMP::Time _time_in(TIME_PARAM::NOW);
    // put
    // Note: the "&(*msg)" thing do the following convertion: boost::shared_ptr --> the object --> memory address
    bool result = async_buffer_list[params.topic_id]->put_void( &(*msg), true, _time_in, false);
    if (!result){ std::cout << params.name << ": buffer full.\n"; }
}
//---------------------------------------------------------------//

// Int32
//---------------------------------------------------------------//
// input
void ROS_INTERFACE::_Int32_CB(const std_msgs::Int32::ConstPtr& msg, const MSG::T_PARAMS & params){
    // Time
    TIME_STAMP::Time _time_in(TIME_PARAM::NOW);
    // put
    // Note: the "&(*msg)" thing do the following convertion: boost::shared_ptr --> the object --> memory address
    bool result = async_buffer_list[params.topic_id]->put_void( &(*msg), true, _time_in, false);
    if (!result){ std::cout << params.name << ": buffer full.\n"; }
}
//---------------------------------------------------------------//

// String
//---------------------------------------------------------------//
// input
void ROS_INTERFACE::_String_CB(const std_msgs::String::ConstPtr& msg, const MSG::T_PARAMS & params){
    // Time
    TIME_STAMP::Time _time_in(TIME_PARAM::NOW);
    std::string _tmp_s;
    _tmp_s = msg->data;
    // put
    bool result = async_buffer_list[params.topic_id]->put_void( &(_tmp_s), true, _time_in, false);
    // bool result = async_buffer_list[params.topic_id]->put_void( &(msg->data), true, _time_in, false);
    if (!result){ std::cout << params.name << ": buffer full.\n"; }
}
bool ROS_INTERFACE::get_String(const int topic_id, std::string & content_out){
    // front and pop
    return ( async_buffer_list[topic_id]->front_void(&content_out, true, _current_slice_time, false) );
}
bool ROS_INTERFACE::get_String(const int topic_id, std::shared_ptr< std::string> & content_out_ptr){
    // front and pop
    return ( async_buffer_list[topic_id]->front_void(&content_out_ptr, true, _current_slice_time, true) );
}
bool ROS_INTERFACE::send_string(const int topic_id, const std::string &content_in){
    // pub_subs_id
    //------------------------------------//
    int _ps_id = _pub_subs_id_list[topic_id];
    //------------------------------------//
    // Content of the message
    std_msgs::String msg;
    msg.data = content_in;
    _publisher_list[ _ps_id ].publish(msg);
    // ROS_INFO("%s", msg.data.c_str());
    return true;
}
//---------------------------------------------------------------//

// GUI2_op
//---------------------------------------------------------------//
// input
void ROS_INTERFACE::_GUI2_op_CB(const opengl_test::GUI2_op::ConstPtr& msg, const MSG::T_PARAMS & params){
    // Time
    TIME_STAMP::Time _time_in(TIME_PARAM::NOW);
    // put
    // Note: the "&(*msg)" thing do the following convertion: boost::shared_ptr --> the object --> memory address
    bool result = async_buffer_list[params.topic_id]->put_void( &(*msg), true, _time_in, false);
    if (!result){ std::cout << params.name << ": buffer full.\n"; }
}
bool ROS_INTERFACE::send_GUI2_op(const int topic_id, const opengl_test::GUI2_op &content_in){
    // pub_subs_id
    //------------------------------------//
    int _ps_id = _pub_subs_id_list[topic_id];
    //------------------------------------//
    // Directly publish the message
    _publisher_list[ _ps_id ].publish(content_in);
    return true;
}
//---------------------------------------------------------------//

// tfGeoPoseStamped
//---------------------------------------------------------------//
void ROS_INTERFACE::_tfGeoPoseStamped_CB(const geometry_msgs::PoseStamped::ConstPtr& msg, const MSG::T_PARAMS & params){
    // Time
    TIME_STAMP::Time _time_in(TIME_PARAM::NOW);
    //
    if(params.is_transform){
        geometry_msgs::TransformStamped _send_tf;
        //
        /*
        ros::Time _now = ros::Time::now();
        ros::Duration _delta = _now - msg->header.stamp;
        std::cout << "_now = " << _now.sec << ", " << _now.nsec << "\n";
        std::cout << "stamp = " << msg->header.stamp.sec << ", " << msg->header.stamp.nsec << "\n";
        std::cout << "_delta = " << _delta.sec << ", " << _delta.nsec << "\n";
        std::cout << "_delta (day) = " << _delta.toSec()/86400.0 << "\n";
        */
        //
        _send_tf.header.stamp = toROStime( _time_in ); // <- The TIME_STAMP::Time::now() is much precise than ros::Time::now(); // msg->header.stamp
        _send_tf.header.frame_id = params.frame_id;
        _send_tf.child_frame_id = params.to_frame;
        _send_tf.transform.translation.x = msg->pose.position.x;
        _send_tf.transform.translation.y = msg->pose.position.y;
        _send_tf.transform.translation.z = msg->pose.position.z;
        _send_tf.transform.rotation.x = msg->pose.orientation.x;
        _send_tf.transform.rotation.y = msg->pose.orientation.y;
        _send_tf.transform.rotation.z = msg->pose.orientation.z;
        _send_tf.transform.rotation.w = msg->pose.orientation.w;
        tfBrocaster_ptr->sendTransform(_send_tf);
        // std::cout << "Recieve the tf\n";

        // Note: Even if the topic is a transform, we still put it into a buffer.
        //       The only thing different is that we are not warning if the buffer full.
        // put
        // Note: the "&(*msg)" thing do the following convertion: boost::shared_ptr --> the object --> memory address
        bool result = async_buffer_list[params.topic_id]->put_void( &(*msg), true, _time_in, false);
        // if (!result){ std::cout << params.name << ": buffer full.\n"; }
        return;
    }
    // else

    // put
    // Note: the "&(*msg)" thing do the following convertion: boost::shared_ptr --> the object --> memory address
    bool result = async_buffer_list[params.topic_id]->put_void( &(*msg), true, _time_in, false);
    if (!result){ std::cout << params.name << ": buffer full.\n"; }
}
//---------------------------------------------------------------//

// Image
//---------------------------------------------------------------//
// input
void ROS_INTERFACE::_Image_CB(const sensor_msgs::ImageConstPtr& msg, const MSG::T_PARAMS & params){
    // Time
    TIME_STAMP::Time _time_in(TIME_PARAM::NOW);

    // Get the (raw) image
    cv_bridge::CvImagePtr cv_ptr;
    try{
      cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
      // std::cout << "image: (rows, cols) = (" << cv_ptr->image.rows << ", " << cv_ptr->image.cols << ")\n";
    }
    catch (cv_bridge::Exception& e){
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }

    /*
    // Use any_ptr, which is actually the fatest method
    boost::any any_ptr;
    {
        std::shared_ptr<cv::Mat> image_ptr = std::make_shared<cv::Mat>(cv_ptr->image);
        // std::cout << "image_ptr.use_count() = " << image_ptr.use_count() << "\n";
        any_ptr = image_ptr;
    }
    bool result = async_buffer_list[params.topic_id]->put_any(any_ptr, true, _time_in, true);
    */

    // test
    // bool result = async_buffer_list[params.topic_id]->put_void( &(cv_ptr->image), true, _time_in, false);


    // put
    bool result = async_buffer_list[params.topic_id]->put_void( &(cv_ptr->image), true, _time_in, false);
    //
    if (!result){
        std::cout << params.name << ": buffer full.\n";
    }
}
bool ROS_INTERFACE::get_Image(const int topic_id, cv::Mat & content_out){
    // front and pop
    return ( async_buffer_list[topic_id]->front_void(&content_out, true, _current_slice_time, false) );
}
bool ROS_INTERFACE::get_Image(const int topic_id, std::shared_ptr<cv::Mat> & content_out_ptr){

    // Use any_ptr, which is actually the fatest method
    /*
    boost::any any_ptr;
    bool result = async_buffer_list[topic_id]->front_any(any_ptr, true, _current_slice_time);
    if (result){
        // content_out_ptr = boost::any_cast< std::shared_ptr< cv::Mat > >( any_ptr );
        std::shared_ptr< cv::Mat > *_ptr_ptr = boost::any_cast< std::shared_ptr< cv::Mat > >( &any_ptr );
        content_out_ptr = *_ptr_ptr;
    }
    return result;
    */

    // front and pop
    return async_buffer_list[topic_id]->front_void(&content_out_ptr, true, _current_slice_time, true);
}
// output
bool ROS_INTERFACE::send_Image(const int topic_id, const cv::Mat &content_in){
    // pub_subs_id
    //------------------------------------//
    int _ps_id = _pub_subs_id_list[topic_id];
    //------------------------------------//
    /*
    // Content of the message
    cv_bridge::CvImagePtr cv_ptr;
    _cv_Mat_copy_func(cv_ptr->image, content_in);
    _image_publisher_list[ _ps_id ].publish(cv_ptr->toImageMsg());
    */

    // Content of the message
    sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", content_in).toImageMsg();
    _image_publisher_list[ _ps_id ].publish(msg);
    //
    return true;
}
//---------------------------------------------------------------//


// CompressedImageROSIT
//---------------------------------------------------------------//
// input
void ROS_INTERFACE::_CompressedImageROSIT_CB(const sensor_msgs::ImageConstPtr& msg, const MSG::T_PARAMS & params){
    // Time
    TIME_STAMP::Time _time_in(TIME_PARAM::NOW);

    // Get the (raw) image
    // cv_bridge::CvImagePtr cv_ptr;
    cv_bridge::CvImageConstPtr cv_ptr;
    try{
      cv_ptr = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::BGR8);
      // std::cout << "image: (rows, cols) = (" << cv_ptr->image.rows << ", " << cv_ptr->image.cols << ")\n";
    }
    catch (cv_bridge::Exception& e){
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }

    // put
    bool result = async_buffer_list[params.topic_id]->put_void( &(cv_ptr->image), true, _time_in, false);
    //
    if (!result){
        std::cout << params.name << ": buffer full.\n";
    }
}
//---------------------------------------------------------------//


// CompressedImageJpegOnly
//---------------------------------------------------------------//
// input
void ROS_INTERFACE::_CompressedImageJpegOnly_CB(const sensor_msgs::CompressedImageConstPtr& msg, const MSG::T_PARAMS & params){
    // Time
    TIME_STAMP::Time _time_in(TIME_PARAM::NOW);

    // Take the reference to _tmp_in_ptr
    std::shared_ptr< cv::Mat > & _tmp_Mat_ptr = _cv_Mat_tmp_ptr_list[ _topic_tid_list[params.topic_id] ]; // tmp Mat
    // if (!_tmp_Mat_ptr){
    //     _tmp_Mat_ptr.reset( new cv::Mat );
    // }

    // cv::Mat image;
    // cv::Mat image_resize;
    try{
        // test
        // TIME_STAMP::Period period_image("image");

        // image = cv::imdecode( (msg->data), cv::IMREAD_UNCHANGED); //convert compressed image data to cv::Mat
        // This caused a seg-fault --> (*_tmp_Mat_ptr) = cv::imdecode( (msg->data), cv::IMREAD_UNCHANGED); //convert compressed image data to cv::Mat
        _tmp_Mat_ptr = std::make_shared<cv::Mat>( cv::imdecode( (msg->data), cv::IMREAD_UNCHANGED) ); //convert compressed image data to cv::Mat
        // cv::resize(
        //     cv::imdecode((msg->data), cv::IMREAD_UNCHANGED),
        //     *_tmp_Mat_ptr,
        //     cv::Size(),
        //     0.33, 0.33,
        //     cv::INTER_LINEAR
        // );

        // period_image.stamp(); period_image.show_msec();
    }
    catch (cv_bridge::Exception& e){
        ROS_ERROR("Could not convert to image!");
    }

    // put
    // bool result = async_buffer_list[params.topic_id]->put_void( &(image), true, _time_in, false);
    bool result = async_buffer_list[params.topic_id]->put_void( &(_tmp_Mat_ptr), true, _time_in, true);
    //
    if (!result){
        std::cout << params.name << ": buffer full.\n";
    }
}
std::string ROS_INTERFACE::_addCompressedToTopicName(std::string name_in){
    std::string key ("compressed");
    std::size_t found = name_in.rfind(key);
    // Note: we assume that the end of topic name does not contain white-space trail.
    if (found == std::string::npos || found != (name_in.size() - key.size()) ){
        // No "compressed" in topic name
        if ( name_in.back() != '/' )
            name_in += "/";
        name_in += "compressed";
        // test
        std::cout << "Fixed topic name (+compressed): [" << name_in << "]\n";
    }
    return name_in;
    //
}
//---------------------------------------------------------------//

// PointCloud2
//---------------------------------------------------------------//
// input
void ROS_INTERFACE::_PointCloud2_CB(const pcl::PCLPointCloud2ConstPtr& msg, const MSG::T_PARAMS & params){
    // Time
    TIME_STAMP::Time _time_in(TIME_PARAM::NOW);

    // Take the reference to _tmp_in_ptr
    std::shared_ptr< pcl::PointCloud<pcl::PointXYZI> > & _tmp_cloud_ptr = _PointCloud2_tmp_ptr_list[ _topic_tid_list[params.topic_id] ]; // tmp cloud
    if (!_tmp_cloud_ptr){
        _tmp_cloud_ptr.reset(new pcl::PointCloud<pcl::PointXYZI>);
    }

    // convert cloud
    // pcl::PCLPointCloud2 pcl_pc;
    // pcl_conversions::toPCL(*msg, pcl_pc);
    pcl::fromPCLPointCloud2(*msg, *_tmp_cloud_ptr);
    // std::cout << "============ Lidar map loaded ============\n  ";
    //

    // Add to buffer
    bool result = async_buffer_list[params.topic_id]->put_void( &(_tmp_cloud_ptr), true, _time_in, true); // <-- It's std::shared_ptr
    //
    if (!result){
        std::cout << params.name << ": buffer full.\n";
    }
}
bool ROS_INTERFACE::get_PointCloud2(const int topic_id, pcl::PointCloud<pcl::PointXYZI> & content_out){
    // front and pop
    return ( async_buffer_list[topic_id]->front_void(&content_out, true, _current_slice_time, false) );
}
bool ROS_INTERFACE::get_PointCloud2(const int topic_id, std::shared_ptr< pcl::PointCloud<pcl::PointXYZI> > & content_out_ptr){
    // front and pop
    return ( async_buffer_list[topic_id]->front_void(&content_out_ptr, true, _current_slice_time, true) );
}
bool ROS_INTERFACE::get_PointCloud2(const int topic_id, std::shared_ptr< pcl::PointCloud<pcl::PointXYZI> > & content_out_ptr, ros::Time &msg_stamp){
    // front and pop
    bool result = ( async_buffer_list[topic_id]->front_void(&content_out_ptr, true, _current_slice_time, true) );
    msg_stamp = toROStime( async_buffer_list[topic_id]->get_stamp() );
    return result;
}
//---------------------------------------------------------------//


// ITRIPointCloud
//---------------------------------------------------------------//
// input
void ROS_INTERFACE::_ITRIPointCloud_CB(const msgs::PointCloud::ConstPtr& msg, const MSG::T_PARAMS & params){
    // Time
    TIME_STAMP::Time _time_in(TIME_PARAM::NOW);

    // tmp cloud
    // Note 1: this have been moved to be a member of the class, so that it won't keep constructing and destructing.
    // Note 2: the pointer is changed to std::shared_ptr instead of the original boost pointer
    // Take the reference to _tmp_in_ptr
    std::shared_ptr< pcl::PointCloud<pcl::PointXYZI> > & _tmp_cloud_ptr = _ITRIPointCloud_tmp_ptr_list[ _topic_tid_list[params.topic_id] ]; // tmp cloud
    if (!_tmp_cloud_ptr){
        _tmp_cloud_ptr.reset(new pcl::PointCloud<pcl::PointXYZI>);
    }
    // std::cout << "(phase 1) _tmp_cloud_ptr->header.seq = " << _tmp_cloud_ptr->header.seq << "\n";
    // Conversion
    //-------------------------//
    _tmp_cloud_ptr->header = pcl_conversions::toPCL( msg->lidHeader );
    _tmp_cloud_ptr->width = msg->pointCloud.size();
    // std::cout << "cloud size = " << _tmp_cloud_ptr->width << "\n";
    _tmp_cloud_ptr->height = 1;
    _tmp_cloud_ptr->is_dense = false;
    //
    // if (_tmp_cloud_ptr->points.size() < msg->pointCloud.size())
        _tmp_cloud_ptr->points.resize( msg->pointCloud.size() );
    // #pragma omp parallel for
    for (long long i = 0; i < _tmp_cloud_ptr->width; ++i)
    {
        _tmp_cloud_ptr->points[i].x = msg->pointCloud[i].x;
        _tmp_cloud_ptr->points[i].y = msg->pointCloud[i].y;
        _tmp_cloud_ptr->points[i].z = msg->pointCloud[i].z;
        _tmp_cloud_ptr->points[i].intensity = msg->pointCloud[i].intensity;
    }
    //-------------------------//
    // std::cout << "lidHeader.seq = " << msg->lidHeader.seq << "\n";
    // Add to buffer
    // std::cout << "(phase 2) _tmp_cloud_ptr->header.seq = " << _tmp_cloud_ptr->header.seq << "\n";
    bool result = async_buffer_list[params.topic_id]->put_void( &(_tmp_cloud_ptr), true, _time_in, true); // <-- It's std::shared_ptr
    // bool result = async_buffer_list[params.topic_id]->put_void( &(_tmp_cloud_ptr), true, _time_in, true); // <-- It's std::shared_ptr
    // std::cout << "(phase 3) _tmp_cloud_ptr->header.seq = " << _tmp_cloud_ptr->header.seq << "\n";

    //
    if (!result){
        std::cout << params.name << ": buffer full.\n";
    }
}
bool ROS_INTERFACE::get_ITRIPointCloud(const int topic_id, pcl::PointCloud<pcl::PointXYZI> & content_out){
    // front and pop
    return ( async_buffer_list[topic_id]->front_void(&content_out, true, _current_slice_time, false) );
}
bool ROS_INTERFACE::get_ITRIPointCloud(const int topic_id, std::shared_ptr< pcl::PointCloud<pcl::PointXYZI> > & content_out_ptr){
    // front and pop
    return ( async_buffer_list[topic_id]->front_void(&content_out_ptr, true, _current_slice_time, true) );
}
bool ROS_INTERFACE::get_ITRIPointCloud(const int topic_id, std::shared_ptr< pcl::PointCloud<pcl::PointXYZI> > & content_out_ptr, ros::Time &msg_stamp){
    // front and pop
    bool result = ( async_buffer_list[topic_id]->front_void(&content_out_ptr, true, _current_slice_time, true) );
    msg_stamp = toROStime( async_buffer_list[topic_id]->get_stamp() );
    return result;
}
bool ROS_INTERFACE::send_ITRIPointCloud(const int topic_id, const pcl::PointCloud<pcl::PointXYZI> &content_in){
    // pub_subs_id
    //------------------------------------//
    int _ps_id = _pub_subs_id_list[topic_id];
    //------------------------------------//

    // Content of the message
    msgs::PointCloud msg;
    // Conversion
    //-------------------------//
    msg.pointCloud.resize( content_in.points.size() );
    // #pragma omp parallel for
    for (size_t i = 0; i < content_in.width; ++i)
    {
        msg.pointCloud[i].x = content_in.points[i].x;
        msg.pointCloud[i].y = content_in.points[i].y;
        msg.pointCloud[i].z = content_in.points[i].z;
        msg.pointCloud[i].intensity = content_in.points[i].intensity;
    }
    //-------------------------//
    _publisher_list[ _ps_id ].publish(msg);
    return true;
}
//---------------------------------------------------------------//


// ITRI3DBoundingBox
//---------------------------------------------------------------//
// input
void ROS_INTERFACE::_ITRI3DBoundingBox_CB(const msgs::LidRoi::ConstPtr& msg, const MSG::T_PARAMS & params){
    // Time
    TIME_STAMP::Time _time_in(TIME_PARAM::NOW);
    // Conversion
    std::shared_ptr<msgs::DetectedObjectArray> _data_ptr(new msgs::DetectedObjectArray() );
    _data_ptr->objects.resize( msg->lidRoiBox.size() );
    for (size_t i=0; i < msg->lidRoiBox.size(); ++i ){
        auto &_data_obj = _data_ptr->objects[i];
        auto &_msg_obj = msg->lidRoiBox[i];
        _data_obj.camInfo.id = i; // Use the sequence number as id
        _data_obj.classId = 0; // _msg_obj.cls;
        //
        _data_obj.bPoint.p0 = _msg_obj.p0;
        _data_obj.bPoint.p1 = _msg_obj.p1;
        _data_obj.bPoint.p2 = _msg_obj.p2;
        _data_obj.bPoint.p3 = _msg_obj.p3;
        _data_obj.bPoint.p4 = _msg_obj.p4;
        _data_obj.bPoint.p5 = _msg_obj.p5;
        _data_obj.bPoint.p6 = _msg_obj.p6;
        _data_obj.bPoint.p7 = _msg_obj.p7;

    }
    // put
    // Note: the "&(*msg)" thing do the following convertion: boost::shared_ptr --> the object --> memory address
    // bool result = async_buffer_list[params.topic_id]->put_void( &(*msg), true, _time_in, false);
    bool result = async_buffer_list[params.topic_id]->put_void( &(*_data_ptr), true, _time_in, false);
    if (!result){ std::cout << params.name << ": buffer full.\n"; }
}
bool ROS_INTERFACE::get_ITRI3DBoundingBox(const int topic_id, msgs::DetectedObjectArray & content_out){
    // front and pop
    return ( async_buffer_list[topic_id]->front_void(&content_out, true, _current_slice_time, false) );
}
bool ROS_INTERFACE::get_ITRI3DBoundingBox(const int topic_id, std::shared_ptr< msgs::DetectedObjectArray > & content_out_ptr){
    // front and pop
    return ( async_buffer_list[topic_id]->front_void(&content_out_ptr, true, _current_slice_time, true) );
}
bool ROS_INTERFACE::get_ITRI3DBoundingBox(const int topic_id, std::shared_ptr< msgs::DetectedObjectArray > & content_out_ptr, ros::Time &msg_stamp){
    // front and pop
    bool result = ( async_buffer_list[topic_id]->front_void(&content_out_ptr, true, _current_slice_time, true) );
    msg_stamp = toROStime( async_buffer_list[topic_id]->get_stamp() );
    return result;
}
bool ROS_INTERFACE::send_ITRI3DBoundingBox(const int topic_id, const msgs::LidRoi &content_in){
    // pub_subs_id
    //------------------------------------//
    int _ps_id = _pub_subs_id_list[topic_id];
    //------------------------------------//
    // Content of the message
    msgs::LidRoi msg;
    msg = content_in;
    _publisher_list[ _ps_id ].publish(msg);
    return true;
}
//---------------------------------------------------------------//

// ITRICamObj
//---------------------------------------------------------------//
// input
void ROS_INTERFACE::_ITRICamObj_CB(const msgs::CamObj::ConstPtr& msg, const MSG::T_PARAMS & params){
    // Time
    TIME_STAMP::Time _time_in(TIME_PARAM::NOW);
    // Convertion
    std::shared_ptr<msgs::DetectedObjectArray> _data_ptr(new msgs::DetectedObjectArray() );
    _data_ptr->objects.resize( msg->camObj.size() );
    for (size_t i=0; i < msg->camObj.size(); ++i ){
        auto &_data_obj = _data_ptr->objects[i];
        auto &_msg_obj = msg->camObj[i];
        _data_obj.camInfo.u = _msg_obj.x;
        _data_obj.camInfo.v = _msg_obj.y;
        _data_obj.camInfo.width = _msg_obj.width;
        _data_obj.camInfo.height = _msg_obj.height;
        _data_obj.camInfo.id = _msg_obj.id;
        _data_obj.camInfo.prob = _msg_obj.prob;
        //
        _data_obj.classId = _msg_obj.cls;
        _data_obj.distance = _msg_obj.distance;
        _data_obj.bPoint = _msg_obj.boxPoint;
        _data_obj.fusionSourceId = _msg_obj.sourceType;
    }
    // put
    // Note: the "&(*msg)" thing do the following convertion: boost::shared_ptr --> the object --> memory address
    // bool result = async_buffer_list[params.topic_id]->put_void( &(*msg), true, _time_in, false);
    bool result = async_buffer_list[params.topic_id]->put_void( &(*_data_ptr), true, _time_in, false);
    if (!result){ std::cout << params.name << ": buffer full.\n"; }
}
//---------------------------------------------------------------//


// ITRIDetectedObjectArray
//---------------------------------------------------------------//
// input
void ROS_INTERFACE::_ITRIDetectedObjectArray_CB(const msgs::DetectedObjectArray::ConstPtr& msg, const MSG::T_PARAMS & params){
    // Time
    TIME_STAMP::Time _time_in(TIME_PARAM::NOW);
    // put
    // Note: the "&(*msg)" thing do the following convertion: boost::shared_ptr --> the object --> memory address
    bool result = async_buffer_list[params.topic_id]->put_void( &(*msg), true, _time_in, false);
    if (!result){ std::cout << params.name << ": buffer full.\n"; }
}
//---------------------------------------------------------------//

// ITRICarInfoCarA
//---------------------------------------------------------------//
// input
void ROS_INTERFACE::_ITRICarInfoCarA_CB(const msgs::TaichungVehInfo::ConstPtr& msg, const MSG::T_PARAMS & params){
    // Time
    TIME_STAMP::Time _time_in(TIME_PARAM::NOW);
    // Conversion
    std::shared_ptr<msgs::VehInfo> _data_ptr(new msgs::VehInfo);
    //
    _data_ptr->ego_x = msg->ego_x;
    _data_ptr->ego_y = msg->ego_y;
    _data_ptr->ego_z = msg->ego_z;
    _data_ptr->ego_heading = msg->ego_heading;
    _data_ptr->ego_speed = msg->ego_speed;
    _data_ptr->yaw_rate = msg->yaw_rate;
    //
    _data_ptr->ukf_ego_x = msg->ukf_ego_x;
    _data_ptr->ukf_ego_y = msg->ukf_ego_y;
    _data_ptr->ukf_ego_heading = msg->ukf_ego_heading;
    _data_ptr->ukf_ego_speed = msg->ukf_ego_speed;
    //
    _data_ptr->road_id = msg->road_id;
    _data_ptr->lanewidth = msg->lanewidth;
    _data_ptr->stoptolane = msg->stoptolane;
    _data_ptr->gps_fault_flag = msg->gps_fault_flag;
    //
    _data_ptr->path1_to_v_x = msg->path1_to_v_x;
    _data_ptr->path1_to_v_y = msg->path1_to_v_y;
    _data_ptr->path2_to_v_x = msg->path2_to_v_x;
    _data_ptr->path2_to_v_y = msg->path2_to_v_y;
    //
    _data_ptr->path3_to_v_x = msg->path3_to_v_x;
    _data_ptr->path3_to_v_y = msg->path3_to_v_y;
    _data_ptr->path4_to_v_x = msg->path4_to_v_x;
    _data_ptr->path4_to_v_y = msg->path4_to_v_y;
    //
    _data_ptr->path5_to_v_x = msg->path5_to_v_x;
    _data_ptr->path5_to_v_y = msg->path5_to_v_y;
    _data_ptr->path6_to_v_x = msg->path6_to_v_x;
    _data_ptr->path6_to_v_y = msg->path6_to_v_y;
    // put
    // Note: the "&(*msg)" thing do the following convertion: boost::shared_ptr --> the object --> memory address
    // bool result = async_buffer_list[params.topic_id]->put_void( &(*msg), true, _time_in, false);
    bool result = async_buffer_list[params.topic_id]->put_void( &(*_data_ptr), true, _time_in, false);
    if (!result){ std::cout << params.name << ": buffer full.\n"; }
}
//---------------------------------------------------------------//

// ITRICarInfo
//---------------------------------------------------------------//
// input
void ROS_INTERFACE::_ITRICarInfo_CB(const msgs::VehInfo::ConstPtr& msg, const MSG::T_PARAMS & params){
    // Time
    TIME_STAMP::Time _time_in(TIME_PARAM::NOW);
    // put
    // Note: the "&(*msg)" thing do the following convertion: boost::shared_ptr --> the object --> memory address
    bool result = async_buffer_list[params.topic_id]->put_void( &(*msg), true, _time_in, false);
    if (!result){ std::cout << params.name << ": buffer full.\n"; }
}
//---------------------------------------------------------------//


// ITRIDynamicPath
//---------------------------------------------------------------//
// input
void ROS_INTERFACE::_ITRIDynamicPath_CB(const msgs::DynamicPath::ConstPtr& msg, const MSG::T_PARAMS & params){
    // Time
    TIME_STAMP::Time _time_in(TIME_PARAM::NOW);
    // put
    // Note: the "&(*msg)" thing do the following convertion: boost::shared_ptr --> the object --> memory address
    bool result = async_buffer_list[params.topic_id]->put_void( &(*msg), true, _time_in, false);
    if (!result){ std::cout << params.name << ": buffer full.\n"; }
}
//---------------------------------------------------------------//

// ITRIFlagInfo
//---------------------------------------------------------------//
// input
void ROS_INTERFACE::_ITRIFlagInfo_CB(const msgs::Flag_Info::ConstPtr& msg, const MSG::T_PARAMS & params){
    // Time
    TIME_STAMP::Time _time_in(TIME_PARAM::NOW);
    // put
    // Note: the "&(*msg)" thing do the following convertion: boost::shared_ptr --> the object --> memory address
    bool result = async_buffer_list[params.topic_id]->put_void( &(*msg), true, _time_in, false);
    if (!result){ std::cout << params.name << ": buffer full.\n"; }
}
//---------------------------------------------------------------//

// ITRITransObj
//---------------------------------------------------------------//
// input
void ROS_INTERFACE::_ITRITransObj_CB(const msgs::TransfObj::ConstPtr& msg, const MSG::T_PARAMS & params){
    // Time
    TIME_STAMP::Time _time_in(TIME_PARAM::NOW);
    // put
    // Note: the "&(*msg)" thing do the following convertion: boost::shared_ptr --> the object --> memory address
    bool result = async_buffer_list[params.topic_id]->put_void( &(*msg), true, _time_in, false);
    if (!result){ std::cout << params.name << ": buffer full.\n"; }
}
//---------------------------------------------------------------//
