#include <ROS_interface_v2.hpp>

// using std::vector;
// using std::string;

#define TOTAL_NUM_THREAD_FOR_ROS_CB     6 // Use 6 threads

// Constructors
ROS_INTERFACE::ROS_INTERFACE():
    _is_started(false),
    _num_topics(0),
    _msg_type_2_topic_params( size_t(MSG::M_TYPE::NUM_MSG_TYPE) ),
    // The temporary containers
    // _ITRIPointCloud_tmp_ptr (new pcl::PointCloud<pcl::PointXYZI>)
    _ref_frame("map"), _stationary_frame("map"),
    _is_using_latest_tf_common_update_time(false),
    _latest_tf_common_update_time(ros::Time(0)),
    _current_slice_time(), _global_delay(0.1f)
{
    //
    _num_ros_cb_thread = TOTAL_NUM_THREAD_FOR_ROS_CB;
    //
    // ros::init(argc, argv, "ROS_interface", ros::init_options::AnonymousName);
    // Remember to call setup_node()
}
ROS_INTERFACE::ROS_INTERFACE(int argc, char **argv):
    _is_started(false),
    _num_topics(0),
    _msg_type_2_topic_params( size_t(MSG::M_TYPE::NUM_MSG_TYPE) ),
    // The temporary containers
    // _ITRIPointCloud_tmp_ptr (new pcl::PointCloud<pcl::PointXYZI>)
    _ref_frame("map"), _stationary_frame("map"),
    _is_using_latest_tf_common_update_time(false),
    _latest_tf_common_update_time(ros::Time(0)),
    _current_slice_time(), _global_delay(0.1f)
{
    //
    _num_ros_cb_thread = TOTAL_NUM_THREAD_FOR_ROS_CB;
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
// Method 1: use add_a_topic to add a single topic one at a time
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
    _num_topics = _topic_param_list.size();
    // get topic_type_id and store them in separated arrays
    size_t i = _num_topics-1;
    _msg_type_2_topic_params[ _topic_param_list[i].type ].push_back( _topic_param_list[i] );
    _topic_tid_list.push_back( _msg_type_2_topic_params[ _topic_param_list[i].type ].size() - 1 );
    //----------------------------//
    return true;
}
// Method 2: use load_topics to load all topics
bool ROS_INTERFACE::load_topics(const std::vector<MSG::T_PARAMS> &topic_param_list_in){
    // Filling the dataset inside the object
    // Note: Do not subscribe/advertise topic now
    //----------------------------//
    // Saving the parameters
    _topic_param_list = topic_param_list_in;
    // Parsing parameters
    //----------------------------//
    _num_topics = _topic_param_list.size();
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
    while(!_is_started){
        std::this_thread::sleep_for( std::chrono::milliseconds(200) );
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


    // test
    buffwr_list.resize( _topic_param_list.size() );

    // test_2
    any_buffer_list.resize( _topic_param_list.size() );


    // String
    _msg_type = int(MSG::M_TYPE::String);
    for (size_t _tid=0; _tid < _msg_type_2_topic_params[_msg_type].size(); ++_tid){
        MSG::T_PARAMS _tmp_params = _msg_type_2_topic_params[_msg_type][_tid];
        // SPSC Buffer
        buffer_list_String.push_back( async_buffer<std::string>( _tmp_params.buffer_length ) );
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

    // tfGeoPoseStamped
    _msg_type = int(MSG::M_TYPE::tfGeoPoseStamped);
    for (size_t _tid=0; _tid < _msg_type_2_topic_params[_msg_type].size(); ++_tid){
        MSG::T_PARAMS _tmp_params = _msg_type_2_topic_params[_msg_type][_tid];
        // SPSC Buffer
        // Note: tf topic don't go through SPSC buffer, they go through tf2 buffer.
        buffer_list_tfGeoPoseStamped.push_back( async_buffer<geometry_msgs::PoseStamped>( _tmp_params.buffer_length ) );
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
        // SPSC Buffer
        buffer_list_Image.push_back( async_buffer<cv::Mat>( _tmp_params.buffer_length ) );
        // Note: assign copy function
        buffer_list_Image[buffer_list_Image.size()-1].assign_copy_func( &ROS_INTERFACE::_cv_Mat_copy_func );

        // test
        buffwr_list[_tmp_params.topic_id].reset(new buffwrImage(_tmp_params.buffer_length) );
        //

        // test_2
        any_buffer_list[_tmp_params.topic_id] = async_buffer<cv::Mat>(_tmp_params.buffer_length);
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

    // PointCloud2
    _msg_type = int(MSG::M_TYPE::PointCloud2);
    for (size_t _tid=0; _tid < _msg_type_2_topic_params[_msg_type].size(); ++_tid){
        MSG::T_PARAMS _tmp_params = _msg_type_2_topic_params[_msg_type][_tid];
        // SPSC Buffer
        buffer_list_PointCloud2.push_back( async_buffer< pcl::PointCloud<pcl::PointXYZI> >( _tmp_params.buffer_length ) );
        // subs_id, pub_id
        if (_tmp_params.is_input){
            // Subscribe
            _pub_subs_id_list[_tmp_params.topic_id] = _subscriber_list.size();
            _subscriber_list.push_back( _nh.subscribe<sensor_msgs::PointCloud2>( _tmp_params.name, _tmp_params.ROS_queue, boost::bind(&ROS_INTERFACE::_PointCloud2_CB, this, _1, _tmp_params)  ) );
        }else{
            // Publish
            _pub_subs_id_list[_tmp_params.topic_id] = _publisher_list.size();
            _publisher_list.push_back( _nh.advertise<sensor_msgs::PointCloud2>( _tmp_params.name, _tmp_params.ROS_queue) );
        }
    }

    // ITRIPointCloud
    _msg_type = int(MSG::M_TYPE::ITRIPointCloud);
    for (size_t _tid=0; _tid < _msg_type_2_topic_params[_msg_type].size(); ++_tid){
        MSG::T_PARAMS _tmp_params = _msg_type_2_topic_params[_msg_type][_tid];
        // SPSC Buffer
        buffer_list_ITRIPointCloud.push_back( async_buffer< pcl::PointCloud<pcl::PointXYZI> >( _tmp_params.buffer_length ) );

        // test
        buffwr_list[_tmp_params.topic_id].reset(new buffwrITRIPointCloud(_tmp_params.buffer_length) );
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
        buffer_list_ITRI3DBoundingBox.push_back( async_buffer<msgs::LidRoi>( _tmp_params.buffer_length ) );
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

    //----------------------------------//



    // Start spinning and loop to the end
    ros::AsyncSpinner spinner(_num_ros_cb_thread); // Use ? threads
    spinner.start();
    _is_started = true; // The flag for informing the other part of system that the ROS has begun.
    std::cout << "ros_iterface started\n";

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


// Combined same buffer-data types
//---------------------------------------------------------//

bool ROS_INTERFACE::get_any_message(const int topic_id, boost::any & content_out_ptr){
    {
        using MSG::M_TYPE;
        // Initialize
        switch ( M_TYPE(_topic_param_list[topic_id].type) ){
            case M_TYPE::String:
                {
                    if (content_out_ptr.empty()){   content_out_ptr =  std::shared_ptr< std::string>(); }
                break;}
            case M_TYPE::tfGeoPoseStamped:
                {
                break;}
            case M_TYPE::Image:
                {
                    if (content_out_ptr.empty()){   content_out_ptr =  std::shared_ptr<cv::Mat>(); }
                break;}
            case M_TYPE::PointCloud2:
                {
                    if (content_out_ptr.empty()){   content_out_ptr =  std::shared_ptr< pcl::PointCloud<pcl::PointXYZI> >(); }
                break;}
            case M_TYPE::ITRIPointCloud:
                {
                    if (content_out_ptr.empty()){   content_out_ptr =  std::shared_ptr< pcl::PointCloud<pcl::PointXYZI> >(); }
                break;}
            case M_TYPE::ITRI3DBoundingBox:
                {
                    if (content_out_ptr.empty()){   content_out_ptr =  std::shared_ptr< msgs::LidRoi >(); }
                break;}
            default:
                // return false;
                break;
        }
        // Get value
        switch ( M_TYPE(_topic_param_list[topic_id].type) ){
            case M_TYPE::String:
                {std::shared_ptr< std::string> *_ptr_ptr = boost::any_cast< std::shared_ptr< std::string> >( &content_out_ptr );
                return get_String(topic_id,  *_ptr_ptr);
                break;}
            case M_TYPE::tfGeoPoseStamped:
                {return false;
                break;}
            case M_TYPE::Image:
                {std::shared_ptr<cv::Mat> *_ptr_ptr = boost::any_cast< std::shared_ptr<cv::Mat> >( &content_out_ptr );
                // std::cout << "*_ptr_ptr.use_count() = " << (*_ptr_ptr).use_count() << "\n"; // NOTE: The result is "1", which is the result we want
                return get_Image(topic_id, *_ptr_ptr );
                break;}
            case M_TYPE::PointCloud2:
                {std::shared_ptr< pcl::PointCloud<pcl::PointXYZI> > *_ptr_ptr = boost::any_cast< std::shared_ptr< pcl::PointCloud<pcl::PointXYZI> > >( &content_out_ptr );
                return get_PointCloud2(topic_id,  *_ptr_ptr);
                break;}
            case M_TYPE::ITRIPointCloud:
                {std::shared_ptr< pcl::PointCloud<pcl::PointXYZI> >  *_ptr_ptr = boost::any_cast< std::shared_ptr< pcl::PointCloud<pcl::PointXYZI> > >( &content_out_ptr );
                return get_ITRIPointCloud(topic_id, *_ptr_ptr );
                break;}
            case M_TYPE::ITRI3DBoundingBox:
                {std::shared_ptr< msgs::LidRoi > *_ptr_ptr = boost::any_cast< std::shared_ptr< msgs::LidRoi > >( &content_out_ptr );
                return get_ITRI3DBoundingBox(topic_id, *_ptr_ptr );
                break;}
            default:
                return false;
        }

        // end switch
    }
}

bool ROS_INTERFACE::get_any_pointcloud(const int topic_id, pcl::PointCloud<pcl::PointXYZI> & content_out){
    {
        using MSG::M_TYPE;
        switch (_topic_param_list[topic_id].type){
            case int(M_TYPE::PointCloud2):
                return get_PointCloud2(topic_id, content_out);
                break;
            case int(M_TYPE::ITRIPointCloud):
                return get_ITRIPointCloud(topic_id, content_out);
                break;
            default:
                return false;
        }
        // end switch
    }
}
bool ROS_INTERFACE::get_any_pointcloud(const int topic_id, std::shared_ptr< pcl::PointCloud<pcl::PointXYZI> > & content_out_ptr){
    {
        using MSG::M_TYPE;
        switch (_topic_param_list[topic_id].type){
            case int(M_TYPE::PointCloud2):
                return get_PointCloud2(topic_id, content_out_ptr);
                break;
            case int(M_TYPE::ITRIPointCloud):
                return get_ITRIPointCloud(topic_id, content_out_ptr);
                break;
            default:
                return false;
        }
        // end switch
    }
}
bool ROS_INTERFACE::get_any_pointcloud(const int topic_id, std::shared_ptr< pcl::PointCloud<pcl::PointXYZI> > & content_out_ptr, ros::Time &msg_stamp){
    {
        using MSG::M_TYPE;
        switch (_topic_param_list[topic_id].type){
            case int(M_TYPE::PointCloud2):
                return get_PointCloud2(topic_id, content_out_ptr, msg_stamp);
                break;
            case int(M_TYPE::ITRIPointCloud):
                return get_ITRIPointCloud(topic_id, content_out_ptr, msg_stamp);
                break;
            default:
                return false;
        }
        // end switch
    }
}
//---------------------------------------------------------//


// Callbacks and public methods of each message type
//---------------------------------------------------------------//

// String
//---------------------------------------------------------------//
// input
void ROS_INTERFACE::_String_CB(const std_msgs::String::ConstPtr& msg, const MSG::T_PARAMS & params){
    // Type_id
    //------------------------------------//
    int _tid = _topic_tid_list[params.topic_id];
    //------------------------------------//
    //
    std::string _tmp_s;
    _tmp_s = msg->data;
    bool result = buffer_list_String[ _tid ].put( _tmp_s);

    if (!result){
        std::cout << params.name << ": buffer full.\n";
    }
}
bool ROS_INTERFACE::get_String(const int topic_id, std::string & content_out){
    // Type_id
    //------------------------------------//
    int _tid = _topic_tid_list[topic_id];
    //------------------------------------//
    return ( buffer_list_String[_tid].front(content_out, true, _current_slice_time) );
}
bool ROS_INTERFACE::get_String(const int topic_id, std::shared_ptr< std::string> & content_out_ptr){
    // Type_id
    //------------------------------------//
    int _tid = _topic_tid_list[topic_id];
    //------------------------------------//
    return ( buffer_list_String[_tid].front(content_out_ptr, true, _current_slice_time) );
}
// output
/*
bool ROS_INTERFACE::_String_pub(){
    bool is_published = false;
    // Loop over
    for (size_t _tid=0; _tid < buffer_list_String.size(); ++_tid){
        if (_msg_type_2_topic_params[int(MSG::M_TYPE::String)][_tid].is_input)
            continue;
        size_t topic_id = _msg_type_2_topic_params[int(MSG::M_TYPE::String)][_tid].topic_id;
        // else, it's output
        std::pair<string,bool> _result_pair = buffer_list_String[_tid].front(true);
        if (_result_pair.second){
            // front and pop
            // Content of the message
            std_msgs::String msg;
            msg.data = _result_pair.first;
            _publisher_list[ _pub_subs_id_list[topic_id] ].publish(msg);
            is_published = true;
            //
            // ROS_INFO("%s", msg.data.c_str());
        }else{
            // empty
        }
        //
    }
    //
    return is_published;
}
*/
bool ROS_INTERFACE::send_string(const int topic_id, const std::string &content_in){
    // pub_subs_id
    //------------------------------------//
    int _ps_id = _pub_subs_id_list[topic_id];
    //------------------------------------//
    // Content of the message
    std_msgs::String msg;
    msg.data = content_in;
    _publisher_list[ _ps_id ].publish(msg);
    //
    // ROS_INFO("%s", msg.data.c_str());

/*
    // Type_id
    //------------------------------------//
    int _tid = _topic_tid_list[topic_id];
    //------------------------------------//
    bool result = buffer_list_String[ _tid ].put( content_in);
    if (!result){
        std::cout << _topic_param_list[topic_id].name << ": buffer full.\n";
    }
*/

}
//---------------------------------------------------------------//


// tfGeoPoseStamped
//---------------------------------------------------------------//
void ROS_INTERFACE::_tfGeoPoseStamped_CB(const geometry_msgs::PoseStamped::ConstPtr& msg, const MSG::T_PARAMS & params){
    // Type_id
    //------------------------------------//
    int _tid = _topic_tid_list[params.topic_id];
    //------------------------------------//
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
        _send_tf.header.stamp = toROStime(TIME_STAMP::Time::now()); // <- The TIME_STAMP::Time::now() is much precise than ros::Time::now(); // msg->header.stamp
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
        return;
    }
    // else
    bool result = buffer_list_tfGeoPoseStamped[ _tid ].put( *msg);
    if (!result){
        std::cout << params.name << ": buffer full.\n";
    }
}
//---------------------------------------------------------------//

// Image
//---------------------------------------------------------------//
// input
void ROS_INTERFACE::_Image_CB(const sensor_msgs::ImageConstPtr& msg, const MSG::T_PARAMS & params){
    // Type_id
    //------------------------------------//
    int _tid = _topic_tid_list[params.topic_id];
    //------------------------------------//

    TIME_STAMP::Time _time_in;
    _time_in.set_now();

    //
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
    // bool result = buffer_list_Image[ _tid ].put( cv_ptr->image, _time_in);

    // test
    /*
    boost::any any_ptr;
    {
        std::shared_ptr<cv::Mat> image_ptr = std::make_shared<cv::Mat>(cv_ptr->image);
        // std::cout << "image_ptr.use_count() = " << image_ptr.use_count() << "\n";
        any_ptr = image_ptr;
    }
    bool result = buffwr_list[params.topic_id]->put_any(any_ptr, true, _time_in, true);
    */

    // test_4
    bool result = buffwr_list[params.topic_id]->put_void( &(cv_ptr->image), true, _time_in, false);


    // test_3
    // bool result = (* boost::any_cast< async_buffer<cv::Mat> >(&any_buffer_list[params.topic_id]) ).put_void( &(cv_ptr->image), true, _time_in, false);

    // test_2
    // bool result = buffer_list_Image[ _tid ].put_void( &(cv_ptr->image), true, _time_in, false);
    //

    // test_2 2
    // std::shared_ptr<cv::Mat> image_ptr = std::make_shared<cv::Mat>(cv_ptr->image);
    // bool result = buffer_list_Image[ _tid ].put_void( &(image_ptr), true, _time_in, true);
    //

    if (!result){
        std::cout << params.name << ": buffer full.\n";
    }
}
bool ROS_INTERFACE::get_Image(const int topic_id, cv::Mat & content_out){
    // Type_id
    //------------------------------------//
    int _tid = _topic_tid_list[topic_id];
    //------------------------------------//
    return ( buffer_list_Image[_tid].front(content_out, true, _current_slice_time) );
}

bool ROS_INTERFACE::get_Image(const int topic_id, std::shared_ptr<cv::Mat> & content_out_ptr){
    // Type_id
    //------------------------------------//
    int _tid = _topic_tid_list[topic_id];
    //------------------------------------//
    // return ( buffer_list_Image[_tid].front(content_out_ptr, true, _current_slice_time) );


    // test
    /*
    boost::any any_ptr;
    bool result = buffwr_list[topic_id]->front_any(any_ptr, true, _current_slice_time);
    if (result){
        // content_out_ptr = boost::any_cast< std::shared_ptr< cv::Mat > >( any_ptr );
        std::shared_ptr< cv::Mat > *_ptr_ptr = boost::any_cast< std::shared_ptr< cv::Mat > >( &any_ptr );
        content_out_ptr = *_ptr_ptr;
    }
    return result;
    */

    // test_4
    return buffwr_list[topic_id]->front_void(&content_out_ptr, true, _current_slice_time, true);



    // test_3
    // return ( (* boost::any_cast< async_buffer<cv::Mat> >(&any_buffer_list[topic_id]) ).front_void(&content_out_ptr, true, _current_slice_time, true));

    // test_2
    // return ( buffer_list_Image[_tid].front_void(&content_out_ptr, true, _current_slice_time, true));
}
// output
bool ROS_INTERFACE::send_Image(const int topic_id, const cv::Mat &content_in){
    // pub_subs_id
    //------------------------------------//
    int _ps_id = _pub_subs_id_list[topic_id];
    //------------------------------------//
    // Content of the message
    cv_bridge::CvImagePtr cv_ptr;
    _cv_Mat_copy_func(cv_ptr->image, content_in);
    _image_publisher_list[ _ps_id ].publish(cv_ptr->toImageMsg());
}
//---------------------------------------------------------------//

// PointCloud2
//---------------------------------------------------------------//
// input
void ROS_INTERFACE::_PointCloud2_CB(const sensor_msgs::PointCloud2::ConstPtr& msg, const MSG::T_PARAMS & params){
    // Type_id
    //------------------------------------//
    int _tid = _topic_tid_list[params.topic_id];
    //------------------------------------//

    TIME_STAMP::Time _time_in;
    _time_in.set_now();

    if (!_PointCloud2_tmp_ptr){
        _PointCloud2_tmp_ptr.reset(new pcl::PointCloud<pcl::PointXYZI>);
    }
    //
    pcl::PCLPointCloud2 pcl_pc;
    pcl_conversions::toPCL(*msg, pcl_pc);
    pcl::fromPCLPointCloud2(pcl_pc, *_PointCloud2_tmp_ptr);
    //
    std::cout << "=== Load LiDAR Map OK!\n  ";

    // Add to buffer
    bool result = buffer_list_PointCloud2[ _tid ].put( _PointCloud2_tmp_ptr, _time_in); // Put in the pointer directly
    //
    if (!result){
        std::cout << params.name << ": buffer full.\n";
    }
}
bool ROS_INTERFACE::get_PointCloud2(const int topic_id, pcl::PointCloud<pcl::PointXYZI> & content_out){
    // Type_id
    //------------------------------------//
    int _tid = _topic_tid_list[topic_id];
    //------------------------------------//
    return ( buffer_list_PointCloud2[_tid].front(content_out, true, _current_slice_time) );
}
bool ROS_INTERFACE::get_PointCloud2(const int topic_id, std::shared_ptr< pcl::PointCloud<pcl::PointXYZI> > & content_out_ptr){
    // Type_id
    //------------------------------------//
    int _tid = _topic_tid_list[topic_id];
    //------------------------------------//
    return ( buffer_list_PointCloud2[_tid].front(content_out_ptr, true, _current_slice_time) );
}
bool ROS_INTERFACE::get_PointCloud2(const int topic_id, std::shared_ptr< pcl::PointCloud<pcl::PointXYZI> > & content_out_ptr, ros::Time &msg_stamp){
    // Advanced method with ros::tf2
    // Type_id
    //------------------------------------//
    int _tid = _topic_tid_list[topic_id];
    //------------------------------------//
    bool result = ( buffer_list_PointCloud2[_tid].front(content_out_ptr, true, _current_slice_time) );
    msg_stamp = toROStime( buffer_list_PointCloud2[_tid].get_stamp() );
    return result;
}
//---------------------------------------------------------------//


// ITRIPointCloud
//---------------------------------------------------------------//
// input
void ROS_INTERFACE::_ITRIPointCloud_CB(const msgs::PointCloud::ConstPtr& msg, const MSG::T_PARAMS & params){
    // Type_id
    //------------------------------------//
    int _tid = _topic_tid_list[params.topic_id];
    //------------------------------------//

    TIME_STAMP::Time _time_in;
    _time_in.set_now();

    // tmp cloud
    // Note 1: this have been moved to be a member of the class, so that it won't keep constructing and destructing.
    // Note 2: the pointer changed to use the std::shared_ptr instead of the original boost pointer

    // pcl::PointCloud<pcl::PointXYZI>::Ptr _ITRIPointCloud_tmp_ptr (new pcl::PointCloud<pcl::PointXYZI>);
    if (!_ITRIPointCloud_tmp_ptr){
        _ITRIPointCloud_tmp_ptr.reset(new pcl::PointCloud<pcl::PointXYZI>);
    }
    // Conversion
    //-------------------------//
    _ITRIPointCloud_tmp_ptr->header = pcl_conversions::toPCL( msg->lidHeader );
    _ITRIPointCloud_tmp_ptr->width = msg->pointCloud.size();
    // std::cout << "cloud size = " << _ITRIPointCloud_tmp_ptr->width << "\n";
    _ITRIPointCloud_tmp_ptr->height = 1;
    _ITRIPointCloud_tmp_ptr->is_dense = false;
    //
    // if (_ITRIPointCloud_tmp_ptr->points.size() < msg->pointCloud.size())
        _ITRIPointCloud_tmp_ptr->points.resize( msg->pointCloud.size() );
    // #pragma omp parallel for
    for (long long i = 0; i < _ITRIPointCloud_tmp_ptr->width; ++i)
    {
        _ITRIPointCloud_tmp_ptr->points[i].x = msg->pointCloud[i].x;
        _ITRIPointCloud_tmp_ptr->points[i].y = msg->pointCloud[i].y;
        _ITRIPointCloud_tmp_ptr->points[i].z = msg->pointCloud[i].z;
        _ITRIPointCloud_tmp_ptr->points[i].intensity = msg->pointCloud[i].intensity;
    }
    //-------------------------//
    // std::cout << "lidHeader.seq = " << msg->lidHeader.seq << "\n";
    // Add to buffer
    // // bool result = buffer_list_ITRIPointCloud[ _tid ].put( *_ITRIPointCloud_tmp_ptr);
    // bool result = buffer_list_ITRIPointCloud[ _tid ].put( _ITRIPointCloud_tmp_ptr, _time_in); // Put in the pointer directly
    //

    // test
    bool result = buffwr_list[params.topic_id]->put_void( &(_ITRIPointCloud_tmp_ptr), true, _time_in, true);
    //

    if (!result){
        std::cout << params.name << ": buffer full.\n";
    }
}
bool ROS_INTERFACE::get_ITRIPointCloud(const int topic_id, pcl::PointCloud<pcl::PointXYZI> & content_out){
    // Type_id
    //------------------------------------//
    int _tid = _topic_tid_list[topic_id];
    //------------------------------------//
    return ( buffer_list_ITRIPointCloud[_tid].front(content_out, true, _current_slice_time) );
}
bool ROS_INTERFACE::get_ITRIPointCloud(const int topic_id, std::shared_ptr< pcl::PointCloud<pcl::PointXYZI> > & content_out_ptr){
    // Type_id
    //------------------------------------//
    int _tid = _topic_tid_list[topic_id];
    //------------------------------------//
    return ( buffer_list_ITRIPointCloud[_tid].front(content_out_ptr, true, _current_slice_time) );
}
bool ROS_INTERFACE::get_ITRIPointCloud(const int topic_id, std::shared_ptr< pcl::PointCloud<pcl::PointXYZI> > & content_out_ptr, ros::Time &msg_stamp){
    // Advanced method with ros::tf2
    // Type_id
    //------------------------------------//
    int _tid = _topic_tid_list[topic_id];
    //------------------------------------//
    // bool result = ( buffer_list_ITRIPointCloud[_tid].front(content_out_ptr, true, _current_slice_time) );

    // test
    bool result = buffwr_list[topic_id]->front_void(&content_out_ptr, true, _current_slice_time, true);
    // std::cout << "here\n";
    //

    // if (!result)
    //     return false;
    // Note: even if we don't get the new content, we do the transform.
    // Got the content's stamp
    msg_stamp = toROStime( buffer_list_ITRIPointCloud[_tid].get_stamp() );
    /*
    // tf2
    std::string _frame_id = _topic_param_list[topic_id].frame_id; // Note: this might be empty, which will be catched by exception.
    // geometry_msgs::TransformStamped _tfStamped_out;
    try{
        // _tfStamped_out = tfBuffer.lookupTransform(_ref_frame, _frame_id, ros::Time(0));
        // _tfStamped_out = tfBuffer.lookupTransform(_ref_frame, _frame_id, _msg_stamp);
        ros::Time _common_time;
        if(_is_using_latest_tf_common_update_time){
            _common_time = _latest_tf_common_update_time;
        }else{
            std::string err_str;
            tfBuffer._getLatestCommonTime(tfBuffer._lookupFrameNumber(_ref_frame), tfBuffer._lookupFrameNumber(_frame_id), _common_time, &err_str);
            // tfBuffer._getLatestCommonTime(tfBuffer._lookupFrameNumber(_ref_frame), tfBuffer._lookupFrameNumber(_stationary_frame), _common_time, &err_str);
        }
        // std::cout << "err_str = <" << err_str << ">\n";
        std::cout << "_common_time = " << _common_time.sec << ", " << _common_time.nsec << "\n";
        if (_msg_stamp > _common_time){
            std::cout << "_msg_stamp > _common_time\n";
            _msg_stamp = _common_time;
        }else{
            std::cout << "_msg_stamp <= _common_time\n";
        }
        _tfStamped_out = tfBuffer.lookupTransform(_ref_frame, _common_time, _frame_id, _msg_stamp, _stationary_frame, ros::Duration(0.2));
        std::cout << "Got transform\n";
    }
    catch (tf2::TransformException &ex) {
        ROS_WARN("%s",ex.what());
        return false;
    }
    */
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
}
//---------------------------------------------------------------//


// ITRIPointCloud
//---------------------------------------------------------------//
// input
void ROS_INTERFACE::_ITRI3DBoundingBox_CB(const msgs::LidRoi::ConstPtr& msg, const MSG::T_PARAMS & params){
    // Type_id
    //------------------------------------//
    int _tid = _topic_tid_list[params.topic_id];
    //------------------------------------//
    //
    bool result = buffer_list_ITRI3DBoundingBox[ _tid ].put( *msg);
    if (!result){
        std::cout << params.name << ": buffer full.\n";
    }
}
bool ROS_INTERFACE::get_ITRI3DBoundingBox(const int topic_id, msgs::LidRoi & content_out){
    // Type_id
    //------------------------------------//
    int _tid = _topic_tid_list[topic_id];
    //------------------------------------//
    return ( buffer_list_ITRI3DBoundingBox[_tid].front(content_out, true, _current_slice_time) );
}
bool ROS_INTERFACE::get_ITRI3DBoundingBox(const int topic_id, std::shared_ptr< msgs::LidRoi > & content_out_ptr){
    // Type_id
    //------------------------------------//
    int _tid = _topic_tid_list[topic_id];
    //------------------------------------//
    return ( buffer_list_ITRI3DBoundingBox[_tid].front(content_out_ptr, true, _current_slice_time) );
}
bool ROS_INTERFACE::get_ITRI3DBoundingBox(const int topic_id, std::shared_ptr< msgs::LidRoi > & content_out_ptr, ros::Time &msg_stamp){
    // Type_id
    //------------------------------------//
    int _tid = _topic_tid_list[topic_id];
    //------------------------------------//
    bool result = buffer_list_ITRI3DBoundingBox[_tid].front(content_out_ptr, true, _current_slice_time);
    msg_stamp = toROStime( buffer_list_ITRI3DBoundingBox[_tid].get_stamp() );
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
}
//---------------------------------------------------------------//
