#include <iostream>
#include <ros/ros.h>
#include <autoware_perception_msgs/DynamicObjectArray.h> //SUB

//-------------ROS Topic-------------
ros::Publisher filtered_objects_pub;

//-------------ROS Param-------------
double confidence_threshold = 0.15; 
autoware_perception_msgs::DynamicObjectArray input;

//=========================Filtered Map PP Functions========================
void MapPPConfidenceFilter(autoware_perception_msgs::DynamicObjectArray &input)
{
    autoware_perception_msgs::DynamicObjectArray output;

    output.header.frame_id = input.header.frame_id;
    output.header.stamp    = ros::Time::now();

    for (int i=0 ; i < input.objects.size() ; i++)
    {   
        double max_confidence = 0;
        autoware_perception_msgs::DynamicObject object_;
        object_.id                            = input.objects[i].id;
        object_.semantic                      = input.objects[i].semantic;
        object_.shape                         = input.objects[i].shape;
        object_.state.pose_covariance         = input.objects[i].state.pose_covariance;
        object_.state.orientation_reliable    = input.objects[i].state.orientation_reliable;
        object_.state.twist_covariance        = input.objects[i].state.twist_covariance;
        object_.state.twist_reliable          = input.objects[i].state.twist_reliable;
        object_.state.acceleration_covariance = input.objects[i].state.acceleration_covariance;
        object_.state.acceleration_reliable   = input.objects[i].state.acceleration_reliable;
        
        for (const auto &path : input.objects[i].state.predicted_paths)
        {
            std::cout << "input confidence " << path.confidence << std::endl;
            if (path.confidence >= max_confidence) 
                max_confidence = path.confidence;   
        }  
        for (const auto &path : input.objects[i].state.predicted_paths)
        {
            if (path.confidence == max_confidence || path.confidence >= confidence_threshold)
                object_.state.predicted_paths.push_back(path);
        }

        std::cout << " Checking filter: " << std::endl
                  << " --max confidence " << max_confidence << std::endl
                  << " --filtered PP    " << object_.state.predicted_paths.size() << std::endl;
        output.objects.push_back(object_);
    }
    filtered_objects_pub.publish(output);
}

//=================================Callback==================================
void callbackMapPPFilter(const autoware_perception_msgs::DynamicObjectArray::ConstPtr &msg)
{
    input = *msg;
    ROS_INFO("[Geofence_Map_PP_Filter] getting onjects info...");
    MapPPConfidenceFilter(input);
}

//===================================Main====================================
int main(int argc, char **argv)
{
    // Initialize-----------------------------------
    ros::init(argc, argv, "Geofence_Map_PP_Filter");
    ros::NodeHandle node;
    ros::param::get(ros::this_node::getName()+"/confidence_threshold", confidence_threshold);
    ROS_INFO("[Geofence_Map_PP_Filter] running...");

    // Subscriber-----------------------------------
    ros::Subscriber GeofenceMapPPFilterSub = node.subscribe("objects", 1000, callbackMapPPFilter);

    // Publisher------------------------------------
    filtered_objects_pub = node.advertise<autoware_perception_msgs::DynamicObjectArray>("objects/filtered", 1);

    // Run------------------------------------------
    ros::spin();

    return 0;
}

