#ifndef GUI_VERSION_CTRL_H
#define GUI_VERSION_CTRL_H


// The version of ros_api, ros topics and message types
//----------------------------------//
#define __ROS_INTERFACE_VER__   2   // 1, 2
// 1 - FlOWER_EXPOSE
// 2 - HINO
#define __HINO_VER__            2   // 1, 2
// 1 - Original HINO
// 2 - post HINO
//----------------------------------//

// The version of the class of detection object
//----------------------------------//
#define __DETECTION_OBJ_VER_   2    // 1, 2
// 1 - FlOWER_EXPOSE
// 2 - HINO
//----------------------------------//

// Determin if showing the tracking result or the raw bounding boxes
//----------------------------------//
#define __IS_USING_TRACKING__  0 // 0, 1
// 0 - Without tracking, showing the raw lidar bounding boxes
// 1 - With tracking, showing the tracking boxes
//----------------------------------//


// Image size
//----------------------------------//
// ver 1: (608, 384)
// ver 2: (1920, 1080)
#if __ROS_INTERFACE_VER__ == 1
    #define _IMAGE_W_       608
    #define _IMAGE_H_       384
    #define _IMAGE_ASP_     1.583333333333333333 // Aspect ratio
#elif __ROS_INTERFACE_VER__ == 2
    #define _IMAGE_W_       1920
    #define _IMAGE_H_       1208
    #define _IMAGE_ASP_     1.5894039735099337 // Aspect ratio
#endif
//----------------------------------//


#endif // GUI_VERSION_CTRL_H
