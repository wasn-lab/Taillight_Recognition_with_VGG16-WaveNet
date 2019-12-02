#ifndef OBJ_CLASS_DEF_H
#define OBJ_CLASS_DEF_H

// Version control
//----------------------------------------//
#include "GUI_version_control.h"
//----------------------------------------//


#include "Common.h"



class OBJECT_CLASS{
public:


    OBJECT_CLASS(){
        //
    #if __DETECTION_OBJ_VER_ == 1
        glm::vec3 obj_class_colors[] = {
            glm::vec3(50, 50, 255), // person
            glm::vec3(255, 153, 102), // bicycle
            glm::vec3(153, 255, 255), // car
            glm::vec3(255, 153, 127), // motorbike
            glm::vec3(255, 255, 0), // not showing aeroplane
            glm::vec3(102, 204, 255), // bus
            glm::vec3(255, 255, 100), // not showing train
            glm::vec3(255, 153, 102), // truck
            glm::vec3(50, 50, 50) // default
        };
        std::string obj_class_strings[] = {
            std::string("person"),
            std::string("bicycle"),
            std::string("car"),
            std::string("motorbike"),
            std::string("aeroplane"),
            std::string("bus"),
            std::string("train"),
            std::string("truck"),
            std::string("unknown")
        };
    #elif __DETECTION_OBJ_VER_ == 2
        glm::vec3 obj_class_colors[] = {
            glm::vec3(50, 50, 50), // Unknown
            glm::vec3(50, 50, 255), // person
            glm::vec3(255, 153, 102), // bicycle
            glm::vec3(255, 153, 127), // motorbike
            glm::vec3(153, 255, 255), // car
            glm::vec3(102, 204, 255), // bus
            glm::vec3(255, 153, 102), // truck
            glm::vec3(50, 50, 50) // default: Unknown
        };
        std::string obj_class_strings[] = {
            std::string("Unknown"),
            std::string("Person"),
            std::string("Bicycle"),
            std::string("Motorbike"),
            std::string("Car"),
            std::string("Bus"),
            std::string("Truck"),
            std::string("unknown")
        };
    #endif
        //
        color_list.assign( obj_class_colors, obj_class_colors +  ( sizeof(obj_class_colors)/sizeof(*obj_class_colors) ) );
        type_str_list.assign( obj_class_strings, obj_class_strings +  ( sizeof(obj_class_strings)/sizeof(*obj_class_strings) ) );
        // Normalize color
        color_normalize_factor = 1.0f/255.0f;
        for (size_t i=0; i < color_list.size(); ++i){
            color_list[i] *= color_normalize_factor;
        }
        //--------------------------------//
    }


    glm::vec3 get_color(size_t obj_class_in){
        if ( obj_class_in >= color_list.size() ){
            return color_list[color_list.size()-1];
        }else{
            return color_list[obj_class_in];
        }
    }

    std::string get_string(size_t obj_class_in){
        if ( obj_class_in >= type_str_list.size() ){
            return type_str_list[type_str_list.size()-1];
        }else{
            return type_str_list[obj_class_in];
        }
    }
protected:
    float color_normalize_factor;
    std::vector<glm::vec3> color_list;
    std::vector<std::string> type_str_list;
private:

};





#endif // OBJ_CLASS_DEF_H
