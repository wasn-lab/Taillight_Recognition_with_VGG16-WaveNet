#! /usr/bin/env python3

import rospy
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import Point
import numpy as np

   
class Traffic_Light_Marker_Array:

    def __init__(self, stopline=None):
        self.mark_array = MarkerArray()
        self.lights_attr = dict() 
        if stopline is None:
            self.x = 0
            self.y = 0
            self.z = 0
            self.unit_vector = (1,0,0)
        else:
            self.x = stopline[0]
            self.y = stopline[1]
            self.z = stopline[2]
            stopline_vector = (stopline[3] - stopline[0], stopline[4] - stopline[1], stopline[5] - stopline[2])
            self.unit_vector = stopline_vector / np.linalg.norm(stopline_vector)
            #print(f"{self.unit_vector}")
            #self.unit_vector = __getUnitVector(stopline[0], stopline[1],stopline[2],stopline[3], stopline[4],stopline[5])

        self.light_radius = 1

        self.light_pole_radius = 0.3
        self.light_pole_hight = 10

        
        self.light_pole = Point(self.x - self.light_pole_radius * self.unit_vector[0], self.y - self.light_pole_radius * self.unit_vector[1], self.z + (self.light_pole_hight / 2))
        
        self.red_light = Point(
                self.x + self.light_radius * self.unit_vector[0], 
                self.y + self.light_radius * self.unit_vector[1], 
                self.z + self.light_pole_hight + self.light_radius * self.unit_vector[2])
       
        self.yellow_light = Point(
                self.red_light.x + self.light_radius * self.unit_vector[0], 
                self.red_light.y + self.light_radius * self.unit_vector[1], 
                self.red_light.z + self.light_radius * self.unit_vector[2])

        self.green_light = Point(
                self.yellow_light.x + self.light_radius * self.unit_vector[0], 
                self.yellow_light.y + self.light_radius * self.unit_vector[1],
                self.yellow_light.z + self.light_radius * self.unit_vector[2])

        self.green_left = Point(
                self.yellow_light.x + self.light_radius * self.unit_vector[0], 
                self.yellow_light.y + self.light_radius * self.unit_vector[1],
                self.yellow_light.z + self.light_radius * self.unit_vector[2])

        self.green_strait = Point(
                self.green_left.x + self.light_radius * self.unit_vector[0], 
                self.green_left.y + self.light_radius * self.unit_vector[1],
                self.green_left.z + self.light_radius * self.unit_vector[2])

        self.green_right = Point(
                self.green_strait.x + self.light_radius * self.unit_vector[0], 
                self.green_strait.y + self.light_radius * self.unit_vector[1],
                self.green_strait.z + self.light_radius * self.unit_vector[2])

        self.count_down = Point(
                self.green_right.x + self.light_radius * self.unit_vector[0], 
                self.green_right.y + self.light_radius * self.unit_vector[1],
                self.green_right.z + self.light_radius * self.unit_vector[2])

        self.lights_attr["r"] = {"id": 1, "pose": self.red_light, "on": (1.0, 0.0, 0.0), "off": (0.3, 0.0, 0.0)}
        self.lights_attr["y"] = {"id": 2, "pose": self.yellow_light, "on": (1.0, 1.0, 0.0), "off": (0.3, 0.3, 0.0)}
        self.lights_attr["g"] = {"id": 3, "pose": self.green_light, "on": (0.0, 1.0, 0.0), "off": (0.0, 0.3, 0.0)}

        self.lights_attr["g_l"] = {"id": 11, "pose": self.green_left, "on": (0.0, 1.0, 0.0), "off": (0.0, 0.3, 0.0)}
        self.lights_attr["g_s"] = {"id": 12, "pose": self.green_strait, "on": (0.0, 1.0, 0.0), "off": (0.0, 0.3, 0.0)}
        self.lights_attr["g_r"] = {"id": 13, "pose": self.green_right, "on": (0.0, 1.0, 0.0), "off": (0.0, 0.3, 0.0)}
         
        '''
        self.green_left = {
            "x_tip": {"x": self.yellow_light["x"] + self.light_radius * 3, "y": self.y, "z":  self.light_pole_hight}, 
            "x_tail": {"x": self.yellow_light["x"] + self.light_radius, "y": self.y, "z":  self.light_pole_hight}
        }
        self.green_straight = {
            "x_tip": {"x": self.green_left["x_tip"]["x"] + self.light_radius * 2, "y": self.y, "z": self.light_pole_hight - self.light_radius}, 
            "x_tail": {"x":self.green_left["x_tip"]["x"] + self.light_radius * 2, "y":self.y, "z": self.light_pole_hight + self.light_radius},  
        }
        self.green_right = {
            "x_tip": {"x": self.green_straight["x_tip"]["x"] + self.light_radius , "y": self.y, "z":  self.light_pole_hight}, 
            "x_tail": {"x":self.green_straight["x_tip"]["x"] + self.light_radius * 3, "y":self.y, "z":  self.light_pole_hight},  
        }
        '''
        #self.count_down ={"x": self.green_right["x_tail"]["x"] + self.light_radius, "y": self.y, "z": self.light_pole_hight}
            
    def __getUnitVector(self, x0, y0, x1, y1):
        distance = ((x0 - x1)**2 + (y0 -y1)**2)**0.5
        return ((x1 -x0)/distance, (y1 - y0)/distance)        

    def __draw_light_pole(self):
        lampole = Marker()
        lampole.header.frame_id = "/map"
        lampole.header.stamp = rospy.Time.now()
        lampole.type = 3
        lampole.id = 6
        # Set the scale of the marker
        lampole.scale.x = self.light_pole_radius
        lampole.scale.y = self.light_pole_radius
        lampole.scale.z = self.light_pole_hight
        # Set the color
        lampole.color.r = 0.7
        lampole.color.g = 0.7
        lampole.color.b = 0.7
        lampole.color.a = 1.0

        # Set the pose of the marker

        lampole.pose.position.x = self.light_pole.x
        lampole.pose.position.y = self.light_pole.y
        lampole.pose.position.z = self.light_pole.z

        lampole.pose.orientation.x = 0.0
        lampole.pose.orientation.y = 0.0
        lampole.pose.orientation.z = 0.0
        lampole.pose.orientation.w = 1.0
        #print("lampole")
        #print(lampole)
        self.mark_array.markers.append(lampole)

    def __draw_light(self, color, on):
        light = Marker()
        light.header.frame_id = "/map"
        light.header.stamp = rospy.Time.now()
        print(f"{color} {on}")
        # set shape, Arrow: 0; Cube: 1 ; Sphere: 2 ; Cylinder: 3
        light.type = 2
        light.id = light.color.r = self.lights_attr[color]["id"]
        # Set the color
        light.color.r = self.lights_attr[color]["on"][0] if on else self.lights_attr[color]["off"][0]
        light.color.g = self.lights_attr[color]["on"][1] if on else self.lights_attr[color]["off"][1]
        light.color.b = self.lights_attr[color]["on"][2] if on else self.lights_attr[color]["off"][2]
        light.color.a = 1.0
        # Set the scale of the marker
        light.scale.x = 1.0
        light.scale.y = 1.0
        light.scale.z = 1.0

        # Set the pose of the marker
        light.pose.position.x = self.lights_attr[color]["pose"].x
        light.pose.position.y = self.lights_attr[color]["pose"].y
        light.pose.position.z = self.lights_attr[color]["pose"].z
        light.pose.orientation.x = 0.0
        light.pose.orientation.y = 0.0
        light.pose.orientation.z = 0.0
        light.pose.orientation.w = 1.0
        #print("light" + color)
        #print(light)
        self.mark_array.markers.append(light)
        
    def __draw_coutdown(self, sec):
        countdowan_text = Marker()
        countdowan_text.header.frame_id = "/map"
        countdowan_text.header.stamp = rospy.Time.now()
        # set shape, Arrow: 0; Cube: 1 ; Sphere: 2 ; Cylinder: 3
        countdowan_text.type = 9
        countdowan_text.id = 7
        # Set the scale of the marker
        countdowan_text.scale.z = 1.0
        countdowan_text.text = str(sec)
        # Set the color
        countdowan_text.color.r = 1.0
        countdowan_text.color.g = 1.0
        countdowan_text.color.b = 1.0
        countdowan_text.color.a = 1.0
        # Set the pose of the marker
        countdowan_text.pose.position.x = self.count_down.x
        countdowan_text.pose.position.y = self.count_down.y
        countdowan_text.pose.position.z = self.count_down.z
        countdowan_text.pose.orientation.x = 0.0
        countdowan_text.pose.orientation.y = 0.0
        countdowan_text.pose.orientation.z = 0.0
        countdowan_text.pose.orientation.w = 1.0 
        self.mark_array.markers.append(countdowan_text) 

    def __draw_arrow(self, direction, on):

        direction_const = dict()
        
        
        arrow = Marker()
        arrow.header.frame_id = "/map"
        arrow.header.stamp = rospy.Time.now()
         
        # set shape, Arrow: 0; Cube: 1 ; Sphere: 2 ; Cylinder: 3
        arrow.type = 0
        arrow.id = self.lights_attr[direction]["id"]

        # Set the scale of the marker
        arrow.scale.x = 0.3
        arrow.scale.y = 0.5
        arrow.scale.z = 0.0

        # Set the color
        arrow.color.r = self.lights_attr[direction]["on"][0] if on else self.lights_attr[direction]["off"][0]
        arrow.color.g = self.lights_attr[direction]["on"][1] if on else self.lights_attr[direction]["off"][1]
        arrow.color.b = self.lights_attr[direction]["on"][2] if on else self.lights_attr[direction]["off"][2]
        arrow.color.a = 1.0

        head = None
        tail = None
        if direction == "g_l":
            head = Point(self.lights_attr[direction]["pose"].x - 0.5 * self.light_radius * self.unit_vector[0],
                         self.lights_attr[direction]["pose"].y - 0.5 * self.light_radius * self.unit_vector[1],
                         self.lights_attr[direction]["pose"].z - 0.5 * self.light_radius * self.unit_vector[2])
            tail = Point(self.lights_attr[direction]["pose"].x + 0.5 * self.light_radius * self.unit_vector[0],
                         self.lights_attr[direction]["pose"].y + 0.5 * self.light_radius * self.unit_vector[1],
                         self.lights_attr[direction]["pose"].z + 0.5 * self.light_radius * self.unit_vector[2])
        elif direction == "g_r":
            head = Point(self.lights_attr[direction]["pose"].x + 0.5 * self.light_radius * self.unit_vector[0],
                         self.lights_attr[direction]["pose"].y + 0.5 * self.light_radius * self.unit_vector[1],
                         self.lights_attr[direction]["pose"].z + 0.5 * self.light_radius * self.unit_vector[2])
            tail = Point(self.lights_attr[direction]["pose"].x - 0.5 * self.light_radius * self.unit_vector[0],
                         self.lights_attr[direction]["pose"].y - 0.5 * self.light_radius * self.unit_vector[1],
                         self.lights_attr[direction]["pose"].z - 0.5 * self.light_radius * self.unit_vector[2])
        elif direction == "g_s":
            head = Point(self.lights_attr[direction]["pose"].x, 
                         self.lights_attr[direction]["pose"].y, 
                         self.lights_attr[direction]["pose"].z + 0.5)
            tail = Point(self.lights_attr[direction]["pose"].x, 
                         self.lights_attr[direction]["pose"].y, 
                         self.lights_attr[direction]["pose"].z - 0.5)
        arrow.points = [tail, head]

        arrow.pose.orientation.x = 0.0
        arrow.pose.orientation.y = 0.0
        arrow.pose.orientation.z = 0.0
        arrow.pose.orientation.w = 1.0   
        self.mark_array.markers.append(arrow) 
   
    def dump_marker_array(self, r=0, y=0, g=0, g_l=0, g_s=0, g_r=0, sec=99):
        print("+++++++++++++++++++++++")
        print(f"{r} {y} {g} ")
        print(f"{bool(1)} {bool(0)} {bool(0)} ")
        self.__draw_light_pole()
        self.__draw_light("r", bool(int(r)))
        self.__draw_light("y", bool(int(y)))
        #self.__draw_light("g", bool(int(g)))
        self.__draw_arrow("g_l", bool(int(g_l)))
        self.__draw_arrow("g_s", bool(int(g_s)))
        self.__draw_arrow("g_r", bool(int(g_r)))
        self.__draw_coutdown(sec)
        print(self.mark_array)
        return self.mark_array

    
    
    def turn_on(self, color):
        pass
    def turn_off(self, color):
        pass
    def flashing(self, color):
        pass
if __name__ == '__main__':
    rospy.init_node('rviz_marker')
    stop_line = [1587.7426, 45506.1915, 66.5132, 1587.553, 45505.8394, 66.5042]
    traffic_light = Traffic_Light_Marker_Array(stop_line)
    print(f"{traffic_light.dump_marker_array()}")
    marker_pub = rospy.Publisher("/my_marker", MarkerArray, queue_size = 2)
    while not rospy.is_shutdown():
        marker_pub.publish(traffic_light.dump_marker_array())
        rospy.rostime.wallsleep(1.0)
