#! /usr/bin/env python3
import rospy
from msgs.msg import Spat
from std_msgs.msg import String
from Traffic_light import Traffic_Light_Marker_Array 
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray



coordinates = dict()

#osm上 紅綠燈的xy 停止線的z
coordinates["1"] = [1960.2815, 44887.2147, 66.9299, 1959.3202, 44887.4887, 66.9299]
coordinates["2"] = [1839.5894, 44898.7828, 66.419, 1839.8394, 44899.9565, 66.419]
coordinates["3"] = [1764.322, 44917.9077, 66.1166, 1764.695, 44919.048, 66.1166]
coordinates["4"] = [1605.8277, 44967.8093, 65.3181, 1606.2915, 44968.9157, 65.3181]
coordinates["5"] = [1414.449, 45051.5615, 64.0365, 1414.8572, 45052.4727, 64.0365]
coordinates["6"] = [1163.1664, 45160.8454,63.1487, 1163.5555, 45161.7659, 63.1487]
coordinates["7"] = [799.3363, 45364.1045, 60.4494, 799.9239, 45365.1504, 60.4494]
coordinates["8"] = [639.4727, 45456.0497, 59.321, 640.1049, 45457.0695, 59.321]
coordinates["9"] = [771.5705, 45657.625, 58.6996, 772.4656, 45656.8271, 58.6996]
coordinates["10"] = [840.1494, 45740.7697, 58.2722, 840.9265, 45740.1408, 58.2722]
coordinates["11"] = [982.3085, 45902.6052, 57.8069, 982.9773, 45901.8625, 57.8069]
coordinates["12"] = [1045.3493, 45962.8145, 57.5033, 1045.8987, 45962.2333, 57.5033]
coordinates["23"] = [1396.7755, 45625.313, 60.2326, 1396.0233, 45624.3783, 60.2326]
coordinates["24"] = [1587.5733, 45506.5097, 61.5952, 1587.0046, 45505.4534, 61.5952]
coordinates["25"] = [2041.7508, 45329.1698, 64.5103, 2041.4269, 45328.0143, 64.5103]
coordinates["26"] = [1852.8168, 45385.4685, 63.4525, 1852.4281, 45384.3333, 63.4525]

rospy.init_node('mileage_listener', anonymous=True)
marker_pub = rospy.Publisher("/traffic_rviz", MarkerArray, queue_size = 2)

def callback(spat):
   print(f"{spat.spat_state}")
   state = format(spat.spat_state, "b")
   state_8_bit = state.zfill(8)
   intersection_id = str(spat.intersection_id)
   
   green =  state_8_bit[2]
   greeb_left = state_8_bit[3]
   green_strait = state_8_bit[4]
   green_right = state_8_bit[5]
   yellow = state_8_bit[6]
   red = state_8_bit[7]
   print(f"red = {red} yellow = {yellow} green = {green}")
   print(f"{state.zfill(8)}")
   print(f"{int(spat.spat_sec)}")
   if intersection_id != "0":
       stop_line = coordinates[intersection_id]
       traffic_light = Traffic_Light_Marker_Array(stop_line)
   else:
       traffic_light = Traffic_Light_Marker_Array()
   marker_pub.publish(traffic_light.dump_marker_array(red, yellow, green, greeb_left, green_strait, green_right,int(spat.spat_sec)))
  

def listener():
    
    rospy.Subscriber("/traffic", Spat, callback)
    rospy.spin()


if __name__ == '__main__':
    listener()
