#!/usr/bin/env python
# license removed for brevity
#test tool
import rospy
import json
from msgs.msg import BackendInfo
import time;
from std_msgs.msg import Header
def talker():
    pub = rospy.Publisher('Backend/Info', BackendInfo, queue_size=10)
    rospy.init_node('talker', anonymous=True)
    rate = rospy.Rate(1) # 10hz
    open_close = 1;
    open_close2 = 0;
    msg = BackendInfo()
  
    while not rospy.is_shutdown():
      
        if open_close == 1:
            open_close = 0
        else:
            open_close = 1

	    if open_close2 == 1:
            open_close2 = 0
	    else:
		    open_close2 = 1

	    msg.motor_temperature = 1.0 
        msg.tire_pressure= 1.0 
        msg.air_pressure= 1.0 
        msg.battery= 100.0 
        msg.steer= 100.0 
        msg.localization= 100.0 
        msg.odometry= 100.0 
        msg.speed= 100.0 
        msg.rotating_speed= 100.0 
        msg.bus_stop= 100.0 
        msg.vehicle_number= 0.1
        msg.gear = 1.0
        msg.hand_brake= 1
        msg.steering_wheel= 0.1 
        msg.door= 0
        msg.air_conditioner= 1
        msg.radar= open_close2 
        msg.lidar= open_close 
        msg.camera= open_close2 
        msg.GPS= open_close 
        msg.headlight= 0 
        msg.wiper= open_close 
        msg.indoor_light= 0 
        msg.gross_power= open_close  
        msg.left_turn_light= 1 
        msg.right_turn_light= 0  
        msg.estop= open_close2
        msg.ACC_state= open_close 
        msg.time= 0.00
        msg.driving_time= 100.0 
        msg.mileage= 100.0 
        msg.gross_voltage= 100.0 
        msg.gross_current= 100.0 
        msg.highest_voltage= 100.0 
        msg.highest_number= 100.0 
        msg.lowest_volage= 100.0 
        msg.lowest_number= 100.0 
        msg.voltage_deviation= 100.0 
        msg.highest_temp_location= 100.0 
        msg.highest_temperature= 100.0
        msg.mode= 1
        msg.emergency_exit= 0.0

        pub.publish(msg)
        rate.sleep()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
