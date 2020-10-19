# Messages that control the behavior of a car
# The messages are sent in the ROS topic /control_command

# See msgs/include/control_command.h for definitions
uint32 target_id
uint32 action_id

# The module that sends the command, see msgs/include/module_id.h
uint32 sender_id
