Header header

uint8 MANUAL = 0
uint8 READY = 1
uint8 SPEED_ONLY = 2
uint8 STEER_ONLY = 3
uint8 AUTO = 4

uint8 GEAR_P = 0
uint8 GEAR_R = 1
uint8 GEAR_N = 2
uint8 GEAR_D = 3

uint8 CONTROL_NOT_READY = 0
uint8 CONTROL_READY = 1
uint8 CONTROL_ENGAGED = 2

float32 speed
float32 steering_angle
float32 steering_cmd
float32 steering_torque
uint8 mode
uint8 gear_state
uint8 eps_state
uint8 throttle_state

uint8 DBW_ENABLE_SWITCH_OFF = 0
uint8 DBW_ENABLE_SWITCH_ON = 1
uint8 DBW_ENABLE_NO_ACTION = 3

uint8 DBW_OFF = 0
uint8 DBW_ON = 1

uint8 STEER_MCU_NO_ERROR = 0
uint8 STEER_MCU_ERROR = 1
uint8 STEER_MCU_ZERO_POINT_NOT_LEARNED = 2
uint8 STEER_MCU_NO_ACTION = 3

uint8 BRAKE_PEDAL_SWITCH_OFF = 0
uint8 BRAKE_PEDAL_SWITCH_ON = 1
uint8 BRAKE_PEDAL_SWITCH_NO_ACTION = 3

uint8 BLINKER_TURN_OFF = 0
uint8 BLINKER_TURN_RIGHT = 1
uint8 BLINKER_TURN_LEFT = 2
uint8 BLINKER_TURN_HAZARD = 3

uint8 AEB_SET_INACTIVE = 0
uint8 AEB_SET_ACTIVE = 1
uint8 AEB_SET_DISALLOW = 2
uint8 AEB_SET_NO_ACTION = 3

uint8 AMT_GEAR_R = 0
uint8 AMT_GEAR_R_SLOW = 1
uint8 AMT_GEAR_N = 2
uint8 AMT_GEAR_D_SLOW = 3
uint8 AMT_GEAR_D = 4
uint8 AMT_GEAR_NO_ACTION = 5

uint8 ENG_CTRL_MODE_ACCEL = 0
uint8 ENG_CTRL_MODE_TORQUE = 1
uint8 ENG_CTRL_MODE_SPEED = 2

float32 engine_speed
float32 engine_speed_cmd
float32 engine_torque_cmd
float32 engine_accel_cmd
float32 engine_target_speed
float32 deceleration_cmd
uint8 enable_amt_ctrl
uint8 enable_engine_ctr
uint8 enable_brake_ctr
uint8 engine_ctrl_mode_cmd
int8 amt_gear_range
uint8 amt_gear_cmd
uint8 gateway_keep_alive
uint8 amt_gear_pos
uint8 blinker
uint8 aeb_set
float32 speedCmd
uint8 brake_pedal
uint8 steer_mcu_error
uint8 ctrl_switch_aeb
uint8 ctrl_switch_amt
uint8 ctrl_switch_blinker
uint8 ctrl_switch_brake
uint8 ctrl_switch_engine
uint8 ctrl_switch_steer
