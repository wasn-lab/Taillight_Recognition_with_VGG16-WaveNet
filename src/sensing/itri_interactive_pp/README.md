# itri_interactive_pp execution method
roslaunch itri_interactive_pp ipp.launch input_topic:=3 delay_node:=2 prediction_horizon:=10 print_log:=0 tf_map:=1 

Input_topic : 
  <!-- 1: /Tracking2D/front_bottom_60 -->
  <!-- 2: /PathPredictionOutput -->
  <!-- 3: /Tracking3D -->
Prediction_horizon : 
  In Prediction_horizon*1/fps = Prediction time
  e.g. 10 * 1/2 = 5sec

delay_node :
  filter out the input to 2fps

tf_map :
  translate local coordinate into global coordinate

