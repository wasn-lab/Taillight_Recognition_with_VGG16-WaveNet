# IPP environment setup
1. execute ipp installation program
```
bash install_pytorch.sh
```
# itri_interactive_pp execution method
## first source environment :
```
source devel/setup.bash
source sandbox/bin/activate
```
## execute roslaunch file :
```
roslaunch itri_interactive_pp ipp.launch delay_node:=2 prediction_horizon:=10 print_log:=0 tf_map:=1 output_csv:=True
```
## IPP need following topic :

### map
```
roslaunch map_pub map_pub.launch & roslaunch planning_launch planning_all_launch.launch
```
### lidar
```
roslaunch lidar b1.launch mode:=9 use_compress:=false
```
### tracking
```
roslaunch itri_tracking_3d tpp.launch input_source:=1 create_polygon_from_bbox:=True show_classid:=True drivable_area_filter:=True
```
# Ros param explaination
## Input_topic : 
  1. /Tracking2D/front_bottom_60
  2. /PathPredictionOutput
  3. /Tracking3D

## Prediction_horizon : 
  In Prediction_horizon*1/fps = Prediction time<br></br>
  e.g. 10 * 1/2 = 5sec

## delay_node :
  0 is for 10fps input<br></br>
  1 is for 2fps input (open fps_filter)<br></br>

## tf_map :
  translate local coordinate into global coordinate

## print_log (print on screen):
  0 is for no output <br></br>
  1 is for execution information output <br></br>
  2 is for prediction value and current position value<br></br>

## output_csv:
  True : output buffer data (processed bag data)<br></br>
  False : no output csv<br></br>
