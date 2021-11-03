# How to use:
0. install pypcd first! https://github.com/dimatura/pypcd
1. update numpy to latest version
2. make a 'bags' folder in the root of the extractor
3. put all Rosbags and folders containing Rosbags in the 'bags' folder
4. run the extractor: python Batch_Rosbag_Extractor.py
5. The Result of the Extraction will be in the newly made 'Extracted' folder

# How to get topics of decompressed camera/lidar raw data:
```sh
roscore
roslaunch lidar b1.launch mode:=9
roslaunch image_compressor decmpr.launch
bash bag_record.sh
bash bag_play.sh
```
Ps1. when started rosbag play, click space immediately to pause the play and resume in 5 seconds. This action is to let decompressor modules warm up for preventing unwanted blank in the recorded bag.

Ps2. Since recorded image raw data is /cam/.../raw, not /cam/..., one need to modify Batch_Rosbag_Extractor.py -- change all /cam/... to /cam/.../raw.

Ps3. If the bag is not recorded on Car B1, modify the launch file when roslaunch lidar.
