# msg_recorder

The "msg_recorder" is a command-line tool/ROS-node that helps records ROS messages to rosbag files with enhanced controllability.

There are mainly two usage:
- Auto-mode: Continuous recording with files backup when triggered
- Manual-mode: Simple start/stop record control

There are two input interfaces which can be used at the same time
- Keyboard input (which can be disabled by passing --NO_KEY_IN argument)
- Sending ROS topics

The recorded rosbag files will be defaultly saved under ~/rosbag_files 
If you want to change the location, you can do one of the following
- Create symlink with name "rosbag_files" under "~/" (recommended, it's useful for changing the storaging place to other disks)
- Modify the path in "rosbag_setting.yaml" configuration file (not recommended for the sake of version control)

# Usage
## Run
With ROS master running,

Auto-mode:
```
$ rosrun msg_recorder msg_recorder.py -A
```

Manual-mode:
```
$ rosrun msg_recorder msg_recorder.py
```

Auto-mode without keyboard input:
```
$ rosrun msg_recorder msg_recorder.py -A --NO_KEY_IN
```

## Keyboard Input
```
----------------------
Type a command and press ENTER:
----------------------
s:start
t:terminate
c:cut file 
k:keep file
q:quit
----------------------
>>>
```

## ROS Topics I/O

```
- /REC/record:      input, require start/stop recording, type: Bool, value={True:start recording, False:stop recording}
- /REC/req_backup:   input, trigger for backing-up bags, type: Empty, value={}
- /REC/is_recording: latched output, displaying the status of the recorder, type: Bool, value={True:recording, False:stopped}
- /REC/trigger_report: latched output, displaying the information of the latest trigger event, type: String, value={s| s = the string appended to the history file}
```

# The Output Bags

The bags are stored in the following directory by default
```
~/rosbag_files/
```
(This path can be modified by changing the "output_dir_tmp" parameter in rosbag_setting.yaml)

The triggered backup bags are stored in the following directory by default
```
~/rosbag_files/backup/
```
(This path can be modified by changing the "output_dir_kept" parameter in rosbag_setting.yaml)

Note that in the backup directory, the program also generate a "backup_history.txt" file which list all the triggered event nformation and associated bags (latest event is at the end of the file).


# Modifying Recorded Topics List

When the "is_recording_all_topics" parameter in \*.yaml file is set to False, topics to be recorded can be explicitly specified.
Modifying the topic names explicitly list in the following files can change the list of topics to be recorded in associated mode.

Auto-mode:
```
record_topics_auto.txt      
```

Manual mode:
```
record_topics.txt
```

## Example record_topics.txt File

```
/abc    
/def
/ghi   

   /jkl
/mno
aa

```
Note:
- Topics not on the list will not be recorded.
- The space in line doesn't matter.
- The empty line will be ignored.
- The '/' at in front of each topic name is not mandatory.


# Modifying the Behavior of the Recorder

Modifying parameters list in the following file can change the behavior of the recorder in associated mode.

Auto-mode:
```
rosbag_setting_auto.yaml    
```

Manual mode:
```
rosbag_setting.yaml
```

## Example rosbag_setting.yaml File
```
#---------------------#
start_at_begining:  false # Determin if it's going to start rosbag record when the program begin
#---------------------#

# I/O setting
#-----------------------------------#
# Topics to be recorded, for explicitly identify the topic names, please list in "record_topics.txt"
is_recording_all_topics: true # false
#-----------------------------------#
output_dir_tmp:     ~/rosbag_files
bag_name_prefix:    my_record
output_dir_kept:    ~/rosbag_files/backup
#-----------------------------------#

# Output behavior setting
#-----------------------------------#
is_splitting:       false
split_size:         1024    # MB, no effect if no split
max_split_num:      6       # Maximum number of split files. The oldest file will be delete When the one is generated, no effect if no split; empty -> None
#-----------------------------------#
record_duration:            # duration to stop/split; empty -> None -> eternal; 1m -> 1 min.; 30 -> 30 sec.; 2h -> 2 hours
#-----------------------------------#
is_compressed:      false
compression_method: lz4     # lz4/bz2, default -> lz4; bz2 -> slow, high compression; lz4 -> fast, low compression
#-----------------------------------#

# Backup setting
#-----------------------------------#
time_pre_trigger:    5.0    # sec.
time_post_trigger:   5.0    # sec.
#-----------------------------------#

# Performance tunning
#-----------------------------------#
buffsize:                   # Input buffer size for all messages before bagging, empty -> None -> 256MB
chunksize:                  # Advanced, memory with SIZE KB for each chunks before writing to disk. Tune if needed. empty -> None -> default 768KB
#-----------------------------------#

```





# The ROS Package

```
msg_recorder
        |- scripts -- msg_recorder.py  
        |- params  
              |- rosbag_setting.yaml
              |- record_topics.txt
              |- rosbag_setting_auto.yaml        
              |- record_topics_auto.txt                    
```                   

# ROS Nodes
```
msg_recorder.py
```

# Dependency
- rosbag/record

# Known Issues

- If the main thread is terminated by SIGKILL (e.g. $ kill -9), the rosbag subprocess won't stop.

  Note: normal termination like SIGTERM (ctrl-c) or key-in: "q" will quit cleanly

  --> If you are facing this problem, find out the PID by $ps ax |grep rosbag/record and kill it by $ kill -9 (PID)
