import yaml
import json
import subprocess
import time
import datetime
from rosbag.bag import Bag
import csv


# topic_str = "/ADV_op/req_run_stop /ADV_op/run_state /ADV_op/sync /ADV_op/sys_fail_reason /ADV_op/sys_ready /CamObjBackTop /CamObjFrontCenter /CamObjFrontLeft /CamObjFrontRight /CamObjFrontTop /CamObjLeftBack /CamObjLeftFront /CamObjRightBack /CamObjRightFront /CameraDetection/occupancy_grid /CameraDetection/polygon /Flag_Info01 /Flag_Info02 /Flag_Info03 /Geofence_PC /LidarAll /LidarDetection /LidarDetection/grid /Path /PathPredictionOutput /PathPredictionOutput/camera /PathPredictionOutput/lidar /PathPredictionOutput/radar /PedCross/Pedestrians /REC/is_recording /REC/req_backup /RadarMarker /SensorFusion /V2X_msg /abs_virBB_array /backend/connected /cam/B_top /cam/F_center /cam/F_left /cam/F_right /cam/F_top /cam/L_front /cam/L_rear /cam/R_front /cam/R_rear /current_pose /dynamic_path_para /front_vehicle_target_point /imu_data /local_waypoints_mark /localization_state /localization_to_veh /marker_array_topic /mileage/brake_status /mm_tp_topic /nav_path /nav_path_astar_final /node_trace/all_alive /occupancy_grid /occupancy_grid_all_expand /occupancy_grid_updates /radFront /radar_point_cloud /rear_vehicle_target_point /rel_virBB_array /ring_edge_point_cloud /tf /tf_static /veh_info"
#
# topic_str = "/ADV_op/req_run_stop /ADV_op/run_state /ADV_op/sync /ADV_op/sys_fail_reason /ADV_op/sys_ready /CameraDetection/occupancy_grid /CameraDetection/polygon /Flag_Info01 /Flag_Info02 /Flag_Info03 /Geofence_PC /LidarAll /LidarDetection /LidarDetection/grid /LightResultOutput /Path /PathPredictionOutput /PathPredictionOutput/camera /PathPredictionOutput/lidar /PathPredictionOutput/radar /PedCross/Pedestrians /REC/is_recording /REC/req_backup /RadarMarker /SensorFusion /V2X_msg /abs_virBB_array /backend/connected /cam/back_top_120 /cam/front_bottom_60 /cam/front_top_close_120 /cam/front_top_far_30 /cam/left_back_60 /cam/left_front_60 /cam/right_back_60 /cam/right_front_60 /cam_obj/back_top_120 /cam_obj/front_bottom_60 /cam_obj/front_top_close_120 /cam_obj/front_top_far_30 /cam_obj/left_back_60 /cam_obj/left_front_60 /cam_obj/right_back_60 /cam_obj/right_front_60 /current_pose /dynamic_path_para /front_vehicle_target_point /imu_data /local_waypoints_mark /localization_state /localization_to_veh /marker_array_topic /mileage/ACC_run /mileage/AEB_run /mileage/Xbywire_run /mileage/brake_event /mm_tp_topic /nav_path /nav_path_astar_final /node_trace/all_alive /occupancy_grid /occupancy_grid_all_expand /occupancy_grid_updates /radFront /radar_point_cloud /rear_vehicle_target_point /rel_virBB_array /ring_edge_point_cloud /tf /tf_static /veh_info"

# Remove ADV_op inner topics
topic_str = "/CameraDetection/occupancy_grid /CameraDetection/polygon /Flag_Info01 /Flag_Info02 /Flag_Info03 /Geofence_PC /LidarAll /LidarDetection /LidarDetection/grid /LightResultOutput /Path /PathPredictionOutput /PathPredictionOutput/camera /PathPredictionOutput/lidar /PathPredictionOutput/radar /PedCross/Pedestrians /RadarMarker /SensorFusion /V2X_msg /abs_virBB_array /backend/connected /cam/back_top_120 /cam/front_bottom_60 /cam/front_top_close_120 /cam/front_top_far_30 /cam/left_back_60 /cam/left_front_60 /cam/right_back_60 /cam/right_front_60 /cam_obj/back_top_120 /cam_obj/front_bottom_60 /cam_obj/front_top_close_120 /cam_obj/front_top_far_30 /cam_obj/left_back_60 /cam_obj/left_front_60 /cam_obj/right_back_60 /cam_obj/right_front_60 /current_pose /dynamic_path_para /front_vehicle_target_point /imu_data /local_waypoints_mark /localization_state /localization_to_veh /marker_array_topic /mm_tp_topic /nav_path /nav_path_astar_final /occupancy_grid /occupancy_grid_all_expand /occupancy_grid_updates /radFront /radar_point_cloud /rear_vehicle_target_point /rel_virBB_array /ring_edge_point_cloud /tf /tf_static /veh_info"

topic_list = topic_str.split()

"""
All the event will be an element in the list in chronological order,
from oldest to latest.
"""
bag_dict_list = []

with open("./backup_history.txt") as _F:
    bag_dict = yaml.load_all(_F)
    # print(bag_dict)
    for _d in bag_dict:
        bag_dict_list.append(_d)
# Sort by timestamp
def get_event_timestamp(d):
  return d['timestamp']
bag_dict_list.sort(key=get_event_timestamp)
# print(bag_dict_list)

# The following is a test
# for _d in bag_dict_list:
#     print("Triggered at: %s" % str(_d["timestamp"]))
#     print("reason: %s" % _d["reason"])
#     print("bags:")
#     for _bag in _d["bags"]: # a list
#         print("- %s" % _bag)
#     print("\n")

def string_to_datetime(date_str):
    """
    This is an utility for converting the string to datetime.
    """
    _date = None
    print("date_str = %s" % date_str)
    # 2019-12-05-10-30-25
    try:
        _date = datetime.datetime.strptime(date_str, "%Y-%m-%d-%H-%M-%S")
        return _date
    except:
        print("Error when converting to datetime")
        pass
    return _date

def parse_backup_start_timestamp(bag_name):
    """
    Input: Fisrt bag name
    Output: datatime object
    """
    idx_head = 0
    for i, c in enumerate(bag_name):
        if c.isdigit():
            idx_head = i
            break
    idx_dot = bag_name.rfind('.', idx_head)
    idx_under_ = bag_name.rfind('_', idx_head, idx_dot)
    if idx_under_ > 0:
        idx_tail = idx_under_
    else:
        idx_tail = idx_dot
    #
    return string_to_datetime( bag_name[idx_head:idx_tail] )

def get_backup_start_timestamp(bag_name):
    """
    Input: Fisrt bag name
    Output: datatime object
    """
    info_dict = yaml.load(Bag(bag_name, 'r')._get_yaml_info())
    start_timestamp = info_dict.get("start", None)
    start_datetime = None
    if start_timestamp is None:
        print("No start time info in bag, try to retrieve the start time by parsing bag name.")
        start_datetime = parse_backup_start_timestamp(bag_name)
    else:
        start_datetime = datetime.datetime.fromtimestamp(start_timestamp)
    # print("info_dict = \n%s" % str(info_dict))
    # print('type(info_dict["start"]) = %s' % type(info_dict["start"]))
    # print(info_dict["start"])
    return start_datetime


def play_bag(file_list, topic_list=None, clock=True, loop=True, start_str=None, duration_str=None, rate_str=None):
    """
    """
    # The command
    cmd_list = ["rosbag", "play"]
    cmd_list += file_list
    if clock:
        cmd_list += ["--clock"]
    if loop:
        cmd_list += ["-l"]
    if start_str: # Note: not None or empty string
        cmd_list += ["-s %s" % start_str]
    if duration_str:
        cmd_list += ["-u %s" % duration_str]
    if rate_str:
        cmd_list += ["-r %s" % rate_str]
    if (not topic_list is None):
        # _topic_str = " ".join(topic_list)
        # cmd_list += ["--topics %s" % _topic_str]
        cmd_list += ["--topics"]
        cmd_list += topic_list
    print("")
    # print("Executing command: %s" % cmd_list)
    print("Command in bash format:\n%s" % " ".join(cmd_list))
    print("")

    print("bags to play:")
    for _bag in file_list:
        print(_bag)
    #
    time_wait_to_start = 0.2
    print("\nWait %f sec. to play.." % time_wait_to_start)
    time.sleep(time_wait_to_start)
    #
    # _ps = subprocess.Popen(cmd_list)
    _ps = subprocess.Popen(' '.join(cmd_list), shell=True)
    print("=== Subprocess started.===")
    try:
        while _ps.poll() is None:
            # report error and proceed
            time.sleep(1.0)
    except (KeyboardInterrupt, SystemExit):
        _ps.terminate()
        print("Terminating...")

    time.sleep(0.5) # Wait for subprocess closs
    result = _ps.poll()
    print("result = %s" % str(result))
    print("=== Subprocess finished.===")

    print("Finish the command.")

#---------------------------------------#
def main():
    """
    """
    global bag_dict_list
    global topic_list

    for idx in range(len(bag_dict_list)):
        _d = bag_dict_list[idx]
        print('%d: reason: "%s" [%s], %d bags' % ((idx+1), _d["reason"], str(_d["timestamp"]), len(_d["bags"])) )

    #
    try:
        with open('./backup_list.csv', mode='w') as csv_file:
            fieldnames = list(bag_dict_list[0].keys())
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

            writer.writeheader()
            for _d in bag_dict_list:
                writer.writerow(_d)
        print("Done writing CSV")
    except:
        print("Error writing CSV")


    # User selection
    try:
        txt_input = raw_input
    except NameError:
        txt_input = input
    #
    str_in = txt_input("Event id to play:")
    # try:
    #     str_in = txt_input("Event id to play:")
    # except EOFError:
    #     print("\nEOFError")
    #
    try:
        id_in = int(str_in)
        id_in -= 1
        print("Play #%d event.\n" % (id_in+1))
    except:
        id_in = None



    if (not id_in is None) and (id_in >= 0) and (id_in < len(bag_dict_list)):
        #
        _d = bag_dict_list[id_in]
        file_list = sorted(_d["bags"])

        # Print the file list
        print("bags to play:")
        for _bag in file_list:
            print(_bag)


        # Calculate the relative time
        # start_datetime = parse_backup_start_timestamp( file_list[0] )
        start_datetime = get_backup_start_timestamp( file_list[0] )
        event_datetime = _d["timestamp"]
        delta_time = event_datetime - start_datetime
        #
        event_sec = delta_time.total_seconds()
        # print("event_sec = %s, type = %s" % (str(event_sec), str(type(event_sec))))
        print("\n---\nThe event happened at %f sec.\n---" % event_sec)

        # Other control items
        s_str_in = txt_input('Start from? (default: 0, unit: sec.) ("e5" --> (event time - 5.0 sec))\n')
        u_str_in = txt_input('Duration? (default: "record-length", unit: sec.) ("e" --> symetric around the event time, event should happen at the middle of playback)\n')
        r_str_in = txt_input("Rate? (default: 1, unit: x)\n")

        s_str_in = s_str_in.strip()
        u_str_in = u_str_in.strip()
        r_str_in = r_str_in.strip()

        # Generate the time around event time
        start_time = 0.0
        duration = 0.0
        if len(s_str_in) > 0:
            if s_str_in[0] == "e":
                try:
                    start_time = event_sec - float(s_str_in[1:])
                except ValueError as e:
                    print(e)
                    default_ahead_t = 5.0
                    print("start_time: No proper time ahead given after 'e', using default value [%f]." % default_ahead_t)
                    start_time = event_sec - default_ahead_t
                #
                if start_time < 0.0:
                    start_time = 0.0
                s_str_in = "%f" % start_time
                print("start_time = %f" % start_time)
            else: # Normal value, no "e" or other commands
                try:
                    start_time = float(s_str_in)
                except ValueError as e:
                    print(e)
                    print("start_time: No proper time given, start from head.")
                    start_time = 0.0
            #
        #
        if len(u_str_in) > 0 and u_str_in[0] == "e":
            duration = (event_sec - start_time)*2.0
            u_str_in = "%f" % duration
            print("duration = %f" % duration)

        # Indicate the event happend relative time
        print("\n-------\nNote: Event happend at %.2f sec.\n-------\n" % (event_sec - start_time))

        play_bag(file_list, topic_list=topic_list, clock=True, loop=True,  start_str=s_str_in, duration_str=u_str_in, rate_str=r_str_in)
    else:
        print("Wrong input type, exit.")
    print("End of main().")


if __name__ == "__main__":

    try:
        main()
    except (KeyboardInterrupt, SystemExit):
        print("Stopping...")
        time.sleep(0.5)
    print("End of player.")
