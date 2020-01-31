import yaml
import json
import subprocess
import time

topic_str = "/CamObjBackTop /CamObjFrontCenter /CamObjFrontLeft /CamObjFrontRight /CamObjFrontTop /Flag_Info01 /Flag_Info02 /Flag_Info03 /Geofence_PC /LidarAll /LidarDetection /PathPredictionOutput/radar /RadarMarker /abs_virBB_array /cam/B_top /cam/F_center /cam/F_left /cam/F_right /cam/F_top /cam/L_front /cam/L_rear /cam/R_front /cam/R_rear /clock /current_pose /dynamic_path_para /imu_data /localization_to_veh /marker_array_topic /mm_tp_topic /nav_path /nav_path_astar_final /occupancy_grid /radar_point_cloud /rel_virBB_array /ring_edge_point_cloud /rosout /rosout_agg /tf /veh_info"
#
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
# print(bag_dict_list)

# The following is a test
# for _d in bag_dict_list:
#     print("Triggered at: %s" % str(_d["timestamp"]))
#     print("reason: %s" % _d["reason"])
#     print("bags:")
#     for _bag in _d["bags"]: # a list
#         print("- %s" % _bag)
#     print("\n")


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

    s_str_in = txt_input("Start from? (default: 0, unit: sec.)\n")
    u_str_in = txt_input('Duration? (default: "record-length", unit: sec.)\n')
    r_str_in = txt_input("Rate? (default: 1, unit: x)\n")

    if (not id_in is None) and (id_in >= 0) and (id_in < len(bag_dict_list)):
        #
        file_list = bag_dict_list[id_in]["bags"]
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
