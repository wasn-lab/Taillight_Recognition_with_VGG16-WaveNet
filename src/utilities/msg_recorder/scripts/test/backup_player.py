import yaml
import json
import subprocess

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
    print(bag_dict)
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



def play_bag(file_list, topic_list=None, clock=True, loop=True):
    """
    """
    # The command
    cmd_list = ["rosbag", "play"]
    cmd_list += file_list
    if clock:
        cmd_list += ["--clock"]
    if loop:
        cmd_list += ["-l"]
    if not topic_list is None:
        # _topic_str = " ".join(topic_list)
        # cmd_list += ["--topics %s" % _topic_str]
        cmd_list += ["--topics"]
        cmd_list += topic_list
    print("")
    print("Executing command: %s" % cmd_list)
    print("Command in bash format:\n%s" % " ".join(cmd_list))
    print("")
    subprocess.call(cmd_list)
    print("Finish the command.")

#---------------------------------------#
def main():
    """
    """
    global bag_dict_list
    global topic_list

    file_list = bag_dict_list[0]["bags"]

    play_bag(file_list, topic_list=topic_list, clock=True, loop=True)
    print("End of main loop.")


if __name__ == "__main__":
    try:
        main()
    except:
        pass
    print("End of player.")
