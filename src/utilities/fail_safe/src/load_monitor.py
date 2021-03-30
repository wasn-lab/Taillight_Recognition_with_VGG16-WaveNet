# Copyright (c) 2021, Industrial Technology and Research Institute.
# All rights reserved.
"""
Publish cpu/gpu load
"""
import re
import subprocess
import json
import rospy
from std_msgs.msg import String

W_RGX = re.compile(r"load average: (?P<load>[ \.\d,]+)")
NVIDIA_SMI_LOAD_RGX = re.compile(r".+[\s]+(?P<load>[\d]+)%[\s]+Default.+")

def get_hostname():
    """
    Return the hostname of the machine, i.e., /etc/hostname
    """
    return str(subprocess.check_output(["hostname"])).strip()


def get_nproc():
    """
    Return the number of cpu cores.
    """
    return int(str(subprocess.check_output(["nproc"])).strip())


def _parse_nvidia_smi_output(line):
    """
    Given |line| as
    | 22%   34C    P8     9W / 250W |     44MiB / 11016MiB |      17%      Default |
    return "17"
    """
    match = NVIDIA_SMI_LOAD_RGX.match(line)
    if match:
        return match.expand(r"\g<load>")
    return ""


def _get_gpu_load_by_nvidia_smi():
    cmd = ["nvidia-smi", "--format=csv,noheader,nounits",
           "--query-gpu=utilization.gpu"]
    ret = None
    try:
        output = str(subprocess.check_output(cmd))
        line = output.splitlines()[-1]
        ret = float(line.strip())
    except subprocess.CalledProcessError:
        ret = -1
    if ret is not None:
        return ret
    cmd = ["nvidia-smi"]  # fall back to nvidia-smi
    ret = -1.0
    try:
        output = str(subprocess.check_output(cmd))
        for line in output.splitlines():
            temp = _parse_nvidia_smi_output(line)
            if temp:
                ret = float(temp)
    except subprocess.CalledProcessError:
        ret = -1.0
    return ret

def _get_gpu_load_by_tegra_stats():
    # Machines like Xavier do not have nvidia-smi.
    return -1.0

def _get_cpu_load():
    line = str(subprocess.check_output(["w"])).splitlines()[0]
    match = W_RGX.search(line)
    if match is None:
        return 100.0
    fields = match.expand(r"\g<load>").split(",")
    if len(fields) > 0:
        return float(fields[0])
    else:
        return 100.0


class LoadMonitor(object):
    def __init__(self):
        self.hostname = get_hostname()
        self.nproc = get_nproc()
        self.use_nvidia_smi = False
        self.check_nvidia_tooling()

    def check_nvidia_tooling(self):
        cmd = ["which", "nvidia-smi"]
        try:
            subprocess.check_output(cmd)
            self.use_nvidia_smi = True
        except subprocess.CalledProcessError:
            self.use_nvidia_smi = False

    def get_ipc_load(self):
        if self.use_nvidia_smi:
            gpu_load = _get_gpu_load_by_nvidia_smi()
        else:
            gpu_load = _get_gpu_load_by_tegra_stats()
        ret = {"hostname": self.hostname,
               "cpu_load": _get_cpu_load(),
               "nproc": self.nproc,
               "gpu_load": gpu_load}
        return json.dumps(ret)

    def run(self):
        # ROS topic/node name does not allow characters like -
        hostname = self.hostname.replace("-", "_")
        node_name = "LoadMonitor_" + hostname
        rospy.init_node(node_name)
        rospy.logwarn("Init %s", node_name)

        rate = rospy.Rate(1)  # FPS: 1

        ipc_load_topic = "/vehicle/report/{}/load".format(hostname)
        ipc_load_publisher = rospy.Publisher(ipc_load_topic, String, queue_size=1)
        rospy.logwarn("Publish data on %s", ipc_load_topic)
        while not rospy.is_shutdown():
            msg = String()
            msg.data = self.get_ipc_load()
            ipc_load_publisher.publish(msg)
            rate.sleep()
