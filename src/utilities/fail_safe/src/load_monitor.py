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

W_RGX = re.compile(r"load average: (?P<load1>[\.\d]+), (?P<load5>[\.\d]+), (?P<load15>[\.\d]+)")
NVIDIA_SMI_LOAD_RGX = re.compile(r".+[\s]+(?P<load>[\d]+)%[\s]+Default.+")
_DEFAULT_GPU_STATES = {
    "gpu_power_draw": -1.0,
    "gpu_temperature": -1,
    "gpu_load": -1.0,
    "gpu_memory_used": -1,
    "gpu_pstate": "NA"}

def get_hostname():
    """
    Return the hostname of the machine, i.e., /etc/hostname
    """
    return str(subprocess.check_output(["hostname"])).strip()


def get_nproc():
    """
    Return the number of cpu cores.
    """
    return int(subprocess.check_output(["nproc"]).decode("utf-8").strip())


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
    cmd = ["nvidia-smi",
           "--format=csv,noheader,nounits",
           "--query-gpu=power.draw,temperature.gpu,utilization.gpu,memory.used,pstate"]
    output = ""
    try:
        output = subprocess.check_output(cmd).decode("utf-8")
        output = output.splitlines()[-1]
    except subprocess.CalledProcessError:
        output = ""
    if not output:
        return _DEFAULT_GPU_STATES
    ret = _DEFAULT_GPU_STATES.copy()

    fields = output.split(",")
    if len(fields) == 5:
        ret["gpu_power_draw"] = float(fields[0].strip())
        ret["gpu_temperature"] = int(fields[1].strip())
        ret["gpu_load"] = float(fields[2].strip())
        ret["gpu_memory_used"] = int(fields[3].strip())
        ret["gpu_pstate"] = fields[4].strip()
    return ret


def _get_gpu_load_by_tegra_stats():
    # Machines like Xavier do not have nvidia-smi.
    return _DEFAULT_GPU_STATES


def _parse_cpu_load_output(text):
    match = W_RGX.search(text)
    if match is None:
        return "100.0"
    return match.expand(r"\g<load1>")


def _get_cpu_load():
    line = str(subprocess.check_output(["uptime"]))
    return float(_parse_cpu_load_output(line))


class LoadMonitor(object):
    def __init__(self):
        self.hostname = get_hostname()
        nproc = get_nproc()
        if self.hostname == "xavier":
            self.cpu_load_threshold = 14.0  # empirical
        else:
            self.cpu_load_threshold = float(nproc)

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
               "cpu_load_threshold": self.cpu_load_threshold}
        for key in gpu_load:
            ret[key] = gpu_load[key]
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
        rospy.logwarn("%s: cpu_load_threshold to %f", self.hostname, self.cpu_load_threshold)
        while not rospy.is_shutdown():
            msg = String()
            msg.data = self.get_ipc_load()
            ipc_load_publisher.publish(msg)
            rate.sleep()
