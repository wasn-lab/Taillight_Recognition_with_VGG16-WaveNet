#!/usr/bin/env python
import rospy, rospkg
import rosnode # For pinging node
import sys
import time
import yaml, json
# File operations
import datetime
# Args
import argparse

from std_msgs.msg import (
    Empty,
    Bool,
    String,
)

this_node_name = "node_tracer"



def main(sys_args):
    global this_node_name

    rospy.init_node(this_node_name, anonymous=True)
    #
    this_node_name = rospy.get_name() # Including the '/'
    print("this_node_name = %s" % this_node_name)
    #
    # _pack_path = ".."
    rospack = rospkg.RosPack()
    _pack_path = rospack.get_path('issue_reporter')
    print("_pack_path = %s" % _pack_path)
    # Loading parameters
    #---------------------------------------------#
    f_path = _pack_path + "/params/"

    # Manual mode
    f_name_params = "setting_node_tracer.yaml"
    f_name_nodes = "wateched_nodes.txt"


    # Read param file
    #------------------------#
    _f = open( (f_path+f_name_params),'r')
    params_raw = _f.read()
    _f.close()
    param_dict = yaml.load(params_raw)
    #------------------------#

    # Read node_list file
    #------------------------#
    node_list = []
    _f = open( (f_path+f_name_nodes),'r')
    for _s in _f:
        # Remove the space and '\n'
        _s1 = _s.rstrip().lstrip()
        # Deal with coments
        _idx_comment = _s1.find('#')
        if _idx_comment >= 0: # Do find a '#'
            _s1 = _s1[:_idx_comment].rstrip() # Remove the comment parts
        if len(_s1) > 0: # Append non-empty string (after stripping)
            if _s1[0] != '/':
                _s1 = '/' + _s1
            node_list.append(_s1)
    _f.close()
    #------------------------#


    # Print the params
    # print("param_dict = %s" % str(param_dict))
    print("\nsettings (in json format):\n%s" % json.dumps(param_dict, indent=4))
    print("\n\ntopic_list:\n---------------" )
    for _nd in node_list:
        print(_nd)
    print("---------------\n\n" )


    # Add the 'topics' to param_dict
    param_dict['nodes'] = node_list


    # test, the param_dict after combination
    # print("param_dict = %s" % str(param_dict))
    # print("param_dict (in json format):\n%s" % json.dumps(param_dict, indent=4))
    #---------------------------------------------#

    # Init ROS communication interface
    #--------------------------------------#
    # Subscriber
    # Publisher
    _node_all_alive_pub = rospy.Publisher("/node_trace/all_alive", Bool, queue_size=1, latch=True) #
    _node_pinged_pub = rospy.Publisher("/node_trace/pinged", String, queue_size=1, latch=True) #
    _node_unpinged_pub = rospy.Publisher("/node_trace/unpinged", String, queue_size=1, latch=True) #
    _node_alive_pub = rospy.Publisher("/node_trace/alive", String, queue_size=1, latch=True) #
    _node_closed_pub = rospy.Publisher("/node_trace/closed", String, queue_size=1, latch=True) #
    _node_untraced_pub = rospy.Publisher("/node_trace/untraced", String, queue_size=1, latch=True) #
    _node_zombi_traced_pub = rospy.Publisher("/node_trace/zombi_traced", String, queue_size=1, latch=True) #
    #--------------------------------------#



    node_dict = dict()
    node_dict["pinged"] = list()
    node_dict["unpinged"] = list()
    node_dict["alive"] = list()
    node_dict["closed"] = list()
    node_dict["untraced"] = list()
    node_dict["zombi_traced"] = list() # Traced node that was died abruptly without unregistration
    # Loop for ping
    rate = rospy.Rate(1.0) # 10hz
    while not rospy.is_shutdown():
        _t1 = time.clock()
        try:
            pinged, unpinged = rosnode.rosnode_ping_all(True)
        except:
            print("Error while pinging all node")
            time.sleep(1.0)
            continue
        #
        _t2 = time.clock()
        print("elapse time = %s ms" % str( (_t2 - _t1)*1000.0 ))
        # Remove this node
        if this_node_name in pinged:
            pinged.remove(this_node_name)
        #
        node_dict["pinged"] = pinged
        node_dict["unpinged"] = unpinged
        #
        node_dict["alive"] = list()
        # node_dict["closed"] = list()
        node_dict["untraced"] = list()
        node_dict["zombi_traced"] = list()
        #
        _closed = list()
        #
        for _nd in pinged:
            if _nd in node_list:
                node_dict["alive"].append(_nd)
            else:
                node_dict["untraced"].append(_nd)
        for _nd in node_list:
            if not _nd in pinged:
                # node_dict["closed"].append(_nd)
                _closed.append(_nd)
                if not _nd in node_dict["closed"]:
                    print("Node <%s> closed." % _nd)
            if _nd in unpinged:
                node_dict["zombi_traced"].append(_nd)
        #
        node_dict["closed"] = _closed
        #
        print("pinged =\n%s" % str(node_dict["pinged"]) )
        print("unpinged =\n%s" % str(node_dict["unpinged"]) )
        print("alive =\n%s" % str(node_dict["alive"]) )
        print("closed =\n%s" % str(node_dict["closed"]) )
        print("untraced =\n%s" % str(node_dict["untraced"]) )
        print("zombi_traced =\n%s" % str(node_dict["zombi_traced"]) )
        #
        if len(node_dict["closed"]) > 0:
            _node_all_alive_pub.publish(False)
        else:
            _node_all_alive_pub.publish(True)
        #
        _node_pinged_pub.publish( str(node_dict["pinged"]) )
        _node_unpinged_pub.publish( str(node_dict["unpinged"]) )
        _node_alive_pub.publish( str(node_dict["alive"]) )
        _node_closed_pub.publish( str(node_dict["closed"]) )
        _node_untraced_pub.publish( str(node_dict["untraced"]) )
        _node_zombi_traced_pub.publish( str(node_dict["zombi_traced"]) )
        #
        rate.sleep()
        # time.sleep(1.0)
    #
    rospy.logwarn("[node_tracer] The node_tracer is going to close.")
    # _node_all_alive_pub.publish(False)
    time.sleep(0.5)
    print("[node_tracer] Leave main()")




if __name__ == '__main__':

    try:
        main(sys.argv)
    except rospy.ROSInterruptException:
        pass
    print("[node_tracer] Closed.")
